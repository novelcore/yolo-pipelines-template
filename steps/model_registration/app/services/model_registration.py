import logging
import re
import time
from collections.abc import Callable
from datetime import datetime, timezone
from importlib.metadata import version as package_version
from typing import Any, Optional

import mlflow
from mlflow.tracking import MlflowClient

from app.models.registration import RegistrationParams, RegistrationResult


def kaos_lib_version() -> str:
    """The installed kaos-lib version."""
    return package_version("kaos-lib")


class ModelRegistrationError(Exception):
    """Raised when model registration fails unrecoverably."""


def lakefs_tag_id(registered_model_name: str, version: int) -> str:
    """``model-<slug(name)>-v<N>``: the lakeFS tag of version N (slug keeps ``[A-Za-z0-9_-]``)."""
    return f"model-{re.sub(r'[^A-Za-z0-9_-]', '-', registered_model_name)}-v{version}"


class ModelRegistrationService:
    """Registers YOLO model checkpoints in the MLflow model registry.

    Parameters
    ----------
    max_retries:
        Number of attempts before giving up on an MLflow call.
    """

    _RETRY_DELAYS = (1, 2, 4)

    def __init__(self, max_retries: int = 3) -> None:
        self._max_retries = max_retries
        self._logger = logging.getLogger(__name__)

    def run(self, params: RegistrationParams) -> RegistrationResult:
        """Execute the full registration pipeline.

        1. Connect to MLflow.
        2. Show best.pt in the experiment's Models tab: a logged model of the training run
           whose weights stay in lakeFS, and register best.pt under params.registered_model_name
           from it (the version links to the run and the logged model).
        3. Set lineage tags on the best.pt version.
        3b. Tag the training run's outputs commit in lakeFS and record the tag on the version.
        4. Optionally register last.pt and tag it.
        5. Optionally assign a registry alias (champion/challenger) to the best.pt version.
        6. Return a RegistrationResult.
        """
        client = MlflowClient()

        self._logger.info(
            "Registering %s from %s under model '%s'",
            params.checkpoint,
            self._uri(params, params.checkpoint),
            params.registered_model_name,
        )
        best_version = self._register_checkpoint(
            client=client,
            model_uri=self._log_model(client, params),
            registered_model_name=params.registered_model_name,
        )
        self._logger.info("Registered best.pt as version %s", best_version)

        self._set_lineage_tags(
            client=client,
            registered_model_name=params.registered_model_name,
            version=best_version,
            params=params,
            checkpoint_type="best",
        )
        self._link_outputs(
            client=client,
            registered_model_name=params.registered_model_name,
            version=best_version,
            params=params,
        )

        last_version: Optional[int] = None
        if params.checkpoint == "best.pt":  # no last.pt of its own when best.pt stands in for it
            last_uri = self._uri(params, "last.pt")
            self._logger.info("Registering last.pt from %s", last_uri)
            last_version = self._register_checkpoint(
                client=client,
                model_uri=last_uri,
                registered_model_name=params.registered_model_name,
            )
            self._logger.info("Registered last.pt as version %s", last_version)
            self._set_lineage_tags(
                client=client,
                registered_model_name=params.registered_model_name,
                version=last_version,
                params=params,
                checkpoint_type="last",
            )

        promoted_to: Optional[str] = None
        if params.promote_to:
            self._with_retry(
                lambda: client.set_registered_model_alias(
                    name=params.registered_model_name,
                    alias=params.promote_to,
                    version=str(best_version),
                )
            )
            promoted_to = params.promote_to
            self._logger.info(
                "Assigned alias '%s' to version %s", promoted_to, best_version
            )

        return RegistrationResult(
            registered_model_name=params.registered_model_name,
            best_version=best_version,
            last_version=last_version,
            registered_at=datetime.now(timezone.utc),
            promoted_to=promoted_to,
        )

    @staticmethod
    def _uri(params: RegistrationParams, name: str) -> str:
        """``s3://<repository>/<commit>/<path><name>``: a file of the training results, pinned to its commit."""
        return f"s3://{params.outputs_repo}/{params.outputs_commit}/{params.outputs_path}{name}"

    def _log_model(self, client: MlflowClient, params: RegistrationParams) -> str:
        """Show best.pt in the experiment's Models tab and return ``models:/<id>`` to register from.

        An MLflow 3 logged model ``<experiment>-best`` whose weights stay in lakeFS
        (``mlflow.create_external_model``), with the training run as its source, the training
        settings as params, the ``kaos.*`` lakeFS tags and the final val metrics. The run gets
        ``kaos.logged_model_id``. A retried step reuses the model this run already has.
        """
        name = f"{params.experiment_name}-best"
        tags = {
            "kaos.outputs_repo": params.outputs_repo,
            "kaos.outputs_commit": params.outputs_commit,
            "kaos.outputs_path": params.outputs_path,
            "kaos.workflow": params.run_name,
        }
        if params.dataset_version:
            tags["kaos.dataset_ref"] = params.dataset_version
        run_id = params.mlflow_run_id
        experiment_id = self._with_retry(lambda: client.get_run(run_id)).info.experiment_id
        existing = self._with_retry(lambda: client.search_logged_models(
            experiment_ids=[experiment_id],
            filter_string=f"source_run_id = '{run_id}' AND name = '{name}'",
            max_results=1,
        ))
        if existing:
            model_id = existing[0].model_id
            self._with_retry(lambda: client.set_logged_model_tags(model_id, tags))
            self._logger.info("Run %s already has logged model %s (%s)", run_id, name, model_id)
        else:
            model_id = self._with_retry(lambda: mlflow.create_external_model(
                name=name,
                source_run_id=run_id,
                experiment_id=experiment_id,
                params={"epochs": str(params.epochs), "imgsz": str(params.image_size),
                        "model_variant": str(params.model_variant)},
                tags=tags,
            )).model_id
            self._logger.info("Logged model %s (%s) for run %s", name, model_id, run_id)
        # step = the last epoch, where the run already has these values: the run's charts get no stray point.
        for key, value in params.val_metrics.items():
            self._with_retry(lambda k=key, v=value: client.log_metric(run_id, k, v, step=params.epochs,
                                                                      model_id=model_id))
        for key, value in {**tags, "kaos.logged_model_id": model_id}.items():
            self._with_retry(lambda k=key, v=value: client.set_tag(run_id, k, v))
        return f"models:/{model_id}"

    def _register_checkpoint(
        self,
        client: MlflowClient,
        model_uri: str,
        registered_model_name: str,
    ) -> int:
        """Register a single checkpoint and return the integer version number."""
        mv = self._with_retry(
            lambda: mlflow.register_model(
                model_uri=model_uri,
                name=registered_model_name,
            )
        )
        return int(mv.version)

    def _set_lineage_tags(
        self,
        client: MlflowClient,
        registered_model_name: str,
        version: int,
        params: RegistrationParams,
        checkpoint_type: str,
    ) -> None:
        """Apply lineage and checkpoint-type tags to a registered model version."""
        # kaos_lib.version: the kaos-lib package this image runs (novelcore/kaos#1147).
        tags: dict[str, str] = {"checkpoint_type": checkpoint_type, "kaos_lib.version": kaos_lib_version()}

        if params.mlflow_run_id:
            tags["training_run_id"] = params.mlflow_run_id
        if params.dataset_version is not None:
            tags["dataset_version"] = params.dataset_version
        if params.dataset_commit is not None:
            tags["dataset_commit"] = params.dataset_commit
        if params.dataset_sample_size is not None:
            tags["dataset_sample_size"] = str(params.dataset_sample_size)
        if params.prepared_dataset is not None:
            tags["prepared_dataset"] = params.prepared_dataset
        if params.model_variant is not None:
            tags["model_variant"] = params.model_variant
        if params.best_map50 is not None:
            tags["best_mAP50"] = str(params.best_map50)
        if params.exported_models:
            for label, uri in params.exported_models.items():
                tags[f"export_{label}"] = uri

        version_str = str(version)
        for key, value in tags.items():

            def _set_tag(k: str = key, v: str = value) -> None:
                client.set_model_version_tag(
                    name=registered_model_name,
                    version=version_str,
                    key=k,
                    value=v,
                )

            self._with_retry(_set_tag)

    def _link_outputs(
        self,
        client: MlflowClient,
        registered_model_name: str,
        version: int,
        params: RegistrationParams,
    ) -> None:
        """Tag the training results' commit in ``outputs-<app>`` as ``model-<slug>-v<N>``
        (kaos#1283) and set ``kaos.*`` tags on the model version.

        An existing tag on the same commit is a success; on another commit it is an error.
        """
        repo, commit = params.outputs_repo, params.outputs_commit

        from kaos_lib.errors import NotFoundError  # noqa: PLC0415
        from kaos_lib.lakefs import LakeFS  # noqa: PLC0415

        tag = lakefs_tag_id(registered_model_name, version)
        lake = LakeFS.connect()
        try:
            existing = lake.tag_commit(repo, tag)
        except NotFoundError:
            lake.create_tag(repo, tag, commit)
            self._logger.info("Tagged %s@%s as %s", repo, commit, tag)
        else:
            if existing != commit:
                raise ModelRegistrationError(
                    f"lakeFS tag {tag} in {repo} already points at {existing}, not at the "
                    f"training commit {commit}; refusing to move it"
                )
            self._logger.info("lakeFS tag %s in %s already on %s", tag, repo, commit)

        tags = {
            "kaos.outputs_repo": repo,
            "kaos.outputs_commit": commit,
            "kaos.lakefs_tag": tag,
            "kaos.workflow": params.run_name,
        }
        for key, value in tags.items():

            def _set_tag(k: str = key, v: str = value) -> None:
                client.set_model_version_tag(
                    name=registered_model_name,
                    version=str(version),
                    key=k,
                    value=v,
                )

            self._with_retry(_set_tag)

    def _with_retry(self, fn: Callable[[], Any]) -> Any:
        """Call fn with exponential backoff, raising ModelRegistrationError on exhaustion."""
        delays = list(self._RETRY_DELAYS[: self._max_retries - 1]) + [None]
        last_exc: Optional[Exception] = None

        for attempt, delay in enumerate(delays, start=1):
            try:
                return fn()
            except Exception as exc:
                last_exc = exc
                if delay is None:
                    break
                self._logger.warning(
                    "MLflow call failed (attempt %d/%d): %s — retrying in %ds",
                    attempt,
                    self._max_retries,
                    exc,
                    delay,
                )
                time.sleep(delay)

        raise ModelRegistrationError(
            f"MLflow call failed after {self._max_retries} attempts: {last_exc}"
        ) from last_exc
