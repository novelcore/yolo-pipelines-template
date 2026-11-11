"""Config-validation service: pre-GPU liveness checks over the resolved config.

This is the pipeline's first *developer* step. Its job is to fail fast — before
any expensive GPU step starts — if the run's external dependencies are missing.

What it does NOT do: re-validate the config schema. In this pipeline the config
tree is composed and TYPE/ENUM-validated upstream by the platform (the
Hydra structured-config layer). Re-implementing a parallel schema here would
just be a second source of truth to drift. So this step reads the already-
resolved params.yaml and checks that the world it references actually exists:

  - pretrained weights: an s3:// weights override actually exists;
  - checkpoint resume : when the config resumes, the checkpoint is present.

The dataset is not checked here: the platform pins every dataset of the run to one
commit before the first step starts, and the training step reads it.

Endpoints + credentials come from the ENV the platform injects into this step
(LAKEFS_ENDPOINT / LAKEFS_ACCESS_KEY / LAKEFS_SECRET_KEY, or the AWS_* chain for
source=s3). The developer wires none of that. A check that must run (an s3://
weights path, a checkpoint to resume) and finds no credentials fails the step.
Only when no platform environment is present at all (a laptop) are the object-
store checks skipped, and the log says so.
"""

import logging
import os
from urllib.parse import urlparse

# Any of these means the platform injected an object store: its absence is a laptop.
_STORE_ENV = ("LAKEFS_ENDPOINT", "LAKEFS_ACCESS_KEY", "LAKEFS_SECRET_KEY", "AWS_ACCESS_KEY_ID",
              "AWS_SECRET_ACCESS_KEY", "S3_ENDPOINT_URL")

# Bounds of quantization.calibration_frames. They mirror QuantizationParams / QATParams
# (steps/model_quantization, steps/qat_finetune: Field(ge=100, le=10000)); this image has no
# pydantic, so step_tests/test_config_validation_bounds.py pins these to the models.
CALIBRATION_FRAMES_MIN = 100
CALIBRATION_FRAMES_MAX = 10000


class ConfigValidationError(Exception):
    """A required external dependency is missing or unreachable."""


class ConfigValidationService:
    def __init__(self, timeout: int = 15, max_retries: int = 3) -> None:
        self._timeout = timeout
        self._max_retries = max_retries
        self._log = logging.getLogger(__name__)

    # ------------------------------------------------------------------ public

    def run(self, resolved: dict) -> None:
        """Run all liveness checks over the resolved params.yaml dict.

        Raises ConfigValidationError on the first missing/unreachable
        dependency. Network checks self-skip when their endpoint/credentials
        were not injected, so this is always safe to call.
        """
        data = resolved.get("data") or {}
        model = resolved.get("model") or {}
        train = resolved.get("train") or {}
        platform = resolved.get("platform") or {}

        self._check_quantization(resolved.get("quantization") or {})

        resume_from = str((train.get("checkpointing") or {}).get("resume_from") or "").strip()
        if not (model.get("pretrained_weights") or resume_from):
            self._log.info("config-validation: nothing to look up in the object store")
            return
        s3 = self._s3_client(str(data.get("source") or "lakefs"))
        if s3 is None:
            self._log.info("object-store checks skipped: no platform environment (running locally)")
            return
        self._check_pretrained_weights(s3, model)
        self._check_checkpoint_resume(s3, resume_from, platform)
        self._log.info("config-validation: all liveness checks passed")

    # ------------------------------------------------------------------ quantization

    @staticmethod
    def _check_quantization(quant: dict) -> None:
        """Fail now, before training, on a calibration_frames the INT8 steps would reject.

        Only when the INT8 steps run (mode ptq or qat): otherwise the value is not used.
        """
        if str(quant.get("mode") or "none").lower() == "none":
            return
        raw = quant.get("calibration_frames", 512)
        try:
            frames = int(str(raw).strip())
        except ValueError:
            raise ConfigValidationError(
                f"quantization.calibration_frames must be a whole number, got {raw!r}"
            ) from None
        if not CALIBRATION_FRAMES_MIN <= frames <= CALIBRATION_FRAMES_MAX:
            raise ConfigValidationError(
                f"quantization.calibration_frames must be between {CALIBRATION_FRAMES_MIN} and "
                f"{CALIBRATION_FRAMES_MAX}, got {frames} (quantization-calibration_frames on the run form)"
            )

    # ------------------------------------------------------------------ s3/lakefs

    def _s3_client(self, source: str):
        """Build a boto3 S3 client from the platform-injected env.

        Returns None only when no platform environment is present at all (a laptop,
        → caller skips). A platform that injected some of it but not what this check
        needs is an error: the check must run.
        """
        if not any(os.environ.get(v) for v in _STORE_ENV):
            return None
        import boto3
        from botocore.config import Config as BotoConfig

        cfg = BotoConfig(
            retries={"max_attempts": self._max_retries, "mode": "adaptive"},
            connect_timeout=self._timeout, read_timeout=self._timeout,
        )

        if source == "lakefs":
            endpoint = os.environ.get("LAKEFS_ENDPOINT")
            if not endpoint:
                raise ConfigValidationError(
                    "LAKEFS_ENDPOINT is not set, so the weights / checkpoint cannot be checked")
            return boto3.client(
                "s3", endpoint_url=endpoint,
                aws_access_key_id=os.environ.get("LAKEFS_ACCESS_KEY"),
                aws_secret_access_key=os.environ.get("LAKEFS_SECRET_KEY"),
                config=cfg,
            )

        # source == s3 (AWS or a custom endpoint such as MinIO)
        if not (os.environ.get("AWS_ACCESS_KEY_ID") or os.environ.get("S3_ENDPOINT_URL")):
            raise ConfigValidationError(
                "no AWS/S3 credentials are set, so the weights / checkpoint cannot be checked")
        return boto3.client(
            "s3", endpoint_url=os.environ.get("S3_ENDPOINT_URL"),
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
            region_name=os.environ.get("AWS_DEFAULT_REGION"),
            config=cfg,
        )

    def _check_pretrained_weights(self, s3, model: dict) -> None:
        weights = model.get("pretrained_weights")
        if not weights:
            return  # standard yolo names ship in the training image
        if not str(weights).startswith("s3://"):
            raise ConfigValidationError(
                f"model.pretrained_weights must be null or an s3:// path, got: {weights!r}"
            )
        parsed = urlparse(weights)
        self._log.info("checking pretrained weights %s", weights)
        try:
            s3.head_object(Bucket=parsed.netloc, Key=parsed.path.lstrip("/"))
        except Exception as exc:  # botocore ClientError 404 etc.
            raise ConfigValidationError(
                f"pretrained weights not found: {weights} ({exc})"
            ) from exc
        self._log.info("pretrained weights OK")

    def _check_checkpoint_resume(self, s3, resume_from: str, platform: dict) -> None:
        """When the config resumes training (``train.checkpointing.resume_from``), prove the checkpoint exists.

        The checkpoint bucket/prefix are platform facts
        (platform.checkpoints, injected from the project lakeFS repo).
        """
        if not resume_from:
            return

        checkpoints = platform.get("checkpoints") or {}
        if resume_from == "auto":
            bucket = str(checkpoints.get("bucket") or "")
            prefix = str(checkpoints.get("prefix") or "").rstrip("/") + "/"
            if not bucket:
                raise ConfigValidationError(
                    "resume_from='auto' but platform.checkpoints.bucket is unset"
                )
            self._log.info("scanning for checkpoints under s3://%s/%s", bucket, prefix)
            resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=100)
            pt = [o for o in resp.get("Contents", []) if o["Key"].endswith(".pt")]
            if not pt:
                raise ConfigValidationError(
                    f"resume_from='auto' but no .pt checkpoint under s3://{bucket}/{prefix}"
                )
            self._log.info("found %d checkpoint(s) for auto-resume", len(pt))
        else:
            parsed = urlparse(resume_from)
            self._log.info("checking checkpoint %s", resume_from)
            try:
                s3.head_object(Bucket=parsed.netloc, Key=parsed.path.lstrip("/"))
            except Exception as exc:
                raise ConfigValidationError(
                    f"checkpoint not found: {resume_from} ({exc})"
                ) from exc
            self._log.info("checkpoint OK")
