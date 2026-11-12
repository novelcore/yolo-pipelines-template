"""Config-validation service: pre-GPU checks over the run's settings.

The pipeline's first step. Its job is to fail fast, before any expensive step starts, on a setting that
a later step would reject or a file a later step would look for and not find.

What it does NOT do: re-validate the config schema. The config tree is composed and TYPE/ENUM-validated
upstream by the platform. Re-implementing a parallel schema here would just be a second source of truth
to drift. It checks:

  - quantization: ``calibration_frames`` is inside the bounds the INT8 steps accept;
  - pretrained weights: an ``s3://<repository>/<ref>/<path>`` weights override exists in lakeFS.

The dataset is not checked here: the platform pins every dataset of the run to one commit before the
first step starts, and ``prepare-data`` reads it.
"""

import logging
from urllib.parse import urlparse

# Bounds of quantization.calibration_frames. They mirror QuantizationParams / QATParams
# (steps/model_quantization, steps/qat_finetune: Field(ge=100, le=10000)); this image has no
# pydantic, so step_tests/test_form_bounds.py pins these to the models.
CALIBRATION_FRAMES_MIN = 100
CALIBRATION_FRAMES_MAX = 10000


class ConfigValidationError(Exception):
    """A setting is out of bounds, or a file it names does not exist."""


class ConfigValidationService:
    def __init__(self) -> None:
        self._log = logging.getLogger(__name__)

    def run(self, resolved: dict) -> None:
        """Run all checks over the composed settings; raises ConfigValidationError on the first failure."""
        self._check_quantization(resolved.get("quantization") or {})
        self._check_pretrained_weights((resolved.get("model") or {}).get("pretrained_weights"))
        self._log.info("config-validation: all checks passed")

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

    def _check_pretrained_weights(self, weights) -> None:
        if not weights:
            return  # standard yolo names are fetched by the training step
        if not str(weights).startswith("s3://"):
            raise ConfigValidationError(
                f"model.pretrained_weights must be empty or an s3://<repository>/<ref>/<path> path, got: {weights!r}"
            )
        parsed = urlparse(weights)
        ref, _, path = parsed.path.lstrip("/").partition("/")
        self._log.info("checking pretrained weights %s", weights)
        from kaos_lib.lakefs import LakeFS  # noqa: PLC0415

        try:
            found = any(obj["path"] == path for obj in LakeFS.connect().list_objects(parsed.netloc, ref, path))
        except Exception as exc:  # the repository or the ref does not exist, or lakeFS cannot be reached
            raise ConfigValidationError(f"pretrained weights not found: {weights} ({exc})") from exc
        if not found:
            raise ConfigValidationError(f"pretrained weights not found: {weights}")
        self._log.info("pretrained weights OK")
