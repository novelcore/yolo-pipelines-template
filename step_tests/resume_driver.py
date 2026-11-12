"""Run model-training once, the way `test_training_run.test_a_continued_step_...` needs it, in a process of its own.

    python resume_driver.py <first|second> <work folder> <prepared dataset> <starting weights>

`first` asks the step to stop during its first epoch (the signal a reclaimed node sends): kaos-lib saves the checkpoint
folder and the step ends with exit code 85. `second` is the step run again against the same checkpoint folder
(KAOS_CHECKPOINT_DIR, set by the caller): it prints how many epochs the continued training ended on.
"""
import signal
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "step_tests"), str(ROOT / "steps" / "model_training")]

import ultralytics  # noqa: E402
from conftest import composed_settings  # noqa: E402
from kaos_lib import testing  # noqa: E402

from app import entry  # noqa: E402
from app.services.model_training import TrainingService  # noqa: E402

mode, work, prepared, weights = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
ultralytics.settings.update({"mlflow": False, "sync": False})
entry.tracking.mlflow = lambda *a, **k: None

if mode == "first":
    original, sent = TrainingService._make_batch_end_callback, []

    def asking_to_stop(epoch_metrics):
        inner = original(epoch_metrics)

        def callback(trainer):
            inner(trainer)
            if not sent:
                sent.append(1)
                signal.raise_signal(signal.SIGUSR1)  # the stop request: the step is saved at the end of this epoch

        return callback

    TrainingService._make_batch_end_callback = staticmethod(asking_to_stop)

settings = composed_settings()
settings["model"]["pretrained_weights"] = weights
settings["train"].update(epochs=2, batch_size=2, image_size=64, close_mosaic=0, amp=False)
with testing.running(work, kind="train", name="model-training", experiment="toy", params=settings,
                     inputs={"prepared": prepared}, produces=[("model", "checkpoints")]) as run:
    entry.main()
print("EPOCHS", run.result["values"]["epochs_completed"])
