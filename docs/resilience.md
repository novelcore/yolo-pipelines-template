# 8. When a machine is lost or a time limit hits

**Goal:** know what the platform does for you when a step is interrupted, so a long run is not lost and your code
needs no special handling.

A step can be interrupted in four ways: its machine is reclaimed (spot or preemptible nodes), the machine fails or is
drained for maintenance, an HPC (Slurm) job reaches its time limit, or the step hangs. Pipelines behave the same on
every kind of machine: your project's own cluster (GPU or CPU, spot included) and HPC.

## What happens automatically

| what happened | a step that writes checkpoints | any other step |
|---|---|---|
| machine lost, reclaimed or drained | runs again and **continues from its last checkpoint** | runs again from the start, at most twice |
| HPC time limit reached | continues from the last checkpoint in a new job (up to 5 continuations) | ends with a message that names the limit |
| the step hangs (HPC: no GPU use and no output for 20 minutes) | ends with the reason `Idle`; with a checkpoint it continues | ends with the reason `Idle` |
| your own code fails (a non-zero exit, out of memory, a Python error) | ends as failed: **not** repeated | ends as failed: **not** repeated |

Running a step again is always safe: a step's results are published only when it succeeds.

Nothing in your code or in `pipeline.py` is needed for this. The platform also decides **by itself** that a step is
resumable once the evidence is there: an earlier continuation loaded its checkpoint and got further than it. You can
declare it earlier (`resumable=`) or opt out (`resumable=False`), see [Settings](#settings-all-optional).

## Checkpoints without code

Every step gets a folder in the environment variable `KAOS_CHECKPOINT_DIR`. The platform keeps a copy of that folder
safe while the step runs (every 10 minutes by default, and when the machine is being taken away) and puts the newest
**verified** copy back before the step starts again. A copy that does not match its checksum is never used; the
previous good one is.

This template's steps import `kaos_lib`, which also hooks the training framework, with nothing for you to call:

| framework | saved | on the repeated step |
|---|---|---|
| Ultralytics (YOLO), used by `model-training` | `last.pt` at the end of each epoch, and `best.pt` once one exists | training continues, same total epochs, same MLflow run; `best.pt` is put back, so the best checkpoint is never lost across a continuation |
| your own loop, through `kaos_lib.checkpoint`, used by `qat-finetune` | the model, optimizer and finished epochs at the end of each epoch | the fine-tune continues from the last finished epoch, same total epochs, same MLflow run |

If a run still ends with no `best.pt` (no epoch after the restart beat the earlier best), `model-training` hands `last.pt` on as the checkpoint for the next steps; a run with no weights at all fails. `qat-finetune` and `model-quantization` refuse an empty or unloadable checkpoint with a message that says so.

`qat-finetune` is a hand-written PyTorch loop, so it uses the second way. It calls `checkpoint.load("qat")` at the start
and `checkpoint.save(..., "qat")` plus `checkpoint.progress({"epoch": n})` after each epoch. When the platform asks the
step to stop (a reclaimed node, a drain, the time limit), it ends the epoch that was running without saving it and exits
so the step continues from the last finished epoch. `model-quantization` is a single short export with nothing to
continue from: a repeat simply runs it again.

If you add a step of your own with plain PyTorch, use the same few calls as `qat-finetune`:

```python
import io, torch
from kaos_lib import checkpoint

raw = checkpoint.load("state")                                   # None on a first run
state = torch.load(io.BytesIO(raw), weights_only=True) if raw else {"epoch": 0}
# ... restore the model and optimizer from `state`, then:
for epoch in range(state["epoch"], epochs):
    train_one_epoch()
    buffer = io.BytesIO()
    torch.save({"epoch": epoch + 1, "model": model.state_dict(), "optimizer": optimizer.state_dict()}, buffer)
    checkpoint.save(buffer.getvalue(), "state")                  # written atomically
    checkpoint.progress({"epoch": epoch + 1})                    # lets the platform see the step moves forward
    if checkpoint.stop_requested():                              # node being taken away, or the time limit
        checkpoint.exit_for_restart()                            # exit code 85: "saved, continue me"
```

Saving the weights with `torch.save` into bytes keeps one file that loads on any kind of machine. If you keep a scheduler
or mixed-precision scaler, put their `state_dict()` in the same file. A model prepared for quantization-aware training
needs its `state_dict()` saved as it is (not copied key by key): it carries version information that its quantization
observers read when loading. This is the only resume mechanism: **Retry** in the run view uses it too. On your laptop `KAOS_CHECKPOINT_DIR` is not set and
none of this happens.

!!! note "Large checkpoints and disk"
    The folder lives on the step's machine disk. A big checkpoint (several GB) needs room: raise `disk=` for the step in
    `pipeline.py` (for example `disk="30Gi"`, as `qat-finetune` does with `disk="20Gi"`). Only the two newest copies are kept, and all are deleted when the step
    succeeds.

## Settings (all optional)

Set them on the step in `pipeline.py`:

```python
train = step("model-training", gpu=True, needs=[validate],
             hpc_time_limit="12h",       # HPC (Slurm) time limit for this step
             checkpoint_every="20m",     # copy the checkpoint folder every 20 minutes (default: 10, adapts to size)
             idle_timeout="40m")         # raise the 20-minute idle rule; 0 turns it off
```

| setting | meaning |
|---|---|
| `hpc_time_limit` | the longest the step may run on HPC (GPU steps). It is a ceiling: the platform may ask the site for less after it has seen the step finish a few times, and runs it once more at your limit if it was cut at the shorter one, so a step is never stopped where your limit would have passed. You never name a queue. A limit longer than the longest HPC run is continued in capped runs if the step is resumable, and refused before it is sent (`TooLongNotResumable`) if not: [Run on HPC (Slurm)](slurm.md). |
| `checkpoint_every` | how often the folder is copied. Saving stays near 10 % of the run time: a slow save lengthens the interval. `checkpoint_every=False` turns copies off for the step (the repeat after a lost machine stays). A CPU step is copied only if you set a value here. |
| `idle_timeout` | how long a step may show no GPU use and no new log lines before it is ended as `Idle` (20 minutes by default; `0` turns the rule off). Raise it for a step with a long silent phase. It applies to GPU steps on your cluster and on HPC. |
| `resumable` | a GPU step is continued when it writes into the checkpoint folder, and the platform also learns it from evidence. Declare it to be sure: `resumable=True` or a lead time such as `resumable="3m"`, which is how long before the HPC time limit the step is told to save (default 10 minutes; use a longer lead for a large checkpoint). A step declared or learned as resumable is the one a limit above the longest run is continued for. `resumable=False` opts the step out: on HPC it is never continued from a checkpoint (a lost machine still re-runs it), and a step longer than the longest run is then refused. |

## What you will see

In the step's log:

| message | meaning |
|---|---|
| `[kaos-checkpoint] continuing from the checkpoint saved at 12:03:11 UTC (3 files)` | this is a repeat; training picks up from that copy |
| `[kaos-checkpoint] no saved checkpoint for this step: starting from the beginning` | a first run, or nothing had been saved yet |
| `[kaos-checkpoint] saved a checkpoint copy: 3 files, 212.4 MB, 4 s` | a regular copy |
| `[kaos-checkpoint] skipping checkpoint ...: the copy does not match its checksum` | a damaged copy was ignored; the previous one is used |
| `[kaos-checkpoint] could not save a checkpoint copy (...); the step is not affected` | the copy failed; the step carries on and tries again later |
| `[kaos-checkpoint] step finished; its checkpoint copies were removed` | the step succeeded and the copies were cleaned up |

In the run view, a repeated step shows its attempts one after the other (`model-training(1)`, `model-training(2)`).
On HPC the step's status line says why a job was repeated, for example that it was stopped for no GPU use and no output
for 20 minutes, or that a continuation did not get past its checkpoint (`NoProgress`): then the step ends instead of
repeating the same hours.

## If it isn't there

| symptom | fix |
|---|---|
| a repeated step starts from epoch 0 | the step wrote nothing to `KAOS_CHECKPOINT_DIR` and no framework hook applies: save with `checkpoint.save` and load with `checkpoint.load` |
| the step ends as failed after its machine was reclaimed | the step ran out of its two repeats; run it again, or pick an on-demand class in `model-training-class` |
| `NoProgress` | the step restarted without loading its checkpoint, or saved the same state again: check that loading restores your epoch counter |
| copies fail with a disk message | raise `disk=` for the step |
| the step was ended as `Idle` but was working | raise `idle_timeout` for it |
| a CPU step is not copied | set `checkpoint_every` on it (short CPU steps are simply repeated) |
