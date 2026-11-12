# Run on HPC (Slurm)

Training can run on a Slurm site when your project is connected to one. Where a step runs, the time limit, the queue and
the reasons a job waits are explained in the guide: [HPC (Slurm)](https://storage.googleapis.com/kaos-dev-511509-kaos-lib/docs/hpc.html). This page holds what is specific to yolo.

- **Pick the class for training only.** On the run form, set `model-training-class` to one of the Slurm classes and submit.
  The other steps stay where they are.
- **The time limit is 12 hours.** `walltime="12h"` on `model-training` in `pipeline.py`. Raise it there for a longer training.
- **Training continues by itself.** It writes `last.pt` and `best.pt` into the checkpoint folder, so a job that reaches its
  limit, or loses its node, is continued from the last epoch with the same total number of epochs and in the same MLflow
  run ([Long jobs](https://storage.googleapis.com/kaos-dev-511509-kaos-lib/docs/long-jobs.html)).
- **`prepare-data` stays on the platform's own machines**, as every step that produces a dataset does
  ([HPC](https://storage.googleapis.com/kaos-dev-511509-kaos-lib/docs/hpc.html#what-differs-on-hpc-slurm)).
- **The INT8 steps** have their own `<step>-class` fields. Leave them on the platform's machines unless you need otherwise.
- **`qat-finetune` and `model-training` keep checkpoints.** `model-quantization` and `prepare-data` keep nothing and run
  again from the start ([What the pipeline does](the-pipeline.md#when-a-step-is-run-again)).
