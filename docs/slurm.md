# Run on HPC (Slurm)

Training can run on an HPC (Slurm) site when your project is connected to one. Where a step runs, the time limit, the queue and
the reasons a job waits are explained in the guide: [HPC (Slurm)](https://storage.googleapis.com/kaos-dev-511509-kaos-lib/docs/hpc.html). This page holds what is specific to yolo.

- **Pick the class for training only.** On the run form, set `model-training-class` to one of the Slurm classes and submit.
  The other steps stay where they are.
- **The time limit starts at 12 hours.** `walltime="12h"` on `model-training` in `pipeline.py` is the default of the run-form field `model-training-walltime`. Whoever starts a run can write another value there for that run; change `walltime=` in `pipeline.py` to change the default for every run.
- **Training is resumable, and that is declared.** `model-training` declares `resumable=True`. On HPC (Slurm) a step with a
  limit above 11 hours 30 minutes is refused without it, and a resumable step may have up to 4140 minutes (69 hours)
  ([why](https://storage.googleapis.com/kaos-dev-511509-kaos-lib/docs/contract.html#a-step-can-be-stopped-and-started-again)).
  Training writes `last.pt` and `best.pt` into the checkpoint folder, so a job that reaches its limit, or loses its
  node, is continued from the last epoch with the same total number of epochs and in the same MLflow run
  ([Long jobs](https://storage.googleapis.com/kaos-dev-511509-kaos-lib/docs/long-jobs.html)).
- **`prepare-data` stays on the platform's own machines**, as every step that produces a dataset does
  ([HPC](https://storage.googleapis.com/kaos-dev-511509-kaos-lib/docs/hpc.html#what-differs-on-hpc-slurm)).
- **Training reads `prepared` from a copy on the site.** `prepare-data` makes the dataset `prepared` on the platform's own
  machines. Before the training job is sent to the queue, the platform copies that exact version to your project's storage
  on the site, and the next run that reads the same version reuses it
  ([what is copied, and when](https://storage.googleapis.com/kaos-dev-511509-kaos-lib/docs/data-reads.html#a-dataset-a-step-made-read-on-hpc-slurm)).
- **The INT8 steps** have their own `<step>-class` fields. Leave them on the platform's machines unless you need otherwise.
- **`qat-finetune` and `model-training` keep checkpoints.** `model-quantization` and `prepare-data` keep nothing and run
  again from the start ([What the pipeline does](the-pipeline.md#when-a-step-is-run-again)).
