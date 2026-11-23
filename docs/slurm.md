# Run on HPC (Slurm)

Training can run on an HPC (Slurm) site when your project is connected to one. Where a step runs, the queue and
the reasons a job waits are explained in the guide: HPC (Slurm) (guide page: hpc). This page holds what is specific to yolo.

- **Pick the class for training only.** On the run form, set `model-training-class` to one of the Slurm classes and submit.
  The other steps stay where they are.
- **There is no time to set.** The platform sets the time of every job. A training longer than one job on the site is
  stopped shortly before the job's end, after a last save, and continued in the next job (guide page: long-jobs).
- **Training continues after an interruption.** It writes `last.pt` and `best.pt` into the checkpoint folder, so a job that
  loses its node, or one that reaches its limit when a checkpoint copy exists, is continued from the last epoch with the same total number of epochs and in the same MLflow
  run (guide page: long-jobs).
- **`prepare-data` stays on the platform's own machines**, as every step that produces a dataset does
  (guide page: hpc).
- **Training reads `prepared` from a copy on the site.** `prepare-data` makes the dataset `prepared` on the platform's own
  machines. Before the training job is sent to the queue, the platform copies that exact version to your project's storage
  on the site, and the next run that reads the same version reuses it
  (guide page: data-reads).
- **The INT8 steps** have their own `<step>-class` fields. Leave them on the platform's machines unless you need otherwise.
- **`qat-finetune` and `model-training` keep checkpoints.** `model-quantization` and `prepare-data` keep nothing and run
  again from the start ([What the pipeline does](the-pipeline.md#when-a-step-is-run-again)).
