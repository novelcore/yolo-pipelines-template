# 7. Run on HPC (Slurm)

**Goal:** send the training step to an HPC (Slurm) supercomputer, if your project has one.

*Slurm* is the job scheduler of a supercomputer. In KAOS the HPC machines come from a provider; today that is MeluXina, Luxembourg's supercomputer. Only projects connected to it see its machines.

1. **Do:** open the run form and look at the `model-training-class` dropdown (the form's read-only `pipeline-info` field at the top also lists the classes). The console's **Machines** page shows the same classes.

    **You should see:** the cluster classes, and, if your project has HPC, classes whose names start with `meluxina` (the provider's name), such as `meluxina-gpu`. No `meluxina` entries means your project is not connected: ask your KAOS contact.

    ![The model-training-class dropdown open, showing cluster and HPC options](img/training-class.png)

2. **Do:** set `model-training-class` to `meluxina-gpu` and submit as usual ([Run the pipeline](run-small-and-full.md)). The other steps stay where they are.

    **You should see:** `model-training` waits first (the HPC cluster is shared, so the job can queue for a while; that is normal), then runs. The dataset is copied to the machine before the job starts, and the weights come back to `outputs-<pipeline>` exactly as for a cluster run.

Things to know:

- **Time limit:** training on HPC has a 12-hour limit (`hpc_time_limit="12h"` on `model-training` in `pipeline.py`). Raise it there for longer trainings.
- **Machines and queues are chosen for you.** You never name a queue or a quality of service. The platform learns from earlier runs how long a step really takes and asks the site for a shorter limit when that gets the job started sooner; if a job is cut at a learned limit it runs once more at your limit. Your limit always stays the ceiling.
- **A step longer than the longest HPC run** (the site's maximum, for example two days) is not cut. If it is resumable it runs as several capped runs, each continuing from the last checkpoint (up to 5 continuations). If it is not resumable it is **refused before it is sent**, with the reason `TooLongNotResumable`, the limit you asked for and the maximum. The two fixes: make it resumable (it writes into the checkpoint folder, or declares `resumable=` on the step), or shorten `hpc_time_limit`. `qat-finetune` and `model-training` already write checkpoints, so they continue by themselves. A resumable step needing more runs than the continuation cap allows is refused with `TooLongBeyondContinuations`.
- **Long trainings continue by themselves:** when the time limit, a node failure or maintenance ends a job, training continues in a new job from its last checkpoint (up to 5 times), with the same total number of epochs and in the same MLflow run. Each new job waits in the queue again and counts against your budget. A continuation that does not get past its checkpoint ends with the reason `NoProgress`; a step with no GPU use and no new log lines for 20 minutes ends as `Idle` (`idle_timeout` changes that, `0` turns it off). Details and messages: [Long runs: checkpoints](https://storage.googleapis.com/kaos-dev-511509-kaos-lib/docs/checkpoint.html); what each step of this pipeline keeps: [What the pipeline does](the-pipeline.md#when-a-step-is-run-again).
- **INT8 steps** have their own `<step>-class` fields; leave them on the cluster unless you need otherwise.
- **Which machine otherwise:** leave the class dropdowns alone and training runs on a cluster GPU.

## Why a job waits or stops

The step's status line in the run view says why. The ones you can meet:

| reason | meaning | what to do |
|---|---|---|
| `WaitingForBudget` | the job fits your project's HPC allocation once running jobs finish | nothing: it starts by itself |
| `ConcurrencyLimit` | your project is at its limit of jobs running at once | nothing: it waits for a free slot |
| `BudgetBlocked` | it can never fit this month within your project's own share of the hours | shorten the step, or ask your KAOS contact for more hours |
| `SiteLimit` | it can never fit this month because the HPC site's own monthly limit on the account is used up (also shown for a job queued behind that limit) | wait for the new month, or ask your KAOS contact |
| `SubmitLimit` | the site allows only so many queued plus running jobs per user and your project's jobs already fill it. The step **fails** at once instead of waiting | resubmit when some jobs have finished |
| `TooLongNotResumable` | see above | make it resumable or shorten it |
| `GPUViolation` | on a shared node, your step computed on a GPU that is not its own, so it was stopped (only that step) | use only the GPU the step is given (`CUDA_VISIBLE_DEVICES`) |

A project has a separate allocation for GPU, CPU, FPGA and large-memory steps. One that is spent blocks only the steps that use it: a spent CPU allocation holds CPU steps and leaves GPU training alone, and the message names the allocation.

More (the classes, staging, time limits, what to do when a job waits or stops): [the Run on HPC (Slurm) page of the hera cookbook](https://novelcore.github.io/hera-pipeline-template/hpc/). See also [When something breaks](troubleshooting.md#a-step-waits-and-never-starts).
