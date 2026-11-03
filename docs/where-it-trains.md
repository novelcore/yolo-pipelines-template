# Where it trains

You may wonder which machine your training runs on. This page explains the choice, in plain words. For most people the answer is "leave it alone, it just works." But the option is there if you need it.

## The default

By default, training runs in your cluster, on a graphics card. You do not choose anything. You press Submit and the training step lands on a GPU. This is the right choice for almost every run.

## The compute class dropdown

Each step on the form has its own dropdown called `<step>-class`, for example `model-training-class`. This picks which kind of machine that step runs on. The options are the machine classes your project was set up with. Your platform contact decides what is in that list.

So the placement is per step. You could put the heavy training step on a big machine and leave the quick steps on small ones. Most of the time the defaults are already sensible, so you can ignore this. The read-only `pipeline-info` field at the top of the form lists the classes your project has.

![The model-training-class dropdown open, showing cluster and MeluXina options](img/training-class.png)

## MeluXina, if your project has it

Some projects are connected to MeluXina, a large supercomputer. If yours is, its machines show up in that same class dropdown, with names that start with `meluxina`. Pick one for the training step and that step runs on MeluXina instead of in the cluster. Every other step stays where it was.

Two things to know about MeluXina.

- There can be a wait. MeluXina is shared, so your step may sit in a queue for a while before it starts. That is normal.
- It only appears if your project is set up for it. If you do not see `meluxina` options, your project is not connected, and that is fine. The cluster GPU works the same way.
- Training has a 12-hour limit there (`hpc_time_limit="12h"` on `model-training` in `pipeline.py`). Raise it in `pipeline.py` for longer trainings.
- The dataset is staged on MeluXina before the job starts, and the weights come back to `outputs-<app>` exactly as for a cluster run.

The hera guide's [MeluXina page](https://novelcore.github.io/hera-pipeline-template/meluxina/) has the details: the classes, staging, time limits, and what to do when a job waits or stops.

## What about GCP

You do not choose GCP. GCP is simply what the cluster runs on underneath. It is not a button on the form. The only real choice you have is the compute class: a cluster machine, or a MeluXina one if it is offered.

## The short version

Leave the class dropdowns alone and training runs on a cluster GPU. That is the normal path. Only touch them if you need a specific class, or if you want to send the training step to MeluXina.

Next, watch it run. Go to [Watch it train](watch-it.md).
