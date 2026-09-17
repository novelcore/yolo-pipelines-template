# Promote a model

When a run finishes, the last step saves the model and gives it a name and a version. That is called registering. Promoting is the next move. It marks a version as ready for a stage, like Staging or Production.

## Two ways to promote

**At run time, on the form.** When you submit a run, there is a field called `registration-promote_to`. Leave it empty and the model is just saved, not promoted. Set it to `Staging` or `Production` and the pipeline promotes the model to that stage as soon as it finishes.

**Later, in MLflow.** If you left it empty and decide later that a model is good, you can promote it by hand in MLflow. Open the model in the registry, pick the version, and set its stage.

!!! info "Screenshot"
    A picture of the registration-promote_to field on the form will go here.

## Which should you use

For a first run, leave `registration-promote_to` empty. Look at your results first. Only promote once you have seen the numbers and you are happy.

When you have a run you trust, the cleanest path is to promote that exact version. You know what it scored, and its settings are saved next to it.

## Naming the model

There is also a field called `registration-registered_model_name`. Leave it empty and the pipeline uses a sensible default name. Set it if you want your models grouped under a name you choose, so every run adds a new version under the same name.

!!! info "Screenshot"
    A picture of the model in the MLflow registry, showing its stage, will go here.

## What promotion means

Promotion does not change the model. It is a label that says "this version is the one for this stage." Your team and your systems can then ask for "the Production model" and always get the right version, without anyone copying files around.

That is the full loop. You uploaded data, ran the pipeline, read the results, and promoted a model.

If something did not go to plan, see [When something breaks](troubleshooting.md).
