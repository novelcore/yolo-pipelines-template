# Promote a model

When a run finishes, the last step saves the model and gives it a name and a version. That is called registering. The version is made from the model that the last step also shows in the experiment's **Models** tab ([Read your results](results.md#your-trained-model-in-the-models-tab)), so the two are linked: the version names its source run and that model, and the model lists the versions made from it. Promoting is the next move. It gives a version a label, `champion` or `challenger`, so everyone knows which one to use.

## Two ways to promote

**At run time, on the form.** When you submit a run, there is a field called `registration-promote_to`. Leave it empty and the model is just saved, not promoted. Set it to `champion` or `challenger` and the pipeline gives the model that label as soon as it finishes.

**Later, in MLflow.** If you left it empty and decide later that a model is good, you can promote it by hand in MLflow. Open the model in the registry, pick the version, and add the alias. This needs rights to manage the model in MLflow; if you cannot, use the form field instead: it works whoever submits.

## Which should you use

For a first run, leave `registration-promote_to` empty. Look at your results first. Only promote once you have seen the numbers and you are happy.

When you have a run you trust, the cleanest path is to promote that exact version. You know what it scored, and its settings are saved next to it.

## Naming the model

There is also a field called `registration-registered_model_name`. Leave it empty and the pipeline uses a sensible default name. Set it if you want your models grouped under a name you choose, so every run adds a new version under the same name.

## Where the model files are

The files are not stored in MLflow: they are in lakeFS, in your pipeline's repository `outputs-<pipeline>`, and the pipeline puts a tag on that commit named `model-<name>-v<version>`, for example `model-object-pose-yolo-v3`. In lakeFS (the **Outputs** link), open that tag to see exactly the files of that version; `best.pt` is under the folder that the model's `kaos.outputs_path` tag names. In MLflow, the version's tags `kaos.outputs_repo`, `kaos.outputs_commit` and `kaos.lakefs_tag` say where they are, and the model it was made from (the Models tab) carries `kaos.outputs_path` too.

So from any of the three you reach the other two:

| You are on | To get to the others |
|---|---|
| the registry version | its source run and its model are linked on the version page; its tags name the lakeFS tag |
| the model in the Models tab | its source run is linked; its registered versions are listed; its tags name the lakeFS commit and folder |
| the lakeFS tag `model-<name>-v<version>` | the commit's details name the MLflow run |

## What promotion means

Promotion does not change the model. It is a label that says "this version is the champion." Your team and your systems can then ask for "the champion model" and always get the right version, without anyone copying files around.

That is the full loop. You uploaded data, ran the pipeline, read the results, and promoted a model.

If something did not go to plan, see [When something breaks](troubleshooting.md).
