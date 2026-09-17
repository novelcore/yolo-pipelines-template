# What the pipeline does

This page has no tasks. It is a short read so the run form makes sense later. Each step below is one job. They run in order, and each one hands its work to the next.

## The six steps

**1. Check your settings.** Before anything heavy runs, the pipeline looks at the options you picked and makes sure they fit together. If something is off, it stops here and tells you, in seconds, before it spends time or a graphics card. This step is your safety net.

**2. Load your dataset.** It finds the dataset you picked in lakeFS and gets it ready for training. It does not copy the whole thing around. It streams the images straight from storage, so even a large dataset works.

**3. Train the model.** This is the main event. It trains a YOLO model on a graphics card, for the number of rounds you asked for. This is the step that takes real time, from minutes to hours depending on your settings.

**4. Fine tune for INT8.** This one is optional. It only runs if you choose the `qat` quantization mode. It keeps training the model in a special way that prepares it to shrink down without losing much accuracy.

**5. Make a smaller model.** Also optional. It runs if you choose `ptq` or `qat`. It produces a smaller, faster INT8 version of the model, then checks that it still behaves like the full one. If the two drift too far apart, it fails on purpose, so you never ship a bad small model by accident.

**6. Save and register the model.** The final step. It saves the trained model and registers it so it has a name and a version. If you asked for it, it can also promote the model to Staging or Production.

## The short version

If you leave quantization off, the pipeline does steps 1, 2, 3, and 6. It trains a model and saves it. That is the simple path.

If you turn quantization on, steps 4 and 5 join in to produce a smaller model too.

!!! note "You do not run these one by one"
    You do not start each step yourself. You press Submit once, and the pipeline runs all of them in order for you. The next pages show you how.

Next, get your data in. Go to [Upload your dataset](upload-dataset.md).
