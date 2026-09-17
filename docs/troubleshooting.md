# When something breaks

Runs do not always go smoothly the first time. That is normal. The pipeline is built to tell you what went wrong in plain words. This page lists the things people hit most, and what to do.

The golden habit. When a step turns red, click it and read the log. The last few lines almost always name the problem.

## The run stopped at the very first step

That first step checks your settings. If it stops there, one of your form values does not fit. The message names it. Common cases:

- A value outside the allowed set. Some fields accept only certain choices. The message lists the ones that are allowed. Pick one of those.
- Two settings that do not agree with each other. The message says which two. Change one of them.

This is the safety net working. It stopped before spending any training time. Fix the field it named and submit again.

## The dataset step cannot find my data

Check the `data-ref` you typed on the form. It must match where you uploaded. If you uploaded to the default, that is `main`. A typo here means the pipeline looks in the wrong place and finds nothing.

Also open lakeFS and confirm your dataset is really there, with `data.yaml`, `images`, and `labels`. If the upload did not finish, run it again. See [Upload your dataset](upload-dataset.md).

## A step is stuck and never starts

Sometimes a step sits waiting instead of running. That usually means the machines are busy, not that anything is wrong with your run. Give it a few minutes. If it stays stuck for a long time, tell your platform contact. It is a platform thing, not your settings.

## Training is taking forever

That is often expected, but you can make a test run quick.

- Use the smallest model, `yolov8n`.
- Set `train-epochs` low, like `10`.
- If your dataset is large, set `data-sample_size` to a small number so it trains on a subset.

Prove the whole thing works with a fast run first. Then scale up.

## The smaller model failed a check

If you turned quantization on, the pipeline makes a smaller INT8 model and then checks it still behaves like the full one. If they drift too far apart, that step fails on purpose. This is a good failure. It stopped a bad small model from being saved. For a first pass, you can leave quantization off and just train.

## I cannot find my run in the results

Look for the name you set in `experiment-name`. If you left it default, several runs may share that name, so sort by time and take the latest. Setting a clear name each time makes runs easy to find later.

## I am stuck and the message does not help

That happens. Copy the message, note what you were doing, and send it to your platform contact. A clear description and the exact message is all they need to help fast.

## A quick checklist before you submit

- [ ] `data-ref` matches where I uploaded, usually `main`.
- [ ] My dataset shows up in lakeFS with data.yaml, images, and labels.
- [ ] For a first test: small model, low epochs, quantization off.
- [ ] `experiment-name` is something I will recognize later.
