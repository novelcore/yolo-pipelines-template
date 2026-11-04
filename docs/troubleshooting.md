# 7. When something breaks

Messages below are the real ones. Click a red box in the run page to read the step's log; the last lines usually name the problem.

## Before a run: `kaosctl`

### `cannot determine the lakeFS URL`

`kaosctl` finds your project from the pipeline folder. Run it **inside your pipeline's clone** ([step 3](create-pipeline.md)), or pass `--url`, or set `LAKEFS_ENDPOINT`.

### `lakeFS rejected the credential (...). Run kaosctl login again.`

Your sign-in expired. Run `kaosctl login`. The browser opens and `kaosctl` waits about three minutes; if it did not open, copy the printed link into the browser yourself.

### `you are not allowed to do this`

Your role on the project is not enough for that command. Creating datasets, pushing and releasing need `dev`, `ops` or `admin`.

### `externally-managed-environment` when installing

Your computer does not let `pip` change its own Python. Install inside a virtual environment, as in [step 1](prepare-dataset.md#install-and-sign-in).

### `dataset-spacecraft already exists. Push to it with kaosctl dataset push <dir> dataset-spacecraft.`

You ran `dataset create` for a name that is taken. Push to the existing dataset, or pick another name.

### `--classes and --param fill a layout's parameters, and no layout is named`

You left out the layout. A new dataset has none unless you ask: `kaosctl dataset create spacecraft --layout yolo --classes spacecraft`.

### `<folder> has N problem(s) for layout yolo:`

`kaosctl layout validate` found problems in your files, listed one per line: `images/val/a.jpg has no labels/val/a.txt`, `class id 3 is outside 0..0`, `coordinates must be within [0,1]`, `expected '<class-id> <x> <y> <w> <h>' + 4 keypoints x 3 values (17 values), got 3`, or `data.yaml: no file matches (required)`. Fix them and run it again ([step 1](prepare-dataset.md#validate)).

### `nothing new to release: the head of dataset-spacecraft main is already v1. Merge a pull request first.`

Nothing was merged since the last release. Merge your pull request, then release ([step 2](release-a-version.md)).

### `not shared with your project: project team-a's lakeFS refused it (...). Ask team-a to share the dataset with your project.`

The dataset belongs to another project that has not shared it with yours. Ask its admins to share it ([Use a shared dataset](shared-dataset.md)).

### `repository dataset-spacecraft does not exist. List datasets with kaosctl dataset list.`

The name is wrong, or the dataset is in another project: check `kaosctl dataset list` and `kaosctl dataset list --shared`.

## Stops at the first step (`config-validation`)

This step checks your form before any training time is spent; its message names what to fix.

### `dataset ... has no data.yaml at the root of its repository - a valid yolo dataset has data.yaml, images/ and labels/.`

The version you picked has no data: usually the form's default `dataset-default@latest`, which is empty in a new project. Pick your own released version ([step 2](release-a-version.md)). Also check in lakeFS that your pull request was merged and `data.yaml`, `images` and `labels` are at the root.

### `dataset dataset-spacecraft@v3: dataset-spacecraft has no version v3. Available: v1, v2.`

That version does not exist. Pick one it lists, or release a new one. A version you just released can take about a minute to reach the dropdown.

### `pretrained weights not found: s3://...` or `model.pretrained_weights must be null or an s3:// path`

`model-pretrained_weights` must be empty or the full `s3://` path of a `.pt` file that exists.

### Other messages

Some fields only accept certain choices (the message lists them) and some pairs of fields must agree (the message names both). Change the field and submit again. [All the fields](parameters.md) list the allowed values.

## During the run

### `model-quantization` fails with `calibration_frames` under 100

```
1 validation error for QuantizationParams
calibration_frames
  Input should be greater than or equal to 100 [type=greater_than_equal, input_value=50, input_type=int]
```

Set `quantization-calibration_frames` to 100 or more (default 512; at most 10000) and submit again. The first step does not catch this, so the training before it was spent.

### Training ends with `No valid labels found`

The labels do not match `kpt_shape` in `data.yaml` (a different number of keypoints). Run `kaosctl layout validate` on your folder: it reports `expected ... keypoints x 3 values`.

### A step waits and never starts

The machines are busy, or on Slurm the job is queued. Give it a few minutes ([Run on Slurm](slurm.md)). If it stays stuck for a long time, tell your KAOS contact.

### Training is taking forever

Make a test run quick: `model` = `yolov8n`, `train-epochs` = `10`, and `data-sample_size` = `50`. Prove the whole thing works, then scale up.

### Registration fails with `lakeFS tag model-... already points at ...`

The model name and version already name a different result. Use another `registration-registered_model_name`, or ask your KAOS contact.

## After the run

### I cannot find my run

Look for the `experiment-name` you set in MLflow. Runs with the same name share one experiment: sort by time. A clear name each time makes runs easy to find ([Results](results.md)).

### The INT8 model looks wrong

Open the quantization run in MLflow and read `parity_passed` and `parity_max_abs_error`. The step does not fail on a bad comparison. Try `qat` instead of `ptq`, or more `quantization-calibration_frames`.

## Everything else

Access, tools and runs are the same as for every pipeline of this kind: [the hera cookbook, "When things go wrong"](https://novelcore.github.io/hera-pipeline-template/cookbook/10-when-things-go-wrong/). If the message does not help, send the exact message and what you were doing to your KAOS contact.

## Checklist before you submit

- [ ] `dataset` is my own `<dataset>@<version>` and that version exists, not the empty `dataset-default`.
- [ ] For a first test: small model, few epochs, `quantization-mode` = `none`.
- [ ] If I use `ptq` or `qat`: `quantization-calibration_frames` is at least 100.
- [ ] `experiment-name` is something I will recognise later.
