# First dataset, first run, your results

This page takes you from nothing to a trained model and its files, with every command.

!!! warning "First, you need an app"
    This page assumes you have **already created an app from this template** and cloned it
    ([What you need](what-you-need.md)). Your app carries `.kubecore/dataset-config.yaml`, the
    small file that tells the tool where your project's lakeFS is. You never need a link or a password.

## 1. Install the tool and sign in

The `kaos-cli` tool comes with kaos-lib, a Python package. Install it once, inside a small private Python space
for this project (a "virtual environment"), in your app folder.

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install "kaos-lib[zitadel] @ https://github.com/kaos-io/kaos-lib/archive/<commit>.zip"
kaos-cli login
```

`<commit>` is the commit in `KAOS_LIB_PACKAGE` in `steps/model_training/Dockerfile` (your
platform contact can also give it to you). On Windows, the second line is `.venv\Scripts\activate`.
Your browser opens once and asks you to sign in. Log in, close the tab, and go back to the
Terminal. The tool remembers you.

!!! tip "Why the virtual environment?"
    Many computers no longer let `pip install` change the system's own Python (a message that
    mentions `externally-managed-environment`). The virtual environment avoids that.

## 2. Create a dataset

```bash
kaos-cli dataset create spacecraft --classes spacecraft
```

This makes the lakeFS repository `dataset-spacecraft` with the YOLO starter layout:
`data.yaml`, `images/{train,val,test}/` and `labels/{train,val,test}/`, and prints a link to it.

This pipeline trains **pose**, so after the first step open `data.yaml` and add the keypoints
of your dataset (`kpt_shape: [4, 3]`, `flip_idx: [0, 3, 2, 1]`, for four keypoints with
visibility): the next step shows how.

!!! note "Templates are optional"
    `kaos-cli dataset create spacecraft --template none` makes an empty `dataset-spacecraft`,
    and a `dataset-*` repository you create by hand in lakeFS becomes a dataset after its first
    commit. Everything below works the same.

## 3. Put your files in

```bash
kaos-cli dataset pull dataset-spacecraft@latest ./dataset-spacecraft
# edit ./dataset-spacecraft/data.yaml (kpt_shape, flip_idx, names)
# copy your images into images/train, images/val, images/test
# and one label file per image into labels/train, labels/val, labels/test
kaos-cli dataset validate ./dataset-spacecraft --format ultralytics
kaos-cli dataset push ./dataset-spacecraft dataset-spacecraft -m "first 200 images"
```

- `validate` lists every problem it finds in one go: a label for no image, a class number
  outside `names`, a keypoint count that does not match `kpt_shape`. Fix them and run it again.
- `push` uploads your folder to a draft and opens a **pull request**; it prints the link.

## 4. Merge the pull request and release a version

Open the pull request link, look at the change, and merge it. Nothing changes on `main` until
you do. Then tag the new state as a version:

```bash
kaos-cli dataset release dataset-spacecraft      # prints v1
kaos-cli dataset versions dataset-spacecraft
```

A run reads a version, not "whatever is there now", so a result always points back to its data.

## 5. Run the pipeline on it

Open the pipeline in the Argo UI ([Run the pipeline](run-it.md)). The form's **`dataset`** field is a
dropdown of `<dataset>@<version>` choices:

```
dataset-spacecraft@v1
dataset-spacecraft@latest
dataset-default@latest
```

Pick `dataset-spacecraft@v1`, set `experiment-name` (for example `my-first-run`), and Submit. If the
new version is not in the list yet, wait a few minutes: the list is refreshed when a dataset or a
version changes.

The checking step confirms the dataset and version exist and have a `data.yaml`; `dataset-loading`
turns your pick into one commit; training reads that commit, even if someone merges a change
to the dataset during the run.

To train on MeluXina, choose a `meluxina-gpu` class for `model-training` on the same form: the dataset
is staged there for you and the results come back to the same place.

## 6. Find your results

- **Numbers, charts and the registered model** are in MLflow ([Read your results](results.md)).
- **Your files** are in the app's repository `outputs-<app>` in lakeFS, branch `main`. Every step
  that wrote files added **one commit**, named after the run:

    | folder | what |
    |---|---|
    | `checkpoints/<experiment>/<run>/` | `best.pt`, `last.pt`, `epoch_NNNN.pt`, exports |
    | `quantization/<experiment>/<run>/` | the INT8 `.tflite` (when quantization is on) |

  `<experiment>` is your `experiment-name`; `<run>` is the name of the Argo run. To get them on
  your computer:

```bash
kaos-cli dataset pull outputs-<app>@main ./results --prefix checkpoints/<experiment>/<run>/
```

The commit's details say which dataset and which commit the run trained on, and the MLflow run id.

If a training step is retried (its machine was reclaimed), it resumes from the last checkpoint it
saved instead of starting over.
