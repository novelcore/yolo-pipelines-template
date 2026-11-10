# 1. Prepare and upload a yolo dataset

**Goal:** your labelled images are in a KAOS dataset, checked, and released as version `v1`.

**You need:** `kaosctl` installed and signed in ([Install and sign in](#install-and-sign-in) below), and your images with one label file each. A *dataset* here is a versioned store of files (it lives in lakeFS, the data store behind the console's **Datasets** page); its name starts with `dataset-`.

## The folder shape

A yolo dataset is a folder like this. The names matter.

```
dataset-spacecraft/
  data.yaml                 the classes, the keypoints and where the splits are
  images/train/a.jpg ...    training images
  images/val/a.jpg ...      validation images
  images/test/ ...          optional
  labels/train/a.txt ...    one text file per image, same name
  labels/val/a.txt ...
```

`data.yaml` for a pose dataset with one class and four keypoints:

```yaml
path: .
train: images/train
val: images/val
test: images/test
nc: 1
names:
  0: spacecraft
kpt_shape: [4, 3]          # 4 keypoints, each as x y visibility
flip_idx: [0, 3, 2, 1]     # which keypoint each one becomes when the image is flipped left-right
```

One line per object in every label file: the class id, the box (centre x, centre y, width, height), then `x y visibility` for each keypoint. All coordinates are between 0 and 1.

```
0 0.481 0.512 0.230 0.310  0.40 0.42 2  0.55 0.41 2  0.56 0.60 2  0.41 0.61 1
```

This pipeline trains **pose**, so `kpt_shape` and `flip_idx` are required for it to work, even though the check below does not insist on them.

## Install and sign in

1. **Do:** install `kaosctl` in a virtual environment and sign in from inside your pipeline's folder (the one you clone in [step 2](create-pipeline.md); the folder tells `kaosctl` which project you are in).

    ```bash
    python3 -m venv ~/.venvs/kaos && . ~/.venvs/kaos/bin/activate
    pip install --index-url https://europe-west1-python.pkg.dev/kaos-dev-511509/kaos-lib-rc/simple/ \
        --extra-index-url https://pypi.org/simple/ "kaos-lib[zitadel,mlflow]==0.4.0rc23"
    kaosctl --version
    kaosctl login
    ```

    **You should see:** `kaosctl 0.4.0rc23`, then your browser opens to sign in. If it does not, copy the link `kaosctl` printed into the browser yourself.

## Create the dataset

2. **Do:** create it **with the yolo layout**. A new dataset has *no* layout unless you ask for one, and without it nothing is checked.

    ```bash
    kaosctl dataset create spacecraft --layout yolo --classes spacecraft
    ```

    **You should see:** the dataset `dataset-spacecraft` created, with the folders above and a starter `data.yaml`. (Several classes: `--classes a,b,c`, the first is id `0`.)

3. **Do:** download the starter, then add your files and the keypoint lines in `data.yaml`.

    ```bash
    kaosctl dataset pull dataset-spacecraft@latest ./dataset-spacecraft
    cp ~/export/images/train/* ./dataset-spacecraft/images/train/
    cp ~/export/labels/train/* ./dataset-spacecraft/labels/train/
    cp ~/export/images/val/*   ./dataset-spacecraft/images/val/
    cp ~/export/labels/val/*   ./dataset-spacecraft/labels/val/
    ```

    `train` and `val` are required; `test` is optional.

## Validate

4. **Do:** check the folder before you upload it.

    ```bash
    kaosctl layout validate ./dataset-spacecraft --layout yolo
    ```

    (`kaosctl dataset validate ./dataset-spacecraft` does the same and assumes `yolo`.)

    **You should see:** `./dataset-spacecraft is a valid yolo dataset ...` and exit code 0. Otherwise every problem is listed at once, with exit code 2, for example:

    ```
    ./dataset-spacecraft has 1 problem(s) for layout yolo:
      - images/val/a.jpg has no labels/val/a.txt
      - labels/val/a.txt:1: coordinates must be within [0,1]: ...
      - labels/val/a.txt:1: class id 3 is outside 0..0: ...
      - labels/val/a.txt:1: expected '<class-id> <x> <y> <w> <h>' + 4 keypoints x 3 values (17 values), got 3: ...
    ```

    Fix them and run it again. `kaosctl layout list` shows the layouts you can use, and `kaosctl layout show yolo` prints the rules.

## Push

5. **Do:** upload it. This opens a *pull request* (a proposed change that a person merges) into the dataset, so the dataset's main data never changes by accident.

    ```bash
    kaosctl dataset push ./dataset-spacecraft dataset-spacecraft -m "first 200 labelled images"
    ```

    **You should see:** a link to the pull request. Open it, look at the change in lakeFS and **Merge** it. Then go on to [release a version](release-a-version.md).

!!! warning "Editing classes in the browser"
    The lakeFS **Edit** on `data.yaml` rewrites the file from the class names. On a pose dataset, check afterwards that `kpt_shape` and `flip_idx` are still in it, or edit `data.yaml` on your computer and `push` it.

More about datasets (listing, pulling, layouts you can make): [Datasets](datasets.md).
