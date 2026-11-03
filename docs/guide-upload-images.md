# Guide: upload images and labels

**Goal:** a yolo pose dataset with your labelled images, checked, merged and released as `v1`, so
the pipeline can train on it.

**You need:** `kaosctl` signed in, inside your app folder
([cookbook 4](https://novelcore.github.io/hera-pipeline-template/cookbook/4-install-kaosctl/)), and your images with one label file each. Your role
is `dev`, `ops` or `admin`.

## 1. Create the dataset

List your classes in label-id order. The first one is id `0`.

```bash
kaosctl dataset create spacecraft --classes spacecraft
```

In the browser: lakeFS → **Datasets** → **New dataset** → layout **yolo**, one class per line.

## 2. Add the keypoints to `data.yaml`

This pipeline trains **pose**, so `data.yaml` needs the shape of your keypoints. Pull the starter
layout and edit it:

```bash
kaosctl dataset pull dataset-spacecraft@latest ./dataset-spacecraft
```

```yaml
# ./dataset-spacecraft/data.yaml
path: .
train: images/train
val: images/val
test: images/test
nc: 1
names:
  0: spacecraft
kpt_shape: [4, 3]          # 4 keypoints, each x y visibility
flip_idx: [0, 3, 2, 1]     # which keypoint each one becomes when the image is flipped left-right
```

## 3. Put in the images and labels

Each image `images/<split>/<name>.jpg` needs `labels/<split>/<name>.txt`, with one line per object:
the class id, the box (centre x, centre y, width, height), then `x y visibility` for each keypoint.
All coordinates are between 0 and 1.

```
0 0.481 0.512 0.230 0.310  0.40 0.42 2  0.55 0.41 2  0.56 0.60 2  0.41 0.61 1
```

```bash
cp ~/export/images/train/* ./dataset-spacecraft/images/train/
cp ~/export/labels/train/* ./dataset-spacecraft/labels/train/
cp ~/export/images/val/*   ./dataset-spacecraft/images/val/
cp ~/export/labels/val/*   ./dataset-spacecraft/labels/val/
```

`train` and `val` are required; `test` is optional.

## 4. Validate

```bash
kaosctl dataset validate ./dataset-spacecraft
```

**You should see** `./dataset-spacecraft is a valid yolo dataset ...` (exit code 0). Otherwise
it lists every problem at once (exit code 2): a label without its image, a class id outside
`names`, a keypoint count that does not match `kpt_shape`, a coordinate outside 0..1.

## 5. Push, merge, release

```bash
kaosctl dataset push ./dataset-spacecraft dataset-spacecraft -m "first 200 labelled images"
```

Open the pull request link, check the change in lakeFS, and **Merge**. Then click
**Release version** on the dataset, or:

```bash
kaosctl dataset release dataset-spacecraft      # v1
```

A few minutes later `dataset-spacecraft@v1` is in the run form's `dataset` dropdown.

!!! warning "Editing classes in the browser"
    lakeFS's **Edit** on `data.yaml` (Edit classes) rewrites the file from the class names. After
    using it on a pose dataset, check that `kpt_shape` and `flip_idx` are still there (verify).
    To be safe, edit `data.yaml` locally and `push` it.

More on datasets (sharing, pulling, the browser path): [Datasets](datasets.md) and
[cookbook 5](https://novelcore.github.io/hera-pipeline-template/cookbook/5-your-data/).

**Next:** [Train on a version and find the model](guide-train-and-find-the-model.md).
