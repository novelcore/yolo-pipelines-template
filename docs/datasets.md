# Datasets

The pipeline reads one dataset and writes one. Datasets, versions and the form field that picks one are explained in the
guide: [Read data](https://storage.googleapis.com/kaos-dev-511509-kaos-lib/docs/read-data.html) and [Produce a dataset](https://storage.googleapis.com/kaos-dev-511509-kaos-lib/docs/produce-a-dataset.html). The commands are in its
[Reference](https://storage.googleapis.com/kaos-dev-511509-kaos-lib/docs/reference.html#commands).

## The folder shape the pipeline reads

A yolo dataset has this shape at the root of the repository. The names matter. Anything else in the repository, such as a
README, is ignored.

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

One line per object in every label file: the class id, the box (centre x, centre y, width, height), then `x y visibility`
for each keypoint. All coordinates are between 0 and 1.

```
0 0.481 0.512 0.230 0.310  0.40 0.42 2  0.55 0.41 2  0.56 0.60 2  0.41 0.61 1
```

This pipeline trains pose, so `kpt_shape` and `flip_idx` are required for it to work, even though the check below does not
insist on them.

## Create, check and push it

A new dataset has no layout unless you name one. The `yolo` layout starts the repository with the files above and checks
every push:

```bash
kaosctl dataset create spacecraft --layout yolo --classes spacecraft
kaosctl dataset pull dataset-spacecraft@latest ./dataset-spacecraft
kaosctl layout validate ./dataset-spacecraft --layout yolo
kaosctl dataset push ./dataset-spacecraft dataset-spacecraft -m "first 200 labelled images"
```

`validate` lists every problem at once and exits with code 2, for example `images/val/a.jpg has no labels/val/a.txt`.
`push` opens a pull request into the dataset; merge it, then release a version with
`kaosctl dataset release dataset-spacecraft`. Several classes: `--classes a,b,c`, the first is id 0.

## The dataset the pipeline writes

The first step that touches your data, `prepare-data`, writes it again as the dataset `dataset-<your app>-prepared`
(`<your app>` is the name of your app; the steps only ever use the short name `prepared`): all of it, or a reproducible subset
when you set `data-sample_size` ([What the pipeline does](the-pipeline.md#prepare-data-the-first-step-writes-a-dataset)).
It is a dataset like yours, with versions, but only the pipeline writes it: you can read it and pick a version, never push
to it. Training reads exactly the version of the run that wrote it. Its `latest` moves only when you ask for it in the run's
advanced config (`plan: {promote: {prepared: true}}`), so you can also train a later run on a prepared version by picking
`dataset-<your app>-prepared@<version>` as the `dataset`.
