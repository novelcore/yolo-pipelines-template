# Datasets

A dataset is a **lakeFS repository** named `dataset-<name>`. Your project starts with one,
`dataset-default`.

| thing | what it is |
|---|---|
| version | a tag on `main`: `v1`, `v2`, … |
| `latest` | the newest data on `main` |
| change | a pull request into `main`, reviewed and merged by a person |
| pick | `dataset = <repository>@<version>` in the run form |
| results | weights and exports go to `outputs-<pipeline>`; the pipeline's own `prepare-data` step writes a dataset of its own ([below](#the-dataset-the-pipeline-writes)) |

A run reads **one commit**: the one its pick resolves to. Merging new data during a run does not
change what that run reads.

## The layout this pipeline reads

```
data.yaml                 path, train, val, test, names, kpt_shape, flip_idx
images/{train,val,test}/   the images
labels/{train,val,test}/   one .txt per image: class x y w h, then the keypoints
```

at the **root** of the repository. Anything else in the repository (a README, notes) is ignored.

This pipeline trains **pose**, so `data.yaml` also needs `kpt_shape` and `flip_idx`
([Upload images and labels](prepare-dataset.md)).

## The dataset the pipeline writes

The first step that touches your data, `prepare-data`, writes it again as the dataset `dataset-<your app>-prepared` (`<your app>` is the name of your app; the steps only ever use the short name `prepared`): all of it,
or a reproducible subset when you set `data-sample_size` ([What the pipeline does](the-pipeline.md#prepare-data-the-first-step-writes-a-dataset)).
It is a dataset like yours, with versions, but only the pipeline writes it: you can read it and pick a version, never push to it.
Training reads exactly the version of the run that wrote it. Its `latest` moves only when you ask for it in the run's advanced
settings (`plan: {promote: {prepared: true}}`), so you can also train a later run on a prepared version by picking
`dataset-<your app>-prepared@<version>` as the `dataset`.

## Layouts

A *layout* is a named folder shape plus the checks for it. A new dataset has **no layout** unless you name one: `kaosctl dataset create <name> --layout yolo --classes a,b` starts a repository with the `yolo`
layout (the starter files above, plus the checks that `validate` and `push` run). `kaosctl layout list` shows the layouts you can use, `kaosctl layout show yolo` prints one, and `kaosctl layout validate <dir> --layout yolo` checks a folder. You can do without it:

- `kaosctl dataset create <name>` (or `--layout none`) makes an empty `dataset-<name>`.
- A repository named `dataset-<something>` created by hand in lakeFS becomes a dataset after its
  first commit.
- A repository with **no layout validates nothing**; `push` takes the files as they are.
  Pass `--layout yolo` to `push` or `validate` to get the YOLO checks anyway.
- In Python, `kaos_lib.datasets.materialize(DatasetRef.parse("<repo>@<ref>"))` reads any dataset you may read, `<ref>` being `vN`,
  `latest`, a branch, a tag or a commit id. The run form's dropdown lists the datasets and their versions, newest first, then `@latest`, and updates within about a minute of a release.

## The commands

`kaosctl` comes with kaos-lib (0.5 or newer; [install it](prepare-dataset.md#install-and-sign-in)).

```bash
kaosctl login
kaosctl dataset create <name> [--layout yolo|none] [--classes a,b] [--description "..."]   # --classes needs --layout yolo
kaosctl dataset list [--shared]
kaosctl dataset versions <repo>
kaosctl dataset pull <repo>@<ref> [dir] [--prefix folder/]
kaosctl dataset validate <dir> [--layout yolo]        # = kaosctl layout validate <dir> --layout yolo
kaosctl dataset push <dir> <repo> -m "message" [--layout yolo] [--no-validate]
kaosctl dataset release <repo>
```

In the browser, lakeFS's **Datasets** page has **New dataset**, upload on a branch with a pull
request, and **Release version**. Sharing a dataset with another project is done in the console
(**Datasets** → the dataset → **Sharing**, by the owning project's admins); see [Use a dataset shared with you](shared-dataset.md). The kaos-lib guide explains each task once: [Datasets](https://storage.googleapis.com/kaos-dev-511509-kaos-lib/docs/datasets.html) (the commands, [in the browser](https://storage.googleapis.com/kaos-dev-511509-kaos-lib/docs/datasets.html#in-the-browser), [layouts](https://storage.googleapis.com/kaos-dev-511509-kaos-lib/docs/datasets.html#layouts)) and [sharing a dataset](https://storage.googleapis.com/kaos-dev-511509-kaos-lib/docs/shared-datasets.html#share-a-dataset-the-owning-project).

Every command prints the next step and the link to what it made. Exit codes: 0 ok, 1 something for
you to fix (one `ERROR:` line), 2 an invalid dataset.

!!! note "Who may do what"
    Reading a dataset needs read rights, which the pipeline has. Creating a dataset, pushing, and
    releasing a version are for people. A step can write only the datasets its `pipeline.py` declares in `produces=`
    (here `prepared`); it can never write yours.
