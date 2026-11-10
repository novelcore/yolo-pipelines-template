# Datasets

A dataset is a **lakeFS repository** named `dataset-<name>`. Your project starts with one,
`dataset-default`.

| thing | what it is |
|---|---|
| version | a tag on `main`: `v1`, `v2`, … |
| `latest` | the newest data on `main` |
| change | a pull request into `main`, reviewed and merged by a person |
| pick | `dataset = <repository>@<version>` in the run form |
| results | never written to a dataset: they go to `outputs-<pipeline>` |

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

`kaosctl` comes with kaos-lib (0.4.0rc23 at the time of writing; [install it](prepare-dataset.md#install-and-sign-in)).

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
(**Datasets** → the dataset → **Sharing**, by the owning project's admins); see [Use a dataset shared with you](shared-dataset.md). The hera guide has one page per task: [create](https://novelcore.github.io/hera-pipeline-template/dataset-create/), [add data](https://novelcore.github.io/hera-pipeline-template/dataset-upload/), [check](https://novelcore.github.io/hera-pipeline-template/dataset-validate/), [release](https://novelcore.github.io/hera-pipeline-template/dataset-release/), [download](https://novelcore.github.io/hera-pipeline-template/dataset-pull/), [share](https://novelcore.github.io/hera-pipeline-template/dataset-share/).

Every command prints the next step and the link to what it made. Exit codes: 0 ok, 1 something for
you to fix (one `ERROR:` line), 2 an invalid dataset.

!!! note "Who may do what"
    Reading a dataset needs read rights, which the pipeline has. Creating a dataset, pushing, and
    releasing a version are for people: a step can never write to a dataset.
