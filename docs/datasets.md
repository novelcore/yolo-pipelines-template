# Datasets

A dataset is a **lakeFS repository** named `dataset-<name>`. Your project starts with one,
`dataset-default`.

| thing | what it is |
|---|---|
| version | a tag on `main`: `v1`, `v2`, … |
| `latest` | the newest data on `main` |
| change | a pull request into `main`, reviewed and merged by a person |
| pick | `dataset = <repository>@<version>` in the run form |
| results | never written to a dataset: they go to `outputs-<app>` |

A run reads **one commit**: the one its pick resolves to. Merging new data during a run does not
change what that run reads.

## The layout this pipeline reads

```
data.yaml                 path, train, val, test, names, kpt_shape, flip_idx
images/{train,val,test}/   the images
labels/{train,val,test}/   one .txt per image: class x y w h, then the keypoints
```

at the **root** of the repository. Anything else in the repository (a README, notes) is ignored.

## Templates are optional

`kubecore dataset create <name>` starts a repository from the `ultralytics` template: the starter
layout above plus the checks that `validate` and `push` run. You can do without it:

- `kubecore dataset create <name> --template none` makes an empty `dataset-<name>`.
- A repository named `dataset-<something>` created by hand in lakeFS becomes a dataset after its
  first commit.
- A repository with **no template label validates nothing**; `push` takes the files as they are.
  Pass `--format ultralytics` to `push` or `validate` to get the YOLO checks anyway.
- Code can read **any** repository with `datasets.materialize("<repo>@<ref>")`, `<ref>` being `vN`,
  `latest`, a branch, a tag or a commit id. The run form's dropdown lists the `dataset-*` repositories.

## The commands

```bash
kubecore login
kubecore dataset create <name> [--classes a,b] [--template ultralytics|none]
kubecore dataset list
kubecore dataset versions <repo>
kubecore dataset pull <repo>@<ref> [dir] [--prefix folder/]
kubecore dataset validate <dir> --format ultralytics
kubecore dataset push <dir> <repo> -m "message" [--format ultralytics]
kubecore dataset release <repo>
```

Every command prints the next step and the link to what it made. Exit codes: 0 ok, 1 something for
you to fix (one `ERROR:` line), 2 an invalid dataset.

!!! note "Who may do what"
    Reading a dataset needs read rights, which the pipeline has. Creating a dataset, pushing, and
    releasing a version are for people: a step can never write to a dataset.
