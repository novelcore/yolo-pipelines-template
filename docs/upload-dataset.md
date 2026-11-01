# Upload your dataset

The pipeline trains on a dataset that lives in lakeFS, the storage for your images and labels. A dataset is its own lakeFS repository, called `dataset-<name>`. Your project starts with one, `dataset-default`.

You do this once per dataset. After it is up, you can run the pipeline on it as many times as you like.

- A **version** is a tag: `v1`, `v2`, and so on. `latest` means the newest data on `main`.
- Changes reach `main` only through a **pull request**, so someone reviews them before they become the new data.
- Your results never go into a dataset. They go to your app's own repository, `outputs-<app>`.

!!! warning "First, you need an app"
    This page assumes you have **already created an app from this template**. Your app carries a small settings file, `.kubecore/dataset-config.yaml`, that tells the tool where your project's lakeFS is. Without an app there is nothing to clone and no lakeFS to upload to.

!!! note "Before you start"
    You need your dataset in the shape from [What you need](what-you-need.md): a `data.yaml`, an `images/` folder, and a `labels/` folder, each split into `train`, `val`, and `test`. Your app already knows your lakeFS, so you do not need any links or passwords.

## Step 1. Get your app on your computer

```bash
git clone https://github.com/<your-org>/<your-app>.git
cd <your-app>
```

Everything else on this page happens inside that folder.

## Step 2. Install the tool and sign in

The `kubecore` tool is a Python package. Install it once, inside a small private Python space for this project (a "virtual environment").

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install "git+https://github.com/novelcore/hera-pipeline-template@<commit>#subdirectory=packages/kubecore"
kubecore login
```

Your platform contact gives you the `<commit>`. On Windows, the second line is `.venv\Scripts\activate` instead. Your browser opens once and asks you to sign in. Log in, close the tab, and go back to the Terminal. The tool remembers you.

!!! tip "Why the virtual environment?"
    Many computers no longer let `pip install` change the system's own Python. They stop with a message that mentions `externally-managed-environment`. The virtual environment avoids that.

## Step 3. Create the dataset and send your files in

```bash
kubecore dataset create cats --classes cat,kitten      # makes dataset-cats with the starter layout
kubecore dataset pull dataset-cats@latest ./dataset-cats
# put your images in images/train, images/val, images/test and one label file per image in labels/...
kubecore dataset validate ./dataset-cats
kubecore dataset push ./dataset-cats dataset-cats -m "first 200 images"
```

`push` checks your folder, uploads it to a draft, and opens a **pull request**. It prints the link. Open it in lakeFS and merge it when it looks right: nothing changes on `main` until you do.

## Step 4. Release a version

```bash
kubecore dataset release dataset-cats     # tags the head of main as the next version: v1, then v2, ...
kubecore dataset versions dataset-cats
kubecore dataset list                     # every dataset you can see
```

The run form's `dataset` field is a dropdown of `dataset-<name>@<version>` choices. A new version shows up the next time your pipeline is released.

## What to remember

You choose `dataset-cats@v1` (or `@latest`) in the run form. The run reads exactly that version, and every result records the commit it came from.

Next, run the pipeline. Go to [Run the pipeline](run-it.md).

!!! tip "Uploading a new version later"
    Pull the dataset, change it, `push` again, merge the pull request, `release`. Old versions stay as they are, so a result always points back to its data.
