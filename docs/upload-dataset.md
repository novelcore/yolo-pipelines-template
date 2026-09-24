# Upload your dataset

The pipeline trains on a dataset that lives in lakeFS. lakeFS is the storage for your images and labels. This page shows you how to put your dataset there, using the upload tool that ships inside your project.

You do this once per dataset. After it is up, you can run the pipeline on it as many times as you like.

!!! note "Before you start"
    You need your dataset in the shape from [What you need](what-you-need.md): a `data.yaml`, an `images/` folder, and a `labels/` folder, each split into `train`, `val`, and `test`. You also need the lakeFS link and the repo name from your platform contact.

## Step 1. Open your project and install the tool

The upload tool is already in your project, in a folder called `dataset_tools`. Open your Terminal, go into your project folder, and install it once.

```bash
pip install ./dataset_tools
```

You only do this the first time.

## Step 2. Tell it your lakeFS, once

The tool needs to know your project's lakeFS link and repo name. You have two ways to give it those.

The easy way, so you never type them again. Copy the example file and fill in the two values your contact gave you.

```bash
cp .kubecore/dataset-config.yaml.example .kubecore/dataset-config.yaml
```

Open that new file in your editor and set `lakefsUrl` and `repo`. Save it.

The other way is to pass them on the command line each time, with `--url` and `--repo`. Either works. The rest of this page assumes you filled in the file.

## Step 3. Start the upload

Run the tool and point it at your dataset folder.

```bash
python3 scripts/upload-dataset.py /path/to/your-dataset
```

Replace `/path/to/your-dataset` with the real folder on your computer. That uploads it as the dataset called `main`.

To keep more than one dataset, give each one a name after the folder. The name is what you will type as `data-ref` when you run.

```bash
python3 scripts/upload-dataset.py /path/to/your-dataset my-cats-v1
```

!!! warning "Same name replaces"
    Uploading again with a name you already used makes that dataset match your folder. Files you do not have on your computer are removed from it. The tool shows you which files and asks before it removes anything. To keep the old one, pick a new name.

If you skipped the config file in Step 2, add your links like this instead.

```bash
python3 scripts/upload-dataset.py /path/to/your-dataset \
  --url <your-lakefs-link> --repo <your-repo-name>
```

!!! info "Screenshot"
    A picture of the Terminal running the upload command will go here.

## Step 4. Sign in when the browser opens

The first time, the tool opens your web browser and asks you to sign in. This is the same sign in you use for the other pages. Log in, and the tool remembers you for next time.

!!! info "Screenshot"
    A picture of the browser sign in page will go here.

## Step 5. Let it upload

The tool checks your dataset, uploads the files, and saves them. You will see it count the files as it goes. When it finishes, it prints a line telling you the upload is done and which data ref to use when you run.

!!! info "Screenshot"
    A picture of the Terminal showing the upload finished, with the data ref line, will go here.

## Step 6. Check it in lakeFS

Open the lakeFS link in your browser. You should see your dataset there, with the `data.yaml`, `images`, and `labels` folders. That confirms it landed.

!!! info "Screenshot"
    A picture of the lakeFS page showing the uploaded dataset tree will go here.

## What to remember

The tool told you a **data ref**. It is the name you gave the dataset, or `main` if you did not give one. Write it down. You will type it into the run form in the next step, so the pipeline knows which dataset to train on.

Next, run the pipeline. Go to [Run the pipeline](run-it.md).

!!! tip "Uploading a new version later"
    To update the dataset, run the same command again with your changed folder. The tool uploads only what changed and keeps a history, so you can always go back.
