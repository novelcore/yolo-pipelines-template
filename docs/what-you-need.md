# What you need

A few things to set up once, before your first run. Take it slow. Each item has a reason.

## Access to the tools

You will use two web pages and one small tool. Your platform contact gives you the links and the access.

- **The run page.** This is where you start a training job and watch it. It is run by a tool called Argo Workflows. You do not need to learn it.
- **The results page.** This is where the numbers and charts land. It is called MLflow.
- **lakeFS.** This is the storage where your dataset lives. You upload your images and labels here.

Ask your contact for the link to each one, plus confirmation that your account can sign in. Without access you cannot upload data or start a run.

## A computer with Python

To upload a dataset you run a small tool on your own computer. It needs Python.

- Most Macs and Linux machines already have Python. To check, open your Terminal and type `python3 --version`. If you see a number, you are set.
- If not, download it from [python.org/downloads](https://www.python.org/downloads/) and install it with the default options.

You only need this for the upload step. Running the pipeline itself happens on the web page, not on your computer.

## Your dataset, in the right shape

The pipeline trains on a YOLO pose dataset. On your computer it should look like this.

```
your-dataset/
├── data.yaml
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

A `data.yaml` file at the top. An `images` folder and a `labels` folder, each split into `train`, `val`, and `test`. If your data is already in this shape, you are ready. If not, ask your contact for help getting it there.

## A quick checklist

- [ ] I have the links to the run page, the results page, and lakeFS.
- [ ] My account can sign in to each one.
- [ ] Python is installed (`python3 --version` shows a number).
- [ ] My dataset has `data.yaml`, `images/`, and `labels/` in the shape above.

When these are done, read [What the pipeline does](the-pipeline.md), then [Upload your dataset](upload-dataset.md).
