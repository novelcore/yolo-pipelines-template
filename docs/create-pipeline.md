# 3. Create a yolo pipeline

**Goal:** a yolo pipeline in your project, ready to run, and its folder on your computer.

A *pipeline* here is a ready-made set of steps (check settings, load data, train, quantize, register) that you run from a form. You get your own copy as a Git repository in your organisation, so you can change it later.

**Who can do it:** `dev`, `ops` or `admin` on the project. Creating the pipeline may need a project admin.

1. **Do:** in the KAOS console open **Pipelines** → **New pipeline** → the **yolo** card, choose the project and a name, and confirm.

    **You should see:** your pipeline in the list, with a status that ends at **Ready** ("Pipeline ready"). The first build takes several minutes.

2. **Do:** clone the repository and check out the `dev` branch.

    ```bash
    git clone https://github.com/<github-org>/<pipeline>.git && cd <pipeline> && git checkout dev
    ```

    **You should see:** the folder with `config/`, `steps/`, `pipeline.py` and `docs/`. This is where you run `kaosctl login` ([step 1](prepare-dataset.md#install-and-sign-in)).

3. **Do:** look at the tools on the pipeline's row: **Run**, **Runs in Argo**, **MLflow**, **Outputs**, **Datasets**.

    **You should see:** each opens in a new tab. *Argo* is the page that shows your runs as boxes. *MLflow* keeps the numbers and charts of every run. *Outputs* is the lakeFS repository `outputs-<pipeline>` where the weights land.

When you push a change to the repository, the pipeline is rebuilt and the next run uses it ([the hera cookbook](https://novelcore.github.io/hera-pipeline-template/) shows how to change a setting or a step).

**Next:** [run it](run-small-and-full.md).
