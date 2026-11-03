# A day with yolo

**Goal:** by the end of the day you have trained two models on your own labelled images, compared
them in MLflow, and picked the weights to keep.

**Who can do it:** `dev`, `ops` or `admin` on the project ([cookbook 1](https://novelcore.github.io/hera-pipeline-template/cookbook/1-youre-in/)).
Creating the app may need a project admin.

Each part below is short and links to the full page.

## Morning: access, app, tools

1. **Sign in** to the console and find your project. If you don't see it, an organisation admin
   adds you to a team ([cookbook 1](https://novelcore.github.io/hera-pipeline-template/cookbook/1-youre-in/)).
2. **Create the app:** console → **Pipelines** → **New pipeline** → the yolo card → the wizard
   (project, name). Wait until its status is **Ready** ("Pipeline ready"). The first build takes
   several minutes ([cookbook 2](https://novelcore.github.io/hera-pipeline-template/cookbook/2-your-apps/)).
3. **Clone it** and check out `dev`:

    ```bash
    git clone https://github.com/<github-org>/<app>.git && cd <app> && git checkout dev
    ```

4. **Open the tools** from the app's row on **Pipelines**: **Runs in Argo**, **MLflow**, **Outputs**,
   **Datasets** ([cookbook 3](https://novelcore.github.io/hera-pipeline-template/cookbook/3-your-tools/)).
5. **Install kaosctl** and sign in, inside the app folder
   ([cookbook 4](https://novelcore.github.io/hera-pipeline-template/cookbook/4-install-kaosctl/)):

    ```bash
    python3 -m venv ~/.venvs/kaos && . ~/.venvs/kaos/bin/activate
    pip install --index-url https://europe-west1-python.pkg.dev/kaos-dev-511509/kaos-lib-rc/simple/ \
        --extra-index-url https://pypi.org/simple/ "kaos-lib[zitadel,mlflow]>=0.4.0rc5"
    kaosctl login
    ```

## Late morning: your labelled images

6. **Create, label, upload**: `kaosctl dataset create spacecraft --classes spacecraft`, add
   `kpt_shape` and `flip_idx` to `data.yaml`, copy in images and one label file per image,
   `kaosctl dataset validate`, then `kaosctl dataset push`
   ([Upload images and labels](guide-upload-images.md)).
7. **Merge** the pull request in lakeFS and **Release version**: `dataset-spacecraft@v1`.

## Afternoon: two trainings

8. **Train A:** **Run** → `experiment-name` = `spacecraft-pose`, `dataset` =
   `dataset-spacecraft@v1`, `model` = `yolov8n`, `train-epochs` = `50` → **Submit**.
9. **Train B:** the same, with `train-epochs` = `100`. Or, to compare two class sets, release the
   other class set as `v2` and pick `dataset-spacecraft@v2`. Both runs can go at once.
10. **Watch** them in Argo. `qat-finetune` and `model-quantization` show as *skipped*, which is
    normal ([Watch it train](watch-it.md)).

Keep the same `experiment-name` for runs you want to compare: they then sit in one MLflow
experiment, and their weights sit side by side under `checkpoints/spacecraft-pose/`.

## Late afternoon: compare and pick

11. **Compare in MLflow:** experiment `spacecraft-pose` → tick both runs → **Compare**. Look at
    `train-epochs` against `val/mAP50_95`, `val/precision` and `val/recall`, and at their curves.
    The experiment's **Models** tab lists both runs' `spacecraft-pose-best` with their final scores
    ([Read your results](results.md)).
12. **Look at the weights yourself:** download both `best.pt` and evaluate them on the same data:

    ```bash
    kaosctl dataset pull outputs-<app>@main ./a --prefix checkpoints/spacecraft-pose/<run-a>/
    kaosctl dataset pull outputs-<app>@main ./b --prefix checkpoints/spacecraft-pose/<run-b>/
    kaosctl run --lock <mlflow-run-id-of-a> -- python my_eval.py ./a/best.pt ./b/best.pt
    ```

    `kaosctl run --lock` makes your script read exactly the dataset commit run A trained on
    ([cookbook 8](https://novelcore.github.io/hera-pipeline-template/cookbook/8-results-and-weights/)).
13. **Pick the weights:** in MLflow's model registry, open `object-pose-yolo`, find the version made
    from the winning run, and give it the alias `champion`. Next time you can do this on the form,
    with `registration-promote_to` = `champion` ([Promote a model](promote.md)). The weights of
    that version are at the lakeFS tag `model-object-pose-yolo-v<N>` in `outputs-<app>`.

## If something goes wrong

[When something breaks](troubleshooting.md) covers the yolo-specific problems.
[Cookbook 10](https://novelcore.github.io/hera-pipeline-template/cookbook/10-when-things-go-wrong/) covers everything else, organised by symptom.
