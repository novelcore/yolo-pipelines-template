# A day with yolo

**Goal:** by the end of the day you have trained two models on your own labelled images, compared them in MLflow, and picked the weights to keep. Each part links to the full page.

**Who can do it:** `dev`, `ops` or `admin` on your project.

## Morning: pipeline and tools

1. **Create the pipeline:** console → **Pipelines** → **New pipeline** → the yolo card. Wait for **Ready**. **You should see** it in your list ([Create a yolo pipeline](create-pipeline.md)).
2. **Clone it** and install `kaosctl`, then sign in inside the folder. **You should see** `kaosctl 0.4.0rc23` and your browser asking you to sign in:

    ```bash
    git clone https://github.com/<github-org>/<pipeline>.git && cd <pipeline> && git checkout dev
    python3 -m venv ~/.venvs/kaos && . ~/.venvs/kaos/bin/activate
    pip install --index-url https://europe-west1-python.pkg.dev/kaos-dev-511509/kaos-lib-rc/simple/ \
        --extra-index-url https://pypi.org/simple/ "kaos-lib[zitadel,mlflow]==0.4.0rc23"
    kaosctl login
    ```

## Late morning: your labelled images

3. **Create, label, check, upload**: `kaosctl dataset create spacecraft --layout yolo --classes spacecraft`, add the images, labels and `kpt_shape` / `flip_idx`, `kaosctl layout validate ./dataset-spacecraft --layout yolo`, `kaosctl dataset push`. **You should see** `is a valid yolo dataset` and a pull request link ([Prepare and upload](prepare-dataset.md)).
4. **Merge** the pull request and release: `kaosctl dataset release dataset-spacecraft`. **You should see** `v1` ([Release a version](release-a-version.md)).

## Afternoon: two trainings

5. **Train A:** **Run** → `experiment-name` = `spacecraft-pose`, `dataset` = `dataset-spacecraft@v1`, `model` = `yolov8n`, `train-epochs` = `50` → **Submit**.
6. **Train B:** the same with `train-epochs` = `100`. Or release other data as `v2` and pick `dataset-spacecraft@v2`. Both runs can go at once.
7. **Watch** them in Argo. **You should see** `qat-finetune` and `model-quantization` *skipped*, which is normal ([Watch it train](watch-it.md)).

Keep the same `experiment-name` for runs you want to compare: they then sit in one MLflow experiment, and their weights sit side by side under `checkpoints/spacecraft-pose/`.

## Late afternoon: compare and pick

8. **Compare in MLflow:** experiment `spacecraft-pose` → tick both runs → **Compare**. **You should see** `train-epochs` against `val/mAP50_95`, `val/precision` and `val/recall`, and their curves. The experiment's **Models** tab lists both runs' `spacecraft-pose-best` with their final scores ([Results](results.md)).
9. **Look at the weights yourself:** download both `best.pt` and evaluate them on the same data. `kaosctl run --lock` makes your script read exactly the dataset commit run A trained on.

    ```bash
    kaosctl dataset pull outputs-<pipeline>@main ./a --prefix checkpoints/spacecraft-pose/<run-a>/
    kaosctl dataset pull outputs-<pipeline>@main ./b --prefix checkpoints/spacecraft-pose/<run-b>/
    kaosctl run --lock <mlflow-run-id-of-a> -- python my_eval.py ./a/best.pt ./b/best.pt
    ```

10. **Pick the weights:** in MLflow's model registry, open `object-pose-yolo`, find the version made from the winning run, and give it the alias `champion`. Next time, set `registration-promote_to` = `champion` on the form ([Promote a model](promote.md)). That version's weights are at the lakeFS tag `model-object-pose-yolo-v<N>` in `outputs-<pipeline>`.

## If something goes wrong

[When something breaks](troubleshooting.md).
