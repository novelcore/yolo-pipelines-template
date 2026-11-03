# Run the pipeline

This is the moment it all pays off. You open a web page, set a few fields, and press Submit. The pipeline then trains a model for you.

## Step 1. Open the run page

In the console, open **Pipelines**, find your app, and click **Run**. That opens the run page in Argo Workflows. From Argo itself: **Workflow Templates** → `<app>-pipeline` → **Submit**.

**Who can do it:** `dev`, `ops` or `admin` on the project ([cookbook 1](https://novelcore.github.io/hera-pipeline-template/cookbook/1-youre-in/)).

![The pipeline template page, with the Submit button](img/argo-submit.png)

## Step 2. The form

When you click Submit, a form opens. It has a lot of fields. **Every field has a default, except that you must pick your own `dataset`.** The default, `dataset-default@latest`, is empty in a new project, so a run on it stops at `config-validation`.

So the trick is simple. Change only the few fields you care about, and leave the rest alone.

![The open submit form](img/submit-form.png)

## Step 3. The fields that matter for a first run

Here are the handful worth setting. Everything else can stay as it is.

| Field | What it does | A good first value |
|---|---|---|
| `experiment-name` | Names your run so you can find it later | something like `my-first-run` |
| `dataset` | Which dataset to train on: `<dataset-repository>@<version>`, or `<owner>/<dataset>@<version>` for one another project shares | your released version, for example `dataset-spacecraft@v1` |
| `model` | Which model size to train | `yolov8n` to start. It is the small, fast one |
| `train-epochs` | How many rounds of training | a small number like `10` for your first test |
| `quantization-mode` | Whether to also make a smaller INT8 model | `none` for now |

That is enough for a first run. A small model, a few epochs, no quantization. It finishes quickly and proves the whole thing works end to end.

Every other field is one setting from `config/`, named by its path with dashes: `train.batch_size` is `train-batch_size`, `data.sample_size` is `data-sample_size`. The `<step>-class` fields choose the machine for each step ([Where it trains](where-it-trains.md)); the `<step>-cpu`, `<step>-mem` and `<step>-disk` fields ask for less of it. The read-only `pipeline-info` field at the top lists the classes.

![The key fields set: experiment-name, dataset, model, sample size](img/form-fields.png)

## Step 4. About the model choices

The `model` field is a dropdown. The options go from small and fast to big and accurate.

- `yolov8n` is the smallest. It trains fastest. Best for a first test.
- `yolov8s` is a middle option.
- `yolov11x` is the largest. It can be more accurate, but it trains much slower and needs more from the machine.

Start small. Move up once you know your dataset works.

## Step 5. Press Submit

When your few fields are set, scroll down and click **Submit**. The page switches to a live view of your run.

That is your part done. The pipeline takes it from here.

## About quantization

For your first runs, leave `quantization-mode` on `none`. The pipeline trains a model and saves it. Simple.

When you are ready for a smaller, faster model, come back and set it to `ptq` or `qat`. That turns on the two extra steps that make an INT8 model. You do not need this to get started.

!!! tip "Wondering which machine it uses?"
    By default training runs on a cluster GPU and you do not pick anything. If you need to choose, or send it to MeluXina, see [Where it trains](where-it-trains.md).

Next, go to [Where it trains](where-it-trains.md), or straight to [Watch it train](watch-it.md).
