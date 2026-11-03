# yolo-pipelines-template

**The yolo pipelines KubeApp template.** A KubeApp with `kubeAppTemplateRef: yolo-pipelines` gets
a complete YOLO pose pipeline: dataset loading, YOLOv8/YOLOv11 training with authenticated MLflow
tracking, optional QAT fine-tuning, INT8 TFLite export (PTQ or QAT), and model registration.
Results go to the app's own `outputs-<app>` lakeFS repository.

It is built on the scaffold [`hera-pipeline-template`](https://github.com/novelcore/hera-pipeline-template)
and is the worked use case of its guide, *the engineer's cookbook*. Start from the scaffold to
write your own pipeline, and read this one to see a real one. The platform injects the
environment's MLflow, lakeFS and compute context at render time. Nothing environment-specific
lives in this repository.

**The guide: [novelcore.github.io/yolo-pipelines-template](https://novelcore.github.io/yolo-pipelines-template/)**.
Its [day with yolo](https://novelcore.github.io/yolo-pipelines-template/worked-day/) takes you
from a new app to two trained models compared in MLflow.

## The pipeline

Seven steps; `qat-finetune` and `model-quantization` run only when `quantization-mode` is not
`none`:

```
config-validation → dataset-loading → data-summary
                                    → model-training → qat-finetune → model-quantization → model-registration
```

The whole pipeline is `pipeline.py`. **The `config/` tree is the submit form**: every plain value in
`config/` becomes a field, and every group folder (`config/model/`, `config/train/optimizer/`, …)
becomes a dropdown.

```
pipeline.py            the steps (reads=, gpu=, needs=, when=, hpc_time_limit=)
config/                every setting: a field or a dropdown on the run form
steps/<name>/          one folder per step: app/ and a Dockerfile
kubecore/              the platform's engine: seeded, do not edit
tests/, step_tests/    engine tests; step-code tests (DEVELOPER.md §15)
docs/                  the guide (mkdocs)
```

## Quick start

```bash
python3 -m venv ~/.venvs/kaos && . ~/.venvs/kaos/bin/activate
pip install --index-url https://europe-west1-python.pkg.dev/kaos-dev-511509/kaos-lib-rc/simple/ \
    --extra-index-url https://pypi.org/simple/ "kaos-lib[zitadel,mlflow]>=0.4.0rc5"
kaosctl login                                              # inside the app folder
kaosctl dataset create spacecraft --layout yolo --classes spacecraft   # yolo layout; add kpt_shape to data.yaml
kaosctl dataset push ./dataset-spacecraft dataset-spacecraft -m "first images"
kaosctl dataset release dataset-spacecraft                 # after merging the pull request
./run.sh                                                   # render the pipeline locally, no cluster
```

`kaosctl` needs kaos-lib **0.4.0rc5** or newer. The library is documented in
[the kaos-lib reference](https://storage.googleapis.com/kaos-dev-511509-kaos-lib/docs/index.html).

## Where to read next

| you want to | read |
|---|---|
| train for the first time | [first dataset to a trained model](https://novelcore.github.io/yolo-pipelines-template/first-run/) |
| compare two trainings and pick the weights | [a day with yolo](https://novelcore.github.io/yolo-pipelines-template/worked-day/) |
| how the steps read data, write results and log models | [docs/datasets-in-the-steps.md](docs/datasets-in-the-steps.md) |
| access, apps, tools, and everything the templates share | [the engineer's cookbook](https://novelcore.github.io/hera-pipeline-template/) |
| add your first step | [ADD-A-STEP.md](ADD-A-STEP.md) |
| the operating manual: config, tests, HPC | [DEVELOPER.md](DEVELOPER.md) |
| how the platform works inside | [MECHANISMS.md](MECHANISMS.md), [PLATFORM.md](PLATFORM.md) |

## License

This template is licensed under the **GNU Affero General Public License v3.0**
(`AGPL-3.0-only`). See [LICENSE](LICENSE) for the full terms.

It depends on a training library that is itself AGPL-3.0, and three files adapt that
library's internals (`steps/model_training/app/services/s3_dataset.py`,
`s3_pose_trainer.py` and `s3_pose_validator.py`), so the template as a whole carries
that licence. Releases before this change were published under Apache-2.0, which may
be redistributed under AGPL-3.0.

An app created from this template inherits these terms for the template's code.
