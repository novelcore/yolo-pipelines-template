# steps/ - the pipeline's step code

One directory per step of the pipeline (`pipeline.py` lists them), each with its own `app/` and `Dockerfile`. A step image carries only its own `app/`. Each step has a README: [config-validation](config_validation/README.md), [prepare-data](prepare_data/README.md), [model-training](model_training/README.md), [qat-finetune](qat_finetune/README.md), [model-quantization](model_quantization/README.md), [model-registration](model_registration/README.md).

`app/names.py` is the same file in every step that names something in MLflow (the app's name first, then the setting); `step_tests/test_names.py` fails when the copies differ.

A step starts with `step.start()`, reads its inputs with `step.input(...)` and the earlier steps' results with `step.upstream(...)`, writes into `step.output(...)` and ends with `step.result(...)` ([the step API](https://storage.googleapis.com/kaos-dev-511509-kaos-lib/docs/reference.html#step-api-kaos_libstep)). It takes no argument and reads no variable. `ctx.params` holds only the config sections the step names in `reads=` in `pipeline.py`; that declaration is checked when the pipeline is built, so a renamed or deleted section fails loudly instead of breaking the step mid-run. To add a step see [Your first step](https://storage.googleapis.com/kaos-dev-511509-kaos-lib/docs/first-step.html).
