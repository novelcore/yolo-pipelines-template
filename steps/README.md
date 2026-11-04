# steps/ - the pipeline's step code

One directory per step of the pipeline (`pipeline.py` lists them), each with its own `app/` and `Dockerfile`. A step image carries only its own `app/`. Each step has a README: [config-validation](config_validation/README.md), [dataset-loading](dataset_loading/README.md), [model-training](model_training/README.md), [qat-finetune](qat_finetune/README.md), [model-quantization](model_quantization/README.md), [model-registration](model_registration/README.md).

A step receives the fully resolved experiment config (`--params`) and should read only the config sections it declares with `reads=` in `pipeline.py`. That declaration is checked when the pipeline is built, so a renamed or deleted section fails loudly instead of breaking the step mid-run. To add a step see [ADD-A-STEP.md](../ADD-A-STEP.md).
