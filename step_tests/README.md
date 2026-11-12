# step_tests/ - tests of the step code

The steps import **kaos-lib** (`kaos_lib`), the platform library from
[kaos-io/kaos-lib](https://github.com/kaos-io/kaos-lib). Install the version the step images pin
(`KAOS_LIB_PACKAGE` in the step Dockerfiles), then run:

```bash
pip install "$(sed -n 's/^ARG KAOS_LIB_PACKAGE="\(.*\)"$/\1/p' steps/*/Dockerfile | head -1)" \
    pytest pillow pydantic psutil "mlflow>=3,<4"
python -m pytest
```

A step is run the way the platform runs it: `kaos_lib.testing.running` prepares its context and calls `main()` in the same process, and
`kaosctl step run` runs it on this machine. Nothing is uploaded and no login is needed.

| test | what it holds |
|---|---|
| `test_step_contract.py` | every entry starts with `step.start()`, ends with `step.result(`, and has no `os.environ`, `getenv`, `argparse` or `sys.argv` |
| `test_prepare_data.py` | `prepare-data` keeps every file or a reproducible subset: the same seed, the same files |
| `test_model_training.py`, `test_checkpoint_handover.py` | the training and the INT8 steps read their inputs with `step.input(...)` and save with `step.output(...)` |
| `test_model_registration.py` | registration reads where the results were saved from the result of `model-training` |
| `test_training_run.py` | a real one-epoch training on the CPU, started on the dataset `prepare-data` wrote (needs `torch` and `ultralytics`; skipped without them; nothing is downloaded) |
| `test_qat_resume.py` | the QAT fine-tune resumes from the checkpoint folder (needs `torch` and `torchao`) |
| `test_config_validation.py`, `test_form_bounds.py` | what `config-validation` refuses |
| `test_kaos_lib_version.py`, `test_repo_layout.py` | the image pins, `.kaos/compat.yaml`, and that an app is `pipeline.py`, `config/` and `steps/` |
| `test_docs_example.py` | the pipeline file and the step the guide of kaos-lib shows equal this template (needs `DOCS_CHECK_KAOS_LIB`; skipped without it) |

A step image copies only its `app/` folder. `test_repo_layout.py` fails if a copy of the platform's pipeline engine, or a step that
composes the settings, comes back: the platform runs its own.
