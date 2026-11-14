# prepare-data

Writes the run's dataset as the dataset `prepared`: every file, or a reproducible subset of it. `model-training`, `qat-finetune` and `model-quantization` read exactly that version.

- **Runs on:** CPU, on the platform's own machines (a step that writes a dataset is not offered an HPC class yet).
- **Reads:** the setting `data` (`data-sample_size`, `data-seed`) and the dataset of the setting `dataset`, pinned by the platform when the run starts: `step.input("dataset")`.
- **Does:** without `data-sample_size` it keeps every file as it is (hard links when the disk is shared, so nothing is copied, and nothing is uploaded when the files equal an existing version). With `data-sample_size` = N it keeps N images of each split under `images/` and their `labels/`, ordered by a hash of `data-seed` and the file's path: the same seed keeps the same images on every machine. `data.yaml` and any file outside `images/` and `labels/` are kept.
- **A first run:** when the `dataset` is the project's default one and holds no `data.yaml`, `images/` or `labels/` (a new project), it writes a small sample made by `app/sample.py` instead (16 training and 4 validation pictures of a rectangle, with exact boxes and corner keypoints), and `data-sample_size` applies to it as to any dataset. Nothing is downloaded or committed: the same files come out of a fixed seed on every machine.
- **Writes:** the dataset `prepared` (`dataset-<pipeline>-prepared`), one version per run; its `latest` moves once, at the end of the run, and only when the person starting the run turns on `promote-prepared` on the run form (this is not the model alias `registration-promote_to`).
- **Hands on:** `sample_size`, `seed` and the images kept of each split.

```
steps/prepare_data/
  app/entry.py       reads the settings and the dataset, ends with step.result()
  app/subset.py      the choice of the images
  app/sample.py      the small sample for a first run (python app/sample.py <folder> writes it there)
  Dockerfile         the image (installs kaos-lib at KAOS_LIB_PACKAGE)
```

Try it: `kaosctl step run prepare-data --input dataset=./a-few-files`. Tests: `step_tests/test_prepare_data.py`, `step_tests/test_sample.py`.
