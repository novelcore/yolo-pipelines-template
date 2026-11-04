# dataset-loading

Turns the `dataset` pick into one lakeFS commit and describes the data for the later steps.

- **Outputs:** `manifest-summary` (including `dataset_commit`) and `data-yaml`. Every later step reads that commit, so a run is not affected by data merged while it runs.
- **Settings:** `data-source`, `data-sample_size`, `data-seed`, `data-path_override` ([docs/parameters.md](../../docs/parameters.md)); models in `app/models/dataset.py`.
- **How data moves:** [docs/datasets-in-the-steps.md](../../docs/datasets-in-the-steps.md).
