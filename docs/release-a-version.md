# 2. Release a version and pick it in the run form

**Goal:** the data you merged becomes an immutable version, `v1`, that you can choose in the run form.

A *version* is a name (`v1`, `v2`, ...) for the exact state of the dataset at one moment. A run trains on one version, so you can always say which data a model saw. `@latest` means "whatever is merged right now".

## Release

1. **Do:** after merging the pull request ([step 1](prepare-dataset.md)), release:

    ```bash
    kaosctl dataset release dataset-spacecraft
    kaosctl dataset versions dataset-spacecraft
    ```

    Or click **Release version** on the dataset in the console's **Datasets** page.

    **You should see:** `v1` (the next release is `v2`, and so on). If nothing was merged since the last release you get `nothing new to release: the head of dataset-spacecraft main is already v1. Merge a pull request first.`

## Pick it in the run form

2. **Do:** open the run form of your pipeline ([Run the pipeline](run-small-and-full.md)) and look at the `dataset` field.

    **You should see:** a dropdown with your datasets and versions, newest version first, then `@latest`: `dataset-spacecraft@v2`, `dataset-spacecraft@v1`, `dataset-spacecraft@latest`. A new release shows up within about a minute; reopen the form if it is not there yet.

3. **Do:** choose `dataset-spacecraft@v1`.

    **You should see:** the run trains on exactly the data of `v1`, even if you merge more data while it runs.

!!! note "Not `dataset-default`"
    The form starts on `dataset-default@latest`, which is empty in a new project. A run on it stops in `model-training` with "has no data.yaml at the root of its repository". Always pick your own version.

A dataset shared with you by another project appears in the same dropdown: [Use a shared dataset](shared-dataset.md).
