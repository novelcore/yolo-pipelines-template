# 6. Use a dataset shared with you

**Goal:** train on a dataset that belongs to another project of your organisation.

A project can *share* a dataset with other projects: read-only, so you can train on it but never change it. Sharing is set by an admin of the owning project in the console (**Datasets** → the dataset → **Sharing**).

1. **Do:** list what is shared with you.

    ```bash
    kaosctl dataset list --shared
    ```

    **You should see:** the datasets other projects share, written `<owner>/dataset-<name>`. Nothing listed: ask the owner to share it with your project.

2. **Do:** pick it in the run form's `dataset` field, as `<owner>/dataset-<name>@<version>`, for example `team-a/dataset-spacecraft@v1`. The dropdown also lists shared datasets.

    **You should see:** the run trains on the owner's version. `prepare-data` reads the owner's version like one of your own and writes it as your project's dataset `prepared`, so the other steps read your project's copy.

3. **Do (optional):** look at the data on your computer.

    ```bash
    kaosctl dataset pull team-a/dataset-spacecraft@v1 ./shared
    ```

You cannot `push`, `release` or change a shared dataset. If the owner has not shared it with your project you get:

```
not shared with your project: project team-a's lakeFS refused it (...). Ask team-a to share the dataset with your project.
```

See [Troubleshooting](troubleshooting.md).
