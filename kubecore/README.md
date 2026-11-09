# kubecore/ - the pipeline engine (provided by the platform, do not edit)

This folder is the engine that turns your `pipeline.py` and `config/` tree into the pipeline you run from the
form. It is provided and replaced by the platform when you upgrade, so an edit here is lost. You use it only through
`pipeline.py` (`from kubecore.authoring import pipeline, step`) and the local preview (`./run.sh`).

| module | what it does for you |
|---|---|
| `authoring.py` | `step()` and `pipeline()`: your steps and how they connect |
| `derive_tree.py` | turns the `config/` tree into the run form |
| `compose.py` | in the pipeline's first step, merges the form with the tree into the one `params.yaml` every step reads |
| `enhance.py` | adds what the platform owns (images, sizing, machine classes); it refuses a `pipeline.py` that sets `serviceAccountName`, `podSpecPatch` or a reserved template name |

Tests of the engine: `python -m pytest tests` ([DEVELOPER.md](../DEVELOPER.md), section 15).
