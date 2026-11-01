"""Where a pipeline's outputs go: the app's own lakeFS outputs repository.

Checkpoints, quantized models and the MeluXina stage-out hand-off are written to
`outputs-<app>` on `main`, never to the dataset repository. The app name comes
from the platform pipeline-context (`app`), so no step needs to know it: the
enhancer injects the result as CHECKPOINT_BUCKET / CHECKPOINT_PREFIX and mirrors
it in `platform.checkpoints.*`. Each step then writes under
`checkpoints/<experiment>/<run>/` (or `quantization/<experiment>/<run>/`) and
commits once per run, naming the run in the commit message.
"""

OUTPUTS_REPO_PREFIX = "outputs-"
OUTPUTS_REF = "main"  # the branch outputs are committed on
CHECKPOINTS_DIR = "checkpoints"


def outputs_repo(ctx: dict) -> str:
    """The app's outputs repository name: `outputs-<app>`."""
    app = ctx.get("app")
    if not app:
        raise ValueError("context is missing 'app' — cannot derive the outputs repository")
    return f"{OUTPUTS_REPO_PREFIX}{app}"


def checkpoint_prefix() -> str:
    """`{branch}/checkpoints`: the lakeFS key prefix (first segment = branch)
    the steps append `<experiment>/<run>/` to."""
    return f"{OUTPUTS_REF}/{CHECKPOINTS_DIR}"
