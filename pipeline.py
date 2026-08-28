"""Your ML pipeline: the DAG. Structure and nothing else.

Each step declares its name, the config sections it reads (reads=), whether it
needs a GPU (gpu=True), what it depends on (needs=), any small result files it
writes (outputs=), and an optional condition (when=). Every tunable parameter
lives in the config/ tree and reaches the submit form automatically — there is
no parameter wiring here.

Conditions reference derived form parameters by name (config path,
dots->dashes): quantization.mode -> quantization-mode.

See README.md / DEVELOPER.md to add a step or a parameter.
"""

from pathlib import Path

from kubecore.authoring import pipeline, step  # platform-owned, do not edit

HERE = Path(__file__).parent

# The name you pass here is for local readability only — the platform RENAMES
# the released WorkflowTemplate to "{your-app}-pipeline" at CI render time, so
# each KubeApp gets its own uniquely-named WFT (no collisions across apps in a
# shared namespace). You don't need to change it per app.
with pipeline("ml-pipeline") as p:
    validate = step("config-validation", reads=["data", "model"])
    preflight = step("preflight", reads=["experiment", "train"], needs=[validate])
    demo = step("hydra-demo", reads=["experiment", "train", "evaluation"], needs=[validate])
    hello    = step("hello-world", reads=["experiment"], needs=[validate])
    load = step("dataset-loading", reads=["data"], needs=[validate],
                outputs=["data-yaml", "manifest-summary"])
    summary = step("data-summary", reads=["data", "summary"], needs=[load])
    # hpc_time_limit: Slurm wall-clock when this step is placed on a meluxina-*
    # class (DEVELOPER.md §8.1). Size it to the longest expected epoch budget.
    train = step("model-training", gpu=True, needs=[load], hpc_time_limit="12h",
                 reads=["experiment", "data", "model", "train", "image_processing", "logging"],
                 outputs=["training-result"])
    qat = step("qat-finetune", gpu=True, needs=[train], disk="20Gi",
               reads=["experiment", "train", "quantization"],
               when="{{workflow.parameters.quantization-mode}} == qat",
               outputs=["qat-result"])
    quant = step("model-quantization", needs=[train, qat],
                 reads=["experiment", "quantization"],
                 when="{{workflow.parameters.quantization-mode}} != none",
                 outputs=["quantization-result"])
    register = step("model-registration", needs=[train, quant],
                    reads=["data", "model", "registration"],
                    outputs=["registration-result"])

if __name__ == "__main__":
    p.write(HERE / "out" / "raw-workflow-template.yaml")
