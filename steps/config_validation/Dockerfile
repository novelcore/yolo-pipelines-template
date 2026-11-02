# config-validation step image (the pipeline's first developer step).
# The platform builds this on push (kaniko -> Zot) and wires the resulting
# image into the WorkflowTemplate as image-config-validation. The in-image
# command is `python -m app.entry` — the same module you run locally.
FROM python:3.11-slim

WORKDIR /app

# Step runtime dependencies:
#   boto3    — S3 checks of an s3:// dataset override, pretrained weights and checkpoints
#   requests — MLflow /health liveness check
#   PyYAML   — params.yaml parsing
RUN pip install --no-cache-dir \
    "boto3>=1.34,<2" \
    "requests>=2.31,<3" \
    "PyYAML>=6,<7"

# kaos-lib, formerly the kubecore package (datasets, run outputs). The PLATFORM picks it:
# the CI passes --build-arg KAOS_LIB_PACKAGE=<requirement> on every build, inside the
# range in .kaos/compat.yaml (docs/kaos-lib-versions.md). The value is the whole pip
# requirement (name, extras, version or URL), so the platform can move it to a kaos-lib
# release with no change here. This default is only for a build on your computer: until
# kaos-lib's first release it is a commit of hera-pipeline-template, where the package is
# still named `kubecore`. It shares the import name `kubecore` with the vendored engine at
# the repo root, which is NOT in this image: only app/ is copied.
ARG KAOS_LIB_PACKAGE="kubecore @ https://github.com/novelcore/hera-pipeline-template/archive/0136a205a82fbe4ed4a82037329e048f58e7e740.zip#subdirectory=packages/kubecore"
RUN pip install --no-cache-dir "${KAOS_LIB_PACKAGE}"

COPY app/ ./app/

# Argo runs: python -m app.entry --params <resolved params.yaml content>
ENTRYPOINT ["python", "-m", "app.entry"]
