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

# The kubecore PACKAGE (datasets, run outputs), pinned to a commit of hera-pipeline-template.
# It shares the import name `kubecore` with the vendored engine at the repo root, which is
# NOT in this image: only app/ is copied. Bump the commit to take a newer package.
ARG KUBECORE_PKG_COMMIT=0136a205a82fbe4ed4a82037329e048f58e7e740
RUN pip install --no-cache-dir \
    "https://github.com/novelcore/hera-pipeline-template/archive/${KUBECORE_PKG_COMMIT}.zip#subdirectory=packages/kubecore"

COPY app/ ./app/

# Argo runs: python -m app.entry --params <resolved params.yaml content>
ENTRYPOINT ["python", "-m", "app.entry"]
