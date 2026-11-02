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

# kaos-lib, the platform library (datasets, run outputs; formerly kaos-lib), from
# github.com/kaos-io/kaos-lib. The platform's CI may pass KAOS_LIB_PACKAGE (a pip spec suffix:
# "@ <url>" or "==X.Y.Z"); the default is this template's pin. Only app/ is copied into the image.
ARG KAOS_LIB_PACKAGE="@ https://github.com/kaos-io/kaos-lib/archive/dc24f292029bb2a49a2cdf5c24eb169d7706d9a5.zip"
RUN pip install --no-cache-dir "kaos-lib[zitadel] ${KAOS_LIB_PACKAGE}"

COPY app/ ./app/

# Argo runs: python -m app.entry --params <resolved params.yaml content>
ENTRYPOINT ["python", "-m", "app.entry"]
