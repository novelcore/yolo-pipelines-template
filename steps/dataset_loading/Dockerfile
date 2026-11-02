# dataset-loading step image.
# The platform builds this on push (kaniko -> Zot) and wires the resulting
# image into the WorkflowTemplate as image-dataset-loading. The in-image command
# is `python -m app.entry` — the same module you run locally.
FROM python:3.11-slim

WORKDIR /app

# Step runtime dependencies:
#   boto3            — S3 / lakeFS (S3-compatible) object access
#   pydantic*        — the service's typed config + domain models
#   PyYAML           — params.yaml + data.yaml parsing
RUN pip install --no-cache-dir \
    "boto3>=1.34,<2" \
    "pydantic>=2,<3" \
    "pydantic-settings>=2,<3" \
    "PyYAML>=6,<7"

# (requests, the lakeFS client's dependency, comes with kaos-lib below.)
# kaos-lib, the platform library (datasets, run outputs; formerly the kubecore package), from
# github.com/kaos-io/kaos-lib. The platform's CI passes KAOS_LIB_PACKAGE, the WHOLE pip requirement (the
# platform registry's wheel); the default is this template's pin. Only app/ is copied into the image.
ARG KAOS_LIB_PACKAGE="kaos-lib[zitadel] @ https://github.com/kaos-io/kaos-lib/archive/dc24f292029bb2a49a2cdf5c24eb169d7706d9a5.zip"
RUN pip install --no-cache-dir "${KAOS_LIB_PACKAGE}"

COPY app/ ./app/

# Argo runs: python -m app.entry --params <resolved params.yaml content> ...
ENTRYPOINT ["python", "-m", "app.entry"]
