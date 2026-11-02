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

# (requests, the lakeFS client's dependency, comes with the kubecore package below.)
# The kubecore PACKAGE (datasets, run outputs), pinned to a commit of hera-pipeline-template.
# It shares the import name `kubecore` with the vendored engine at the repo root, which is
# NOT in this image: only app/ is copied. Bump the commit to take a newer package.
ARG KUBECORE_PKG_COMMIT=819920b6d73e0fe48db1cbd5643636cb07b57209
RUN pip install --no-cache-dir \
    "https://github.com/novelcore/hera-pipeline-template/archive/${KUBECORE_PKG_COMMIT}.zip#subdirectory=packages/kubecore"

COPY app/ ./app/

# Argo runs: python -m app.entry --params <resolved params.yaml content> ...
ENTRYPOINT ["python", "-m", "app.entry"]
