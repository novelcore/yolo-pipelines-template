# model-quantization step image — REAL quantization service (kubeline vendored
# into app/). Mirrors the kubeline model_quantization image: Poetry base deps +
# the Ultralytics TFLite INT8 export toolchain (tensorflow/onnx2tf/...) baked in
# from NVIDIA's index (Ultralytics would otherwise pip-install these at RUNTIME,
# which fails in an offline pod), + a build-time import check.
# In-image command: python -m app.entry --params <params> ...
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir poetry

# Base deps via Poetry (pinned to the vendored kubeline lock). --no-root: the step
# runs `python -m app.entry` directly, so the root package need not be installed.
COPY pyproject.toml poetry.lock* README.md ./
RUN poetry config virtualenvs.create false \
 && poetry install --only=main --no-root --no-interaction --no-ansi

COPY app/ ./app/

# Root install (fast — deps already present) so the package's entry_points
# register, incl. the mlflow.request_auth_provider "zitadel" plugin used for
# MLflow auth (kubecore-operator#868). Without this the zitadel provider is
# undiscoverable and the (fatal) Ultralytics MLflow callback kills the run.
RUN poetry install --only=main --no-interaction --no-ansi

# Ultralytics TFLite INT8 export toolchain (PTQ path). Baked at build time so the
# offline pod does not pip-install at runtime. Versions per the ultralytics 8.4.x
# check_requirements window + onnx2tf's tested stack. onnx_graphsurgeon is on
# NVIDIA's index, not PyPI.
RUN pip install --no-cache-dir --extra-index-url https://pypi.ngc.nvidia.com \
    "tensorflow==2.19.0" \
    "tf-keras<=2.19.0" \
    "onnx2tf>=1.26.3,<1.29.0" \
    "onnx>=1.12.0,<2.0.0" \
    "onnxslim>=0.1.82" \
    "onnxruntime>=1.18.1" \
    "onnx-graphsurgeon>=0.3.26" \
    "sng4onnx>=1.0.1" \
    "protobuf>=5,<6" \
    "ai-edge-litert>=1.2.0"

# Build-time verification: the export toolchain must import (fails the build fast
# instead of at runtime).
RUN python -c "\
import tensorflow, tf_keras, onnx, onnx2tf, onnxslim, onnxruntime, sng4onnx, onnx_graphsurgeon; \
from ultralytics import YOLO; \
print('tensorflow', tensorflow.__version__, '| export toolchain import OK')"

ENTRYPOINT ["python", "-m", "app.entry"]
