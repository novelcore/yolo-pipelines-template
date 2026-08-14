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
 && poetry install --only=main --no-root --no-interaction --no-ansi \
 # charset-normalizer repair: poetry lives in system site-packages
 # (virtualenvs.create false) and drags in a newer charset-normalizer whose
 # Cython-compiled extensions (cd/md .so, 3.5.0+) survive the locked
 # downgrade as ORPHANS no RECORD owns — compiled modules then shadow the
 # locked version's .py files -> "module 'charset_normalizer.md' has no
 # attribute 'CharInfo'" at runtime. --force-reinstall cannot remove
 # orphans, so: scorched-earth remove ALL remnants, clean-install the
 # exact locked version, and ASSERT the import at build time so this
 # class fails the build, never a training run.
 && CN_VER=$(grep -A1 'name = "charset-normalizer"' poetry.lock | sed -n 's/^version = "\(.*\)"/\1/p' | head -1) \
 && pip uninstall -y charset-normalizer \
 && rm -rf /usr/local/lib/python3.12/site-packages/charset_normalizer* \
 && pip install --no-cache-dir --no-deps "charset-normalizer==${CN_VER}" \
 && python -c "from charset_normalizer import md; import requests"

COPY app/ ./app/

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
