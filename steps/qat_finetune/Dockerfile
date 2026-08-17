# qat-finetune step image — REAL QAT service (kubeline vendored into app/).
# Mirrors the kubeline qat_finetune image: Poetry base deps + the PINNED CUDA/QAT
# stack via pip (exact versions, spike-tested — CON-01), plus a build-time import
# check so a broken QAT import path fails `docker build` in minutes, not 1-2h into
# a run. In-image command: python -m app.entry --params <params> ...
FROM python:3.11-slim

WORKDIR /app

# System libraries required by OpenCV, PyTorch, and litert-torch.
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Dependency install — poetry NEVER touches the runtime python.
# History (kubecore-operator#1008 thread): installing poetry into system
# site-packages let poetry's own requests/charset-normalizer stack collide
# with the locked versions — orphaned compiled extensions survived every
# repair attempt and resurfaced on cold (cache-less) builds as
# "module 'charset_normalizer.md' has no attribute 'CharInfo'" at runtime.
# Now poetry lives in its own venv (/opt/poetry) and is used ONLY to export
# the lock; the runtime python gets exactly poetry.lock via plain pip.
# The final import assert makes any residue fail the BUILD, never a run.
RUN python -m venv /opt/poetry \
 && /opt/poetry/bin/pip install --no-cache-dir poetry poetry-plugin-export
COPY pyproject.toml poetry.lock* README.md ./
RUN /opt/poetry/bin/poetry export --only=main -f requirements.txt --output /tmp/requirements.txt \
 && pip install --no-cache-dir -r /tmp/requirements.txt \
 && rm -rf /opt/poetry /tmp/requirements.txt \
 && python -c "from charset_normalizer import md; import requests"

# Pinned CUDA + QAT stack via pip (CON-01 — exact versions, custom torch index).
COPY requirements-cuda.txt requirements-qat.txt ./
RUN pip install --no-cache-dir -r requirements-cuda.txt
RUN pip install --no-cache-dir -r requirements-qat.txt

COPY app/ ./app/

# Build-time import verification (CPU — no GPU required): imports the actual
# symbols qat_service.py uses, so a broken import path fails the build fast.
RUN python -c "\
import torch, torchvision, torchao; \
from ultralytics import YOLO; \
from torchao.quantization.pt2e.quantize_pt2e import prepare_qat_pt2e, convert_pt2e; \
from torchao.quantization.pt2e import allow_exported_model_train_eval; \
from litert_torch.quantize.pt2e_quantizer import PT2EQuantizer, get_symmetric_quantization_config; \
import inspect; assert 'fold_quantize' in inspect.signature(convert_pt2e).parameters, 'convert_pt2e lost fold_quantize (CON-01)'; \
print('torch', torch.__version__, '| torchao', torchao.__version__, '| QAT import verification passed')"

# Optional GPU smoke-test tool (not run at build time).
COPY tools/ ./tools/

# GPU runtime: slim base has no CUDA libs; GKE mounts the driver at /usr/local/nvidia.
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/nvidia/lib:${LD_LIBRARY_PATH}
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV PATH=/usr/local/nvidia/bin:${PATH}

ENTRYPOINT ["python", "-m", "app.entry"]
