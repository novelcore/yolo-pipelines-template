# model-training step image — REAL training (kubeline trainer vendored into app/).
# Deps are installed via POETRY from the vendored pyproject.toml + poetry.lock, so
# every dependency version is PINNED IDENTICAL to the kubeline pipeline (parity).
# The platform builds this on push (kaniko -> Zot) and wires it as
# image-model-training. In-image command: python -m app.entry --params <params>.
FROM python:3.12-slim

WORKDIR /app

# System libs required by OpenCV / PyTorch / pynvml.
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
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

# The kubecore PACKAGE (datasets, run outputs), pinned to a commit of hera-pipeline-template.
# It shares the import name `kubecore` with the vendored engine at the repo root, which is
# NOT in this image: only app/ is copied. Bump the commit to take a newer package.
ARG KUBECORE_PKG_COMMIT=5e718a1e957bc9c5a1d6c13c12afe687e91d05b1
RUN pip install --no-cache-dir \
    "https://github.com/novelcore/hera-pipeline-template/archive/${KUBECORE_PKG_COMMIT}.zip#subdirectory=packages/kubecore"

COPY app/ ./app/

# GPU runtime: this slim base has no CUDA libs. GKE mounts the host NVIDIA driver
# into the container at /usr/local/nvidia/lib64; without it on LD_LIBRARY_PATH,
# torch cannot find libcuda.so and silently trains on CPU even on a T4 node
# (kubeline RCA yolo-green3, 2026-06-26). Put the driver libs on the path.
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/nvidia/lib:${LD_LIBRARY_PATH}
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV PATH=/usr/local/nvidia/bin:${PATH}

# Argo runs: python -m app.entry --params <resolved params.yaml content> ...
ENTRYPOINT ["python", "-m", "app.entry"]
