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

RUN pip install --no-cache-dir poetry

# Deps first (cached on the lock) — torch + the full CUDA stack (~3GB) via
# ultralytics, plus MLflow, boto3/GCS, psutil/pynvml, and the Zitadel MLflow-auth
# deps. --no-root: install only third-party deps; the step runs `python -m app.entry`
# directly (app/ is on WORKDIR), so the root package need not be installed.
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
