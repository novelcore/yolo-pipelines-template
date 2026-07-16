# preflight — a lightweight CPU gate that echoes the resolved config summary
# before the heavy steps run. The exact shape ADD-A-STEP.md describes.
FROM python:3.11-slim
WORKDIR /work
RUN pip install --no-cache-dir "PyYAML>=6,<7"
COPY app/ ./app/
ENTRYPOINT ["python", "-m", "app.entry"]
