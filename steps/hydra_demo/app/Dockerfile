FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir PyYAML
COPY app/ ./app/
ENTRYPOINT ["python", "-m", "app.entry"]