# config-validation step image (the pipeline's first step).
# The platform builds this on push and wires the resulting image into the pipeline. The in-image
# command is `python -m app.entry` (no arguments): the same module `kaosctl step run config-validation` runs.
FROM python:3.11-slim

WORKDIR /app

# kaos-lib, the platform library (github.com/kaos-io/kaos-lib). The platform's CI passes KAOS_LIB_PACKAGE, the
# WHOLE pip requirement of the version it chose; there is no default: a local build passes it, or use `kaosctl step run` / `kaosctl dev`, which take your platform's choice.
ARG KAOS_LIB_PACKAGE=""
RUN test -n "${KAOS_LIB_PACKAGE}" || { echo "pass --build-arg KAOS_LIB_PACKAGE, or use kaosctl step run / kaosctl dev, which take your platform's choice" >&2; exit 1; }
RUN pip install --no-cache-dir "${KAOS_LIB_PACKAGE}"

COPY app/ ./app/

ENTRYPOINT ["python", "-m", "app.entry"]
