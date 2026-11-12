# config-validation step image (the pipeline's first step).
# The platform builds this on push and wires the resulting image into the pipeline. The in-image
# command is `python -m app.entry` (no arguments): the same module `kaosctl step run config-validation` runs.
FROM python:3.11-slim

WORKDIR /app

# kaos-lib, the platform library (github.com/kaos-io/kaos-lib). The platform's CI passes KAOS_LIB_PACKAGE, the
# WHOLE pip requirement of the version it chose; the default is what a local build installs (kaos-lib 0.5.0).
ARG KAOS_LIB_PACKAGE="kaos-lib[zitadel] @ https://github.com/kaos-io/kaos-lib/archive/f422375c291f99bce326de3c606b957ec6cb228d.zip"
RUN pip install --no-cache-dir "${KAOS_LIB_PACKAGE}"

COPY app/ ./app/

ENTRYPOINT ["python", "-m", "app.entry"]
