# prepare-data step image. The platform builds it on every push and runs `python -m app.entry` (no arguments):
# the same module `kaosctl step run prepare-data` runs on your machine.
FROM python:3.11-slim

WORKDIR /app

# kaos-lib, the platform library (github.com/kaos-io/kaos-lib). The platform's CI passes KAOS_LIB_PACKAGE, the
# WHOLE pip requirement of the version it chose; the default is what a local build installs (kaos-lib 0.5.0).
ARG KAOS_LIB_PACKAGE="kaos-lib[zitadel] @ https://github.com/kaos-io/kaos-lib/archive/94115aed01aff9a7c702a7510ae6cbda804e875c.zip"
RUN pip install --no-cache-dir "${KAOS_LIB_PACKAGE}"

COPY app/ ./app/

ENTRYPOINT ["python", "-m", "app.entry"]
