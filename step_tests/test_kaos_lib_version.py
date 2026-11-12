"""The kaos-lib version the step images run (PRD novelcore/kaos#1147 package delivery, #1911 the step contract).

The platform picks the version: CI passes --build-arg KAOS_LIB_PACKAGE=<spec> to every step image that declares the
ARG, only inside the range in .kaos/compat.yaml.
"""
import re
from importlib.metadata import version

import pytest
import yaml
from conftest import ROOT

STEPS = sorted(p.parent.name for p in (ROOT / "steps").glob("*/Dockerfile"))
# The ARG holds the whole requirement: name[extras] then ==X.Y.Z or " @ <url>".
SPEC_RE = re.compile(r'^ARG KAOS_LIB_PACKAGE="([A-Za-z0-9._-]+(?:\[[A-Za-z0-9,._-]+\])?(?: ?==\S+| @ https://\S+))"$', re.M)


def test_every_step_has_an_image():
    assert STEPS == ["config_validation", "model_quantization", "model_registration", "model_training",
                     "prepare_data", "qat_finetune"]


@pytest.mark.parametrize("step", STEPS)
def test_every_step_takes_the_platform_spec(step):
    text = (ROOT / "steps" / step / "Dockerfile").read_text()
    assert SPEC_RE.search(text), "ARG KAOS_LIB_PACKAGE with a ==X.Y.Z or @ <url> default"
    assert 'RUN pip install --no-cache-dir "${KAOS_LIB_PACKAGE}"' in text
    assert SPEC_RE.search(text).group(1).startswith("kaos-lib[zitadel] @ "), "the default names the package and its extras"
    assert "--params" not in text, "a step image runs `python -m app.entry` with no arguments"


def test_every_step_installs_the_same_default():
    specs = {SPEC_RE.search((ROOT / "steps" / s / "Dockerfile").read_text()).group(1) for s in STEPS}
    assert len(specs) == 1, specs


def test_quantization_installs_the_package_after_the_export_toolchain():
    """kaniko keys every layer after the ARG on its value: a new version must not re-download TensorFlow."""
    text = (ROOT / "steps" / "model_quantization" / "Dockerfile").read_text()
    assert text.index("ARG KAOS_LIB_PACKAGE") > text.index('"tensorflow==')


def test_the_template_declares_the_range_of_its_package():
    compat = yaml.safe_load((ROOT / ".kaos" / "compat.yaml").read_text())
    assert compat == {"kaos-lib": ">=0.5.0rc1,<0.6"}
    assert version("kaos-lib").startswith("0.5."), "the package the steps are tested with is inside the range"
