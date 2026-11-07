"""The kaos-lib version the step images run (PRD novelcore/kaos#1147 package delivery).

The platform picks the version: CI passes --build-arg KAOS_LIB_PACKAGE=<spec> to every step
image that declares the ARG, only inside the range in .kaos/compat.yaml. Each step logs
`kaos-lib <version>` first; training tags its MLflow run and registration its model versions
with `kaos_lib.version`.
"""
import ast
import re
from importlib.metadata import version

import pytest
import yaml
from conftest import ROOT, import_in
from test_models_linked import LINKED, FakeMlflow, _install, _merged_commit, _register

PACKAGE_STEPS = ["config_validation", "dataset_loading", "model_quantization", "model_registration",
                 "model_training", "qat_finetune"]
NO_PACKAGE = ["compose_and_validate", "data_summary"]  # the vendored engine / PyYAML only
# The ARG holds the whole requirement: name[extras] then ==X.Y.Z or " @ <url>".
SPEC_RE = re.compile(r'^ARG KAOS_LIB_PACKAGE="([A-Za-z0-9._-]+(?:\[[A-Za-z0-9,._-]+\])?(?: ?==\S+| @ https://\S+))"$', re.M)
INSTALLED = version("kaos-lib")  # the installed package, as the step images pin it


@pytest.mark.parametrize("step", PACKAGE_STEPS)
def test_every_package_step_takes_the_platform_spec(step):
    text = (ROOT / "steps" / step / "Dockerfile").read_text()
    assert SPEC_RE.search(text), "ARG KAOS_LIB_PACKAGE with a ==X.Y.Z or @ <url> default"
    assert 'RUN pip install --no-cache-dir "${KAOS_LIB_PACKAGE}"' in text
    assert SPEC_RE.search(text).group(1).startswith("kaos-lib[zitadel] @ "), "the default names the package and its extras"
    assert "KUBECORE_PKG_COMMIT" not in text


@pytest.mark.parametrize("step", NO_PACKAGE)
def test_steps_without_the_package_take_no_spec(step):
    assert "KAOS_LIB_PACKAGE" not in (ROOT / "steps" / step / "Dockerfile").read_text()


def test_every_step_installs_the_same_default():
    specs = {SPEC_RE.search((ROOT / "steps" / s / "Dockerfile").read_text()).group(1) for s in PACKAGE_STEPS}
    assert len(specs) == 1, specs


def test_quantization_installs_the_package_after_the_export_toolchain():
    """kaniko keys every layer after the ARG on its value: a new version must not re-download TensorFlow."""
    text = (ROOT / "steps" / "model_quantization" / "Dockerfile").read_text()
    assert text.index("ARG KAOS_LIB_PACKAGE") > text.index('"tensorflow==')


def test_the_template_declares_the_range_of_its_package():
    compat = yaml.safe_load((ROOT / ".kaos" / "compat.yaml").read_text())
    assert compat == {"kaos-lib": ">=0.4.0rc15,<0.5"}
    assert INSTALLED.startswith("0.4."), "the package the steps are tested with is inside the range"


@pytest.mark.parametrize("step", PACKAGE_STEPS)
def test_every_package_step_logs_the_kaos_lib_version_first(step):
    tree = ast.parse((ROOT / "steps" / step / "app" / "entry.py").read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    assert ast.unparse(main.body[0]) == "print(f'kaos-lib {kaos_lib_version()}', flush=True)"


def test_the_version_follows_the_distribution_after_the_rename(monkeypatch):
    mod = import_in("model_training", "app.entry")

    def renamed(dist):
        if dist == "kaos-lib":
            return "0.3.0"
        raise mod.PackageNotFoundError(dist)

    assert mod.kaos_lib_version() == INSTALLED
    monkeypatch.setattr(mod, "package_version", renamed)
    assert mod.kaos_lib_version() == "0.3.0"


def test_training_tags_its_mlflow_run_with_the_kaos_lib_version(monkeypatch):
    mlf = FakeMlflow()
    _install(monkeypatch, mlf)
    mod = import_in("model_training", "app.services.model_training")
    mod.TrainingService._tag_kubecore_metadata("mlflow-1")
    assert mlf.run_tags["mlflow-1"]["kaos_lib.version"] == INSTALLED


def test_registration_tags_the_model_versions_with_the_kaos_lib_version(fake, connected, monkeypatch):
    mlf = FakeMlflow({**LINKED, "kaos.outputs_commit": _merged_commit(fake)})
    _register(monkeypatch, mlf)
    assert mlf.version_tags[("yolo", 1)]["kaos_lib.version"] == INSTALLED
    assert mlf.version_tags[("yolo", 2)]["kaos_lib.version"] == INSTALLED
