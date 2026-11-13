"""The project's MLflow accepts an experiment or a registered model named `<app>` or `<app>.<x>` only (PRD kaos#1911).
The settings hold the `<x>`; the steps put the app's name in front, from the step's context, never from the code."""
import re

import pytest
from conftest import ROOT, import_in

STEPS = sorted(p.parent.parent.name for p in (ROOT / "steps").glob("*/app/names.py"))


@pytest.fixture(scope="module")
def names():
    return import_in("config_validation", "app.names")


def test_the_experiment_is_the_app_and_the_setting(names):
    assert names.experiment("spacecraft", "baseline") == "spacecraft.baseline"
    assert names.experiment("spacecraft", "run_2-x") == "spacecraft.run_2-x"


def test_an_empty_experiment_setting_gives_the_app_name_alone(names):
    """The platform hands the app's name on as the experiment when the setting is empty."""
    assert names.experiment("spacecraft", "spacecraft") == "spacecraft"


def test_the_registered_model_is_the_app_and_the_setting(names):
    assert names.registered_model("spacecraft", "pose") == "spacecraft.pose"
    assert names.registered_model("spacecraft", "") == "spacecraft"
    assert names.registered_model("spacecraft", None) == "spacecraft"
    assert names.registered_model("spacecraft", " pose ") == "spacecraft.pose"


@pytest.mark.parametrize("bad", ["a b", "a.b", "a/b", "../a", "-a", "_a", "é", "a" * 64, "a:b", "a;b", "a*"])
def test_a_setting_with_characters_the_rule_refuses_is_refused_with_the_setting_named(names, bad):
    with pytest.raises(ValueError, match=r"experiment\.name is .*letters, digits, hyphens and underscores"):
        names.experiment("spacecraft", bad)
    with pytest.raises(ValueError, match=r"registration\.registered_model_name is .*Change it on the run form"):
        names.registered_model("spacecraft", bad)


def test_every_step_that_names_something_has_the_same_file():
    assert STEPS == ["config_validation", "model_quantization", "model_registration", "model_training", "qat_finetune"]
    texts = {(ROOT / "steps" / s / "app" / "names.py").read_text() for s in STEPS}
    assert len(texts) == 1, "steps/*/app/names.py differ: change one and copy it to the others"


def test_the_defaults_of_the_form_hold_only_the_end_of_a_name(names):
    from conftest import composed_settings
    s = composed_settings()
    assert names.experiment("any-app", s["experiment"]["name"]) == f"any-app.{s['experiment']['name']}"
    assert names.registered_model("any-app", s["registration"]["registered_model_name"]).startswith("any-app.")


def test_no_step_hands_the_bare_experiment_to_mlflow_and_no_name_is_written_in_the_code():
    for step in STEPS:
        entry = (ROOT / "steps" / step / "app" / "entry.py").read_text()
        assert "experiment_name=ctx.experiment" not in entry, step
    for path in (ROOT / "steps").rglob("*.py"):
        text = path.read_text()
        assert "object-pose" not in text, path
        for call in re.findall(r"(?:set_experiment|create_experiment|register_model|create_registered_model)\(([^)]*)\)", text):
            assert not re.match(r"\s*['\"]", call), f"{path}: a name written in the code: {call}"


def test_config_validation_refuses_a_bad_name_before_any_machine_is_paid_for(tmp_path):
    from kaos_lib import testing
    entry = import_in("config_validation", "app.entry")
    from conftest import composed_settings
    settings = composed_settings(experiment={"name": "my experiment", "description": ""})
    with testing.running(tmp_path, kind="control", app="spacecraft", experiment="my experiment", params=settings) as run:
        with pytest.raises(Exception, match=r"experiment\.name is 'my experiment'"):
            entry.main()
    assert run.result is None


def test_config_validation_passes_the_form_defaults(tmp_path):
    from conftest import composed_settings
    from kaos_lib import testing
    entry = import_in("config_validation", "app.entry")
    settings = composed_settings()
    with testing.running(tmp_path, kind="control", app="spacecraft", experiment=settings["experiment"]["name"],
                         params=settings) as run:
        entry.main()
    assert run.result["values"] == {"checked": True}


# ---------------------------------------------------------------- MLflow's own rules (its validators, not copies)

def test_every_name_the_template_makes_passes_mlflows_own_validators(names):
    """An app name with hyphens and digits, as real apps have (yolo-777c0431). MLflow allows a dot in an experiment's and
    a registered model's name, not in a logged model's."""
    from mlflow.utils.validation import _validate_experiment_name, _validate_logged_model_name, _validate_model_name
    from conftest import composed_settings
    s = composed_settings()
    for app in ("yolo-777c0431", "a1", "x" * 20):
        for setting in (s["experiment"]["name"], "e2e-777c0431-gcp", app, "run_2"):
            experiment = names.experiment(app, setting)
            _validate_experiment_name(experiment)
            _validate_logged_model_name(names.logged_model(experiment))
            assert experiment == app or experiment.startswith(app + ".")
        for setting in (s["registration"]["registered_model_name"], "", "v2"):
            _validate_model_name(names.registered_model(app, setting))
    assert "." not in names.logged_model("yolo-777c0431.e2e-777c0431-gcp")


def test_the_logged_model_name_is_the_one_registration_creates(tmp_path):
    """Through the real registration step, with MLflow's validator applied to the name it passes to MLflow."""
    import importlib
    from mlflow.utils.validation import _validate_logged_model_name
    from kaos_lib import testing
    from conftest import composed_settings, step_app
    import test_model_registration as reg
    with step_app("model_registration"):
        entry = importlib.import_module("app.entry")
        entry.tracking.mlflow = lambda *a, **k: None
        svc = importlib.import_module("app.services.model_registration")
        seen = []
        import mlflow
        fake = reg.FakeMlflow()
        orig = fake.create_external_model
        fake.create_external_model = lambda name, **k: (_validate_logged_model_name(name), seen.append(name), orig(name, **k))[2]
        mp = pytest.MonkeyPatch()
        mp.setattr(mlflow, "create_external_model", fake.create_external_model, raising=False)
        mp.setattr(mlflow, "register_model", fake.register_model, raising=False)
        mp.setattr(svc, "MlflowClient", lambda: fake)
        from kaos_lib.lakefs import LakeFS
        mp.setattr(LakeFS, "connect", classmethod(lambda cls, *a, **k: reg.FakeLake()))
        try:
            reg.run(entry, tmp_path)
        finally:
            mp.undo()
    assert seen == ["yolo-exp-best"]
