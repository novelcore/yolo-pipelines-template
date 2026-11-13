"""The names this app gives in the project's MLflow.

The project's MLflow accepts an experiment or a registered model only when its name is `<app>` or
`<app>.<something>`: the app's own name first. So the settings hold only the `<something>` and these functions add the
app's name, which the platform tells every step (`step.start().app`). No name is written in this repository.

The same file is in every step that names something in MLflow (steps/*/app/names.py);
step_tests/test_names.py fails when they differ.
"""
import re

# What may follow `<app>.`: letters, digits, hyphens and underscores, starting with a letter or a digit.
_SUFFIX = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,62}$")


def _suffix(setting: str, value: str) -> str:
    value = str(value).strip()
    if not _SUFFIX.match(value):
        raise ValueError(
            f"{setting} is {value!r}. It may hold only letters, digits, hyphens and underscores (at most 63), "
            "starting with a letter or a digit: the project's MLflow accepts names of the form <app> or "
            "<app>.<name> only, and the pipeline adds the app's name in front. Change it on the run form."
        )
    return value


def experiment(app: str, name: str) -> str:
    """The MLflow experiment of a run: `<app>.<name>`, or `<app>` when `experiment.name` is empty (the platform then
    hands the app's name on as the experiment, and so does this)."""
    return app if name == app else f"{app}.{_suffix('experiment.name', name)}"


def registered_model(app: str, name: str) -> str:
    """The registered model of a run: `<app>.<name>`, or `<app>` when `registration.registered_model_name` is empty."""
    name = str(name or "").strip()
    return app if not name else f"{app}.{_suffix('registration.registered_model_name', name)}"
