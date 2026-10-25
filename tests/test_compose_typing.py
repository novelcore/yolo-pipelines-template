"""compose-and-validate keeps every form value in the type its config leaf
declares (Hydra numeric/bool/null coercion bug: a bare `label=3e286280` parsed
as 3x10^286280 = inf and params.yaml said `label: .inf`).

Runs against this template's real config tree (config/ + the vendored
platform group). Run:  python -m pytest tests/ -v
"""

import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kubecore import compose  # noqa: E402

STRING_LOOKALIKES = ["3e286280", "1234", "1e5", "true", "null", "nan", "inf", "0x1f",
                     "-7", "False", "~", "1_000"]


def resolved(tokens, render_defaults=None):
    """Full compose path; returns params.yaml text and its parsed form (the
    downstream steps' view)."""
    text = compose.compose_and_validate(tokens, render_defaults=render_defaults or {})
    return text, yaml.safe_load(text)


# ------------------------------------------------------- string leaves


@pytest.mark.parametrize("value", STRING_LOOKALIKES)
def test_untyped_string_leaf_stays_string(value):
    _, out = resolved([f"summary.label={value}"])
    assert out["summary"]["label"] == value
    assert isinstance(out["summary"]["label"], str)


@pytest.mark.parametrize("value", STRING_LOOKALIKES)
def test_schema_str_leaf_stays_string(value):
    # data.ref is `str` in the DataConfig structured schema
    _, out = resolved([f"data.ref={value}"])
    assert out["data"]["ref"] == value


def test_reported_case_3e286280_is_not_inf():
    text, out = resolved(["experiment.name=3e286280"])
    assert ".inf" not in text
    assert out["experiment"]["name"] == "3e286280"


@pytest.mark.parametrize("value", [
    "my run", 'say "hi"', "it's", "a=b", "k=v=w", "📊 Data summary",
    "C:\\data\\x", "trail\\", 'back\\"quote', "a,b", "[1,2]", "{a: 1}", "$x",
])
def test_string_special_characters_round_trip(value):
    _, out = resolved([f"experiment.description={value}"])
    assert out["experiment"]["description"] == value


def test_empty_string_default_leaf_takes_text():
    # train.freeze defaults to "" (an untyped string); model_training int()s it
    _, out = resolved(["train.freeze=10"])
    assert out["train"]["freeze"] == "10"


def test_enum_leaf_accepts_its_value():
    _, out = resolved(["quantization.mode=ptq"])
    assert out["quantization"]["mode"] == "ptq"


# ------------------------------------------------- numeric / bool leaves


def test_int_param_stays_int():
    _, out = resolved(["train.epochs=10"])
    assert out["train"]["epochs"] == 10 and type(out["train"]["epochs"]) is int


def test_float_param_stays_float():
    _, out = resolved(["train.lrf=0.5"])
    assert out["train"]["lrf"] == 0.5 and type(out["train"]["lrf"]) is float


def test_float_param_given_integer_text_stays_float():
    _, out = resolved(["train.warmup_epochs=5", "image_processing.mosaic=1"])
    assert type(out["train"]["warmup_epochs"]) is float and out["train"]["warmup_epochs"] == 5.0
    assert type(out["image_processing"]["mosaic"]) is float


def test_bool_param_stays_bool():
    _, out = resolved(["train.cos_lr=False", "train.export.enabled=TRUE"])
    assert out["train"]["cos_lr"] is False
    assert out["train"]["export"]["enabled"] is True


def test_schema_int_leaf():
    _, out = resolved(["data.seed=7"])
    assert out["data"]["seed"] == 7


@pytest.mark.parametrize("token", ["train.epochs=ten", "train.epochs=1.5", "train.lrf=fast",
                                   "train.amp=yes", "data.seed=0x1f"])
def test_invalid_literal_for_typed_leaf_fails_compose(token):
    with pytest.raises(compose.ComposeError):
        resolved([token])


# --------------------------------------------- groups, quoting, elision


def test_leaf_type_follows_the_selected_group():
    _, out = resolved(["train/callbacks=none", "train.callbacks.patience=5",
                       "train.optimizer.lr=1"])
    assert out["train"]["callbacks"]["patience"] == 5
    assert type(out["train"]["optimizer"]["lr"]) is float


def test_top_level_group_swap_is_a_group():
    tokens = compose.order_overrides(
        ["model.variant=3e286280", "model=yolov8s"], {"model": "yolov8n"})
    assert tokens == ["model=yolov8s", 'model.variant="3e286280"']
    _, out = resolved(["model=yolov8s", "model.variant=3e286280"])
    assert out["model"]["variant"] == "3e286280"


def test_already_quoted_values_are_kept():
    assert compose.order_overrides(['summary.label="3e286280"'], {}) == ['summary.label="3e286280"']
    assert compose.order_overrides(["train.epochs='10'"], {}) == ["train.epochs='10'"]
    _, out = resolved(["summary.label='3e286280'"])
    assert out["summary"]["label"] == "3e286280"


def test_untouched_leaf_elided():
    tokens = compose.order_overrides(
        ["summary.label=📊 Data summary", "train.epochs=10\r"],
        {"summary.label": "📊 Data summary", "train.epochs": "100"},
    )
    assert tokens == ["train.epochs=10"]


def test_emitted_tokens():
    tokens = compose.order_overrides(
        ["summary.label=3e286280", "train.epochs=10", "train.lrf=1", "train.amp=TRUE",
         'experiment.description=a "b" \\c\\'], {})
    assert tokens == ['summary.label="3e286280"', "train.epochs=10", "train.lrf=1.0",
                      "train.amp=true", 'experiment.description="a \\"b\\" \\c\\\\"']


def test_unknown_key_still_rejected():
    with pytest.raises(Exception):
        resolved(["train.epoch=10"])
