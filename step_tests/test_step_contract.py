"""Every step gets everything from kaos_lib.step: no argument, no variable, no platform plumbing of its own."""
import ast

import pytest
from conftest import ROOT

ENTRIES = sorted((ROOT / "steps").glob("*/app/entry.py"))
NOT_IN_A_STEP = ("os.environ", "getenv", "argparse", "sys.argv")  # a step takes no arguments and reads no variable
# What the platform and kaos-lib do now. A step that names one of these has its own copy of it again.
GONE = ("KUBECORE_", "CHECKPOINT_BUCKET", "CHECKPOINT_PREFIX", "KAOS_LIB_LOCK", "lakefs_io", "resource_monitor",
        "lock.write", "lock.record", "--params", "training-result", "outputs.run(", "write_run(")


def test_every_step_of_the_pipeline_has_an_entry():
    assert [p.parts[-3] for p in ENTRIES] == ["config_validation", "model_quantization", "model_registration",
                                              "model_training", "prepare_data", "qat_finetune"]


@pytest.mark.parametrize("entry", ENTRIES, ids=lambda p: p.parts[-3])
def test_an_entry_starts_the_step_and_ends_with_a_result(entry):
    text = entry.read_text()
    assert "step.start()" in text and "step.result(" in text
    assert not [word for word in NOT_IN_A_STEP if word in text]


@pytest.mark.parametrize("entry", ENTRIES, ids=lambda p: p.parts[-3])
def test_an_entry_has_no_platform_plumbing(entry):
    tree = ast.parse(entry.read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    assert ast.unparse(main.body[0]).startswith("ctx = step.start()"), "start() comes first"
    assert not [word for word in GONE if word in entry.read_text()]


def test_no_step_code_reads_a_platform_variable_or_keeps_a_copy_of_what_kaos_lib_does():
    code = "\n".join(p.read_text() for p in (ROOT / "steps").glob("*/app/**/*.py"))
    assert not [word for word in GONE if word in code and word not in ("--params", "training-result")]


def test_the_helpers_every_step_used_to_copy_are_gone():
    for name in ("lakefs_io.py", "resource_monitor.py"):
        assert not list((ROOT / "steps").glob(f"*/app/services/{name}")), name
