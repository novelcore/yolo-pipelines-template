"""Every step gets everything from kaos_lib.step: no argument and no variable of its own."""
import ast

import pytest
from conftest import ROOT

ENTRIES = sorted((ROOT / "steps").glob("*/app/entry.py"))
NOT_IN_A_STEP = ("os.environ", "getenv", "argparse", "sys.argv")  # a step takes no arguments and reads no variable


@pytest.mark.parametrize("entry", ENTRIES, ids=lambda p: p.parts[-3])
def test_an_entry_starts_the_step_and_ends_with_a_result(entry):
    text = entry.read_text()
    assert "step.start()" in text and "step.result(" in text
    assert not [word for word in NOT_IN_A_STEP if word in text]


@pytest.mark.parametrize("entry", ENTRIES, ids=lambda p: p.parts[-3])
def test_an_entry_starts_with_start(entry):
    tree = ast.parse(entry.read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    assert ast.unparse(main.body[0]).startswith("ctx = step.start()"), "start() comes first"
