"""The guide of kaos-lib reads this pipeline on its page "A real pipeline", and shows two of its files from a copy.

Point DOCS_CHECK_KAOS_LIB at a checkout of kaos-lib to compare: the copies must equal this template byte for byte.
Without it the test is skipped.
"""
import os
from pathlib import Path

import pytest
from conftest import ROOT

LIB = os.environ.get("DOCS_CHECK_KAOS_LIB")
pytestmark = pytest.mark.skipif(not LIB, reason="set DOCS_CHECK_KAOS_LIB to a checkout of kaos-lib")
SHOWN = ("pipeline.py", "steps/prepare_data/app/entry.py")


@pytest.mark.parametrize("name", SHOWN)
def test_should_equal_the_copy_the_guide_of_kaos_lib_shows(name):
    copy = Path(LIB) / "docs" / "examples" / "yolo-pipeline" / name
    assert copy.read_bytes() == (ROOT / name).read_bytes(), f"{name} differs from docs/examples/yolo-pipeline of kaos-lib"
