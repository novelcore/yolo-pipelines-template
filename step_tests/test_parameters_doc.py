"""docs/parameters.md lists every field of the run form. It is generated: after a change to config/ or to the
bounds a step checks, run `python scripts/gen-parameters.py` and commit the result."""
import subprocess
import sys

from conftest import ROOT


def test_the_page_of_run_form_fields_is_up_to_date():
    done = subprocess.run([sys.executable, str(ROOT / "scripts" / "gen-parameters.py"), "--check"],
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stdout + done.stderr
