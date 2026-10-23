"""Shell scripts parse."""
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("f", ["setup.sh", "install.sh", "deploy.sh", "webapp/run.sh", "webapp/make_cert.sh",
                               "webapp/models_dir.sh", "compile/compile.sh", "compile/build_stages.sh"])
def test_bash_syntax(f):
    subprocess.run(["bash", "-n", str(ROOT / f)], check=True)
