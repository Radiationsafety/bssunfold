import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _run(cmd, timeout=600):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=timeout
    )


def test_bandit_no_high_severity():
    proc = _run(
        [sys.executable, "-m", "bandit", "-r", "src/bssunfold", "-f", "json", "-q"]
    )
    try:
        results = json.loads(proc.stdout or "{}").get("results", [])
    except json.JSONDecodeError:
        pytest.fail(f"bandit produced no JSON output:\n{proc.stdout}{proc.stderr}")
    high = [r for r in results if r.get("severity") == "HIGH"]
    assert not high, "\n".join(
        f"{r['filename']}:{r.get('line_number')} {r['test_id']} {r['issue_text']}"
        for r in high
    )
