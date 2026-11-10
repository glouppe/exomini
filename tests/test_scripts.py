import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]


def run(*args, cwd):
    subprocess.run([sys.executable, str(ROOT / args[0]), *args[1:]], cwd=cwd, check=True,
                   env={"PYTHONPATH": str(ROOT), "PATH": "/usr/bin:/bin", "CUDA_VISIBLE_DEVICES": ""})


def test_pipeline(tmp_path):
    """Every script end to end on exo6, with tiny budgets."""
    run("simulate.py", "--describe", cwd=tmp_path)
    run("simulate.py", "--n", "10", cwd=tmp_path)
    run("simulate.py", "--theta", "1300", "4", "0.5", "0.8", "0", "1.2", "--out", "obs.npz", cwd=tmp_path)
    run("npe.py", "--sims", "1000", "--epochs", "2", "--n-is", "300", cwd=tmp_path)
    run("tsnpe.py", "--rounds", "2", "--sims", "500", "--epochs", "2", "--n-is", "300", "--K", "8", cwd=tmp_path)
    run("nested.py", "--nlive", "30", "--dlogz", "50", "--workers", "2", cwd=tmp_path)
    run("npe.py", "--sims", "500", "--epochs", "1", "--n-is", "300", "--obs", "obs.npz", "--name", "npe-obs",
        cwd=tmp_path)
    run("npe.py", "--sims", "500", "--epochs", "1", "--n-is", "300", "--obs", "3", "--name", "npe-pair3", cwd=tmp_path)
    run("compare.py", "--runs", "nested", "npe", "tsnpe", cwd=tmp_path)
    out = tmp_path / "runs" / "exo6"
    for f in ["corner.png", "ppc.png", "table.md", "npe/samples.npz", "tsnpe/rounds.json", "nested/summary.json"]:
        assert (out / f).exists(), f
