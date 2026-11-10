r"""Reading and writing runs: one folder per run, runs/<model>/<name>/, with

    samples.npz    samples (posterior samples, n x D); for NPE and TSNPE also q (draws of q(θ | x_obs))
    summary.json   method, cost, diagnostics
    x_obs.npz      the observation the run conditioned on (x, and θ if known)

Files are written to a temporary file first, then renamed: a crash never leaves a half-written file.
"""

import argparse
import json
import os

import numpy as np

from .model import MODELS, Model


def parser(doc: str, name: str | None = None) -> argparse.ArgumentParser:
    """The options shared by the scripts: --model, and for the inference scripts (name = the default run name)
    --obs, --seed, --out and --name."""
    p = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="exo6", choices=MODELS, help="forward model (default: exo6)")
    if name is not None:
        p.add_argument("--obs", default="0", help="observation: index of a test pair (0-7, 0 = the benchmark) or "
                                                  "an npz file with x (and θ, if known); default: 0")
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--out", default="runs", help="results go to OUT/<model>/<name>/ (default: runs)")
        p.add_argument("--name", default=name, help=f"run name (default: {name})")
    return p


def setup(args) -> tuple[Model, np.ndarray, str]:
    """The model, the observation and the run folder (created, with a copy of the observation)."""
    model = Model(args.model)
    if args.obs.isdigit():
        theta, x = (a[int(args.obs)] for a in model.test_pairs())
    else:
        d = np.load(args.obs)
        x, theta = d["x"], d["theta"] if "theta" in d else None
    path = os.path.join(args.out, args.model, args.name)
    os.makedirs(path, exist_ok=True)
    write_npz(os.path.join(path, "x_obs.npz"), x=x, **({} if theta is None else {"theta": theta}))
    print(f"{model}, run folder {path}")
    return model, x, path


def save(path: str, samples: np.ndarray, q: np.ndarray | None = None, **summary):
    write_npz(os.path.join(path, "samples.npz"), samples=samples, **({} if q is None else {"q": q}))
    write_json(os.path.join(path, "summary.json"), summary)


def load(path: str) -> dict:
    """{samples, q (or None), x, theta (or None), summary}."""
    d, o = np.load(os.path.join(path, "samples.npz")), np.load(os.path.join(path, "x_obs.npz"))
    with open(os.path.join(path, "summary.json")) as f:
        summary = json.load(f)
    return {"samples": d["samples"], "q": d["q"] if "q" in d else None, "x": o["x"],
            "theta": o["theta"] if "theta" in o else None, "summary": summary}


def write_npz(path: str, **arrays):
    _write(path, lambda f: np.savez(f, **arrays))


def write_json(path: str, obj):
    _write(path, lambda f: f.write(json.dumps(obj, indent=1).encode()))


def _write(path: str, write):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        write(f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
