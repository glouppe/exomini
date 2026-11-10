r"""Nested sampling (dynesty) with the exact likelihood: the classical reference, and the evidence log Z.

One likelihood evaluation is one simulation, done in CPU workers by random slice sampling ("rslice"); dynesty's
default for few parameters (ellipsoids) keeps the main process busy and the workers idle instead. Not ground
truth either: on exo30, runs with 1000 and 2000 live points disagree by 1.0 in log Z. Checkpoints to
dynesty.save, and resumes from it.

    python nested.py --model exo6
    python nested.py --model exo30 --nlive 2000
"""

import os
import sys
import time
from multiprocessing import Pool

import dynesty
import numpy as np
import torch

from toysbi import Model, runs

_model, _x = None, None


def _init(name, x):  # in each worker
    global _model, _x
    torch.set_num_threads(1)
    _model, _x = Model(name, device="cpu"), x


def _log_likelihood(theta):
    return float(_model.log_likelihood(theta[None], _x)[0])


class _PriorTransform:  # unit cube -> θ; a class, so that it can be sent to the workers
    def __init__(self, lower, upper):
        self.lower, self.upper = lower, upper

    def __call__(self, c):
        return self.lower + c * (self.upper - self.lower)


def main():
    p = runs.parser(__doc__, "nested")
    p.add_argument("--nlive", type=int, default=500, help="live points (default: 500)")
    p.add_argument("--dlogz", type=float, default=0.5, help="stop when the remaining log Z is below (default: 0.5)")
    p.add_argument("--workers", type=int, help="CPU processes (default: all available)")
    args = p.parse_args()
    model, x_obs, path = runs.setup(args)
    workers = args.workers or (len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count())
    t0 = time.time()

    checkpoint = f"{path}/dynesty.save"
    with Pool(workers, initializer=_init, initargs=(model.name, x_obs)) as pool:
        if os.path.exists(checkpoint):
            sampler = dynesty.NestedSampler.restore(checkpoint, pool=pool)
            sampler.run_nested(resume=True, checkpoint_file=checkpoint, checkpoint_every=600,
                               print_progress=sys.stdout.isatty())
        else:
            sampler = dynesty.NestedSampler(_log_likelihood, _PriorTransform(model.lower, model.upper), model.D,
                                            nlive=args.nlive, sample="rslice", pool=pool, queue_size=workers,
                                            rstate=np.random.default_rng(args.seed))
            sampler.run_nested(dlogz=args.dlogz, checkpoint_file=checkpoint, checkpoint_every=600,
                               print_progress=sys.stdout.isatty())
    res = sampler.results
    calls = int(np.sum(res.ncall))
    print(f"log Z {res.logz[-1]:.2f} ± {res.logzerr[-1]:.2f}, {calls} likelihood calls")
    runs.save(path, res.samples_equal(), method=f"nested sampling, {args.nlive} live points", n_sims=calls,
              logz=float(res.logz[-1]), logz_err=float(res.logzerr[-1]), seconds=time.time() - t0)
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
