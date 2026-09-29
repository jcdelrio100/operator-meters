"""Cached Burgers trajectories for the nonlinearity sweep (WP2, WP2b, WP3, WP4, WP4b).

For every eps in EPS we integrate NTR train + NTE test random smooth initial conditions
(amplitude 1) and, for the out-of-distribution test of WP4, NTE initial conditions of
amplitude AMP_OOD.  Snapshots are taken every SUB RK4 steps (dt = 2e-3), i.e. the learned
one-step map spans a horizon DELTA = SUB * dt = 0.17, long enough for the nonlinearity to act.

Cache: results/cache/burgers_eps{eps}.npz  (regenerated deterministically when absent).
"""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from common import integrate, random_ic, N

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(os.path.dirname(HERE), "results", "cache")
os.makedirs(CACHE, exist_ok=True)

EPS = [0.0, 0.1, 0.2, 0.4, 0.7, 1.0, 1.5, 2.0]
DT = 2e-3
SUB = 85
DELTA = SUB * DT
NSNAP = 32                      # snapshots per trajectory (incl. u0) -> 31 one-step pairs
NTR, NTE = 60, 20
AMP_OOD = 1.6


def _gen(eps, seed, n, amp):
    rng = np.random.default_rng(seed)
    T = (NSNAP - 1) * DELTA
    return np.array([integrate(random_ic(rng, amp=amp), eps, T, dt=DT, sub=SUB) for _ in range(n)])


def load(eps):
    f = os.path.join(CACHE, f"burgers_eps{eps:g}.npz")
    if os.path.exists(f):
        z = np.load(f)
        return z["tr"], z["te"], z["ood"]
    t0 = time.time()
    tr = _gen(eps, 1000, NTR, 1.0)
    te = _gen(eps, 2000, NTE, 1.0)
    ood = _gen(eps, 3000, NTE, AMP_OOD)
    np.savez_compressed(f, tr=tr, te=te, ood=ood)
    print(f"  generated eps={eps:g}: {tr.shape} train, {te.shape} test, {ood.shape} ood "
          f"({time.time()-t0:.0f}s)")
    return tr, te, ood


def pairs(traj):
    """(X, Y) one-step pairs from an array of trajectories [n_traj, NSNAP, N]."""
    X = traj[:, :-1].reshape(-1, N)
    Y = traj[:, 1:].reshape(-1, N)
    return X, Y


if __name__ == "__main__":
    for e in EPS:
        load(e)
    print("cache complete")
