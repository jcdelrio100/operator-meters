"""WP3 — reading the operator (sparse identification, own STLSQ, no external dependency).

For each eps: integrate a few fresh smooth initial conditions with the fine RK4 step, estimate
u_t by a forward difference of consecutive snapshots (NOT the analytic right-hand side), build
the library  [u, u_x, u_xx, u_xxx, u^2, u u_x, u u_xx, (u_x)^2]  with spectral derivatives and
run sequential thresholded least squares.  Truth: u_t = nu u_xx - eps u u_x, nu = 0.05.

Outputs: figures/phys_wp3_sindy.png, results/phys_wp3.json
"""
import os, sys, json
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from common import integrate, random_ic, spectral_deriv, NU, N
from data import EPS

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
FIG = os.path.join(ROOT, "figures"); RES = os.path.join(ROOT, "results")
NAMES = ["u", "u_x", "u_xx", "u_xxx", "u^2", "u u_x", "u u_xx", "(u_x)^2"]
DT = 2e-3


def library(u):
    ux, uxx, uxxx = spectral_deriv(u, 1), spectral_deriv(u, 2), spectral_deriv(u, 3)
    return np.stack([u, ux, uxx, uxxx, u ** 2, u * ux, u * uxx, ux ** 2], -1)


def stlsq(Theta, dudt, thresh=0.02, iters=10):
    xi = np.linalg.lstsq(Theta, dudt, rcond=None)[0]
    for _ in range(iters):
        small = np.abs(xi) < thresh
        xi[small] = 0
        big = ~small
        if big.sum() == 0:
            break
        xi[big] = np.linalg.lstsq(Theta[:, big], dudt, rcond=None)[0]
    return xi


def identify(eps, n_ic=6, n_steps=400, seed=0):
    rng = np.random.default_rng(seed + 7)
    Th, Ut = [], []
    for _ in range(n_ic):
        traj = integrate(random_ic(rng), eps, n_steps * DT, dt=DT, sub=1)      # fine snapshots
        for u0, u1 in zip(traj[:-1:4], traj[1::4]):
            Th.append(library(u0)); Ut.append((u1 - u0) / DT)                 # forward difference
    Theta = np.concatenate(Th, 0); dudt = np.concatenate(Ut, 0)
    # scale columns for a threshold that means the same for every term
    s = np.linalg.norm(Theta, axis=0) / np.sqrt(len(Theta)); s[s == 0] = 1
    xi = stlsq(Theta / s, dudt, thresh=0.02) / s
    return xi


def run():
    rows = []
    for eps in [e for e in EPS if e > 0]:
        xi = identify(eps)
        rows.append(dict(eps=eps, xi=xi.tolist(), nu_hat=float(xi[2]), eps_hat=float(-xi[5]),
                         others_max=float(np.abs(np.delete(xi, [2, 5])).max())))
        print(f"eps={eps:3.1f}  nu_hat={xi[2]:.4f} (0.0500)  coef(u u_x)={xi[5]:+.4f} ({-eps:+.3f})  "
              f"max|other|={rows[-1]['others_max']:.2e}")
    return rows


def plot(rows):
    e = np.array([r["eps"] for r in rows])
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot(e, -e, "k--", lw=1, label=r"identity $-\varepsilon$")
    ax[0].plot(e, [-r["eps_hat"] for r in rows], "o", color="#B0173A", label=r"recovered coefficient of $u\,u_x$")
    ax[0].set_xlabel(r"$\varepsilon$"); ax[0].set_ylabel("coefficient"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[0].set_title("the nonlinear term is read exactly")
    ax[1].axhline(NU, color="k", ls="--", lw=1, label=r"true $\nu$")
    ax[1].plot(e, [r["nu_hat"] for r in rows], "s", color="#31557F", label=r"recovered $\nu$")
    ax[1].set_ylim(0, 0.08); ax[1].set_xlabel(r"$\varepsilon$"); ax[1].legend(); ax[1].grid(alpha=.3)
    ax[1].set_title("the diffusion coefficient stays pinned")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "phys_wp3_sindy.png"), dpi=160); plt.close(fig)


if __name__ == "__main__":
    rows = run(); plot(rows)
    json.dump(dict(rows=rows, names=NAMES, nu=NU), open(os.path.join(RES, "phys_wp3.json"), "w"), indent=1)
    print("done")
