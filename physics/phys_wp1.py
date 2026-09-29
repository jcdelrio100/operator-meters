"""WP1 + WP1b — heterogeneity meter and oracle-free discovery of the diagonalizing basis.

WP1  : sweep the heterogeneity knob beta of  u_t = d/dx(a du/dx), a = 1 + beta f(x);
       measure the off-diagonal fraction D(B, P) of the one-step propagator in the fixed
       real Fourier basis and in the true eigenbasis (oracle), plus the alignment of the
       eigenbasis with Fourier.
WP1b : remove the oracle.  Search a structured Sturm-Liouville family
       L(theta) = -G^T diag(a_theta) G, a_theta = 1 + truncated Fourier series (2M params),
       minimising ONLY D(Phi_theta, P_true) from theta = 0 (Fourier), Powell, 3 restarts.
       Score the recovered coefficient against the truth (Pearson r after standardizing,
       since eigenvectors are invariant under a -> c a).  Three conditions:
         well-specified  (truth in the family, M = 3)
         mis-specified   (truth has a 5th harmonic, family M = 3)
         re-specified    (same truth, family widened to M = 5)

Outputs: figures/phys_wp1_meter.png, figures/phys_wp1_discovery.png, results/phys_wp1.json
"""
import os, sys, json, time
import numpy as np
from scipy.optimize import minimize
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from common import (N, X, XF, FOURIER, L_of_coeff, propagator, eigenbasis, offdiag_fraction,
                    alignment, field_from_harmonics, truth_field)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIG = os.path.join(ROOT, "figures"); RES = os.path.join(ROOT, "results")
os.makedirs(FIG, exist_ok=True); os.makedirs(RES, exist_ok=True)

DT = 0.05
BETAS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]


def true_propagator(beta, kind="well"):
    f, ff = truth_field(kind)
    a_faces = 1 + beta * ff
    a_cent = 1 + beta * f
    L = L_of_coeff(a_faces)
    return propagator(L, DT), a_cent, L


# ------------------------------------------------------------------ WP1: the meter
def wp1():
    rows = []
    for b in BETAS:
        P, a, L = true_propagator(b)
        Phi, _ = eigenbasis(L)
        rows.append(dict(beta=b,
                         D_fourier=offdiag_fraction(FOURIER, P),
                         D_eigen=offdiag_fraction(Phi, P),
                         align=alignment(Phi)))
        print(f"beta={b:4.1f}  D_Fourier={rows[-1]['D_fourier']:.3f}  "
              f"D_eigen={rows[-1]['D_eigen']:.2e}  align={rows[-1]['align']:.3f}")
    return rows


# ------------------------------------------------------------------ WP1b: discovery
def discover(P, M, restarts=3, seed=0):
    """Minimise D(Phi_theta, P) over theta in R^{2M}; return best theta and objective."""
    rng = np.random.default_rng(seed)

    def obj(theta):
        af = field_from_harmonics(theta, XF)
        if af.min() < 0.05:                       # keep the operator elliptic
            return 1.0 + 10 * (0.05 - af.min())
        Phi, _ = eigenbasis(L_of_coeff(af))
        return offdiag_fraction(Phi, P)

    best = None
    starts = [np.zeros(2 * M)] + [rng.normal(0, 0.05, 2 * M) for _ in range(restarts - 1)]
    for th0 in starts:
        r = minimize(obj, th0, method="Powell",
                     options=dict(xtol=1e-6, ftol=1e-10, maxfev=20000))
        if best is None or r.fun < best.fun:
            best = r
    return best.x, float(best.fun)


def standardized_corr(a, b):
    a = (a - a.mean()) / (a.std() + 1e-12); b = (b - b.mean()) / (b.std() + 1e-12)
    return float(np.mean(a * b))


def wp1b():
    out = {}
    conds = [("well-specified", "well", 3), ("mis-specified", "mis", 3), ("re-specified", "mis", 5)]
    for name, kind, M in conds:
        rows = []
        for b in BETAS:
            if b == 0.0:
                continue
            P, a_true, _ = true_propagator(b, kind)
            D0 = offdiag_fraction(FOURIER, P)
            t0 = time.time()
            theta, Dstar = discover(P, M)
            a_rec = field_from_harmonics(theta, X)
            r = standardized_corr(a_rec, a_true)
            rows.append(dict(beta=b, D_fourier=D0, D_discovered=Dstar, corr=r,
                             a_true=a_true.tolist(), a_rec=a_rec.tolist(), theta=theta.tolist()))
            print(f"[{name:14s} M={M}] beta={b:3.1f}  D: {D0:.3f} -> {Dstar:.6f}   "
                  f"corr(a)={r:.4f}   ({time.time()-t0:.0f}s)")
        out[name] = rows
    return out


# ------------------------------------------------------------------ figures
def plot_meter(rows):
    b = [r["beta"] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot(b, [r["D_fourier"] for r in rows], "o-", color="#31557F", label="fixed Fourier basis")
    ax[0].plot(b, [r["D_eigen"] for r in rows], "s-", color="#2F6B3D", label="true eigenbasis (oracle)")
    ax[0].set_xlabel(r"heterogeneity $\beta$"); ax[0].set_ylabel("off-diagonal fraction $D$")
    ax[0].set_title("does the basis diagonalize the propagator?"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].plot([r["align"] for r in rows], [r["D_fourier"] for r in rows], "o-", color="#31557F")
    for r in rows:
        ax[1].annotate(f"β={r['beta']:g}", (r["align"], r["D_fourier"]), fontsize=7,
                       xytext=(3, 3), textcoords="offset points")
    ax[1].set_xlabel("alignment of true eigenbasis with Fourier"); ax[1].set_ylabel("$D_{Fourier}$")
    ax[1].set_title("the meter"); ax[1].grid(alpha=.3); ax[1].invert_xaxis()
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "phys_wp1_meter.png"), dpi=160); plt.close(fig)


def plot_discovery(d):
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    w, m, r = d["well-specified"], d["mis-specified"], d["re-specified"]
    b = [x["beta"] for x in w]
    ax[0].plot(b, [x["D_fourier"] for x in w], "o--", color="#31557F", label="Fourier (start)")
    ax[0].plot(b, [x["D_discovered"] for x in w], "s-", color="#B0173A", label="discovered SL basis")
    ax[0].plot(b, [x["D_discovered"] for x in m], "s-", color="#B0173A", alpha=.3, label="family too narrow")
    ax[0].set_yscale("symlog", linthresh=1e-4)
    ax[0].set_xlabel(r"$\beta$"); ax[0].set_ylabel("off-diagonal fraction"); ax[0].legend(fontsize=8)
    ax[0].set_title("discovery without an oracle"); ax[0].grid(alpha=.3)
    x = w[-1]
    ax[1].plot(X, x["a_true"], "k-", lw=2, label="true $a(x)$")
    ax[1].plot(X, x["a_rec"], "-", color="#B0173A", lw=1.2, label="recovered (up to scale)")
    ax[1].set_title(f"recovered coefficient, β={x['beta']:g}  (r = {x['corr']:.4f})")
    ax[1].set_xlabel("x"); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
    ax[2].plot(b, [x["corr"] for x in w], "s-", color="#B0173A", label="well-specified (M=3)")
    ax[2].plot(b, [x["corr"] for x in m], "^-", color="#888888", label="mis-specified (M=3)")
    ax[2].plot(b, [x["corr"] for x in r], "v-", color="#2F6B3D", label="re-specified (M=5)")
    ax[2].set_ylim(-0.1, 1.05); ax[2].set_xlabel(r"$\beta$"); ax[2].set_ylabel("corr(recovered, true)")
    ax[2].set_title("recovery vs heterogeneity"); ax[2].legend(fontsize=8); ax[2].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "phys_wp1_discovery.png"), dpi=160); plt.close(fig)


if __name__ == "__main__":
    meter = wp1()
    plot_meter(meter)
    disc = wp1b()
    plot_discovery(disc)
    slim = {k: [{kk: vv for kk, vv in row.items() if kk not in ("a_true", "a_rec", "theta")}
                for row in v] for k, v in disc.items()}
    json.dump(dict(meter=meter, discovery=slim, dt=DT, betas=BETAS),
              open(os.path.join(RES, "phys_wp1.json"), "w"), indent=1)
    print("done")
