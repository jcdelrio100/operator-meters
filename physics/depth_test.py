"""WP2b — depth is not the lever; structure is.

Four nonlinear-branch designs on the identical frozen-DMD residual, trained one-step with the
same budget, scored by held-out capture of the DMD residual across the nonlinearity sweep:
  mlp      one-step MLP (no depth)
  resint4  residual integrator, 4 stages (neural-ODE-like)
  resint8  residual integrator, 8 stages
  quad     structure-matched quadratic branch on [u, u_x, u*u_x]
  quadint4 the same quadratic term sub-stepped 4 times (structure + depth, shared weights)

Outputs: figures/depth_test.png, results/depth_test.json
"""
import os, sys, json, time
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from data import EPS, load, pairs
from models import dmd, Gated, train_1step, evaluate

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
FIG = os.path.join(ROOT, "figures"); RES = os.path.join(ROOT, "results")
KINDS = ["mlp", "resint4", "resint8", "quad", "quadint4"]
LABEL = {"mlp": "one-step MLP (no depth)", "resint4": "residual integrator, 4 stages",
         "resint8": "residual integrator, 8 stages", "quad": r"structure-matched $u\,\partial_x u$", "quadint4": r"structured integrator, 4 sub-steps"}
EPOCHS = 1500; LAM = 1e-3; WD = 3e-5
SWEEP = [e for e in EPS if e > 0]


def run():
    out = {k: [] for k in KINDS}
    for eps in SWEEP:
        tr, te, _ = load(eps)
        Xtr, Ytr = pairs(tr); Xte, Yte = pairs(te)
        A = dmd(Xtr, Ytr)
        for k in KINDS:
            t0 = time.time()
            m = train_1step(Gated(A, k, seed=0), Xtr, Ytr, epochs=EPOCHS, lam=LAM, wd=WD, seed=0)
            ev = evaluate(m, Xte, Yte); ev["eps"] = eps
            out[k].append(ev)
            print(f"eps={eps:3.1f} {k:8s} capture={100*ev['capture']:6.1f}%  firing={ev['firing']:.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    return out


def plot(out):
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    cols = {"mlp": "#888888", "resint4": "#31557F", "resint8": "#7FA0C8", "quad": "#B0173A", "quadint4": "#5A0A1E"}
    for k in KINDS:
        ax.plot([r["eps"] for r in out[k]], [100 * r["capture"] for r in out[k]], "o-",
                color=cols[k], label=LABEL[k])
    ax.axhline(0, color="k", lw=.5)
    ax.set_xlabel(r"nonlinearity $\varepsilon$"); ax.set_ylabel("held-out capture of DMD residual (%)")
    ax.set_title("depth versus structure"); ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "depth_test.png"), dpi=160); plt.close(fig)


if __name__ == "__main__":
    out = run(); plot(out)
    summary = {k: float(np.mean([r["capture"] for r in out[k] if r["eps"] >= 0.7])) for k in KINDS}
    print("mean held-out capture (eps >= 0.7):", {k: f"{100*v:.1f}%" for k, v in summary.items()})
    json.dump(dict(sweep=out, mean_capture_eps_ge_0p7=summary, epochs=EPOCHS, lam=LAM, wd=WD),
              open(os.path.join(RES, "depth_test.json"), "w"), indent=1)
    print("done")
