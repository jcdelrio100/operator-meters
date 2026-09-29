"""WP2 — nonlinearity meter and the gate that fires.

For every eps of the Burgers sweep: fit the closed-form linear one-step propagator A (DMD)
on train pairs; its held-out residual is the nonlinearity meter.  Then freeze A and train a
gated black-box branch  u1 = A u0 + g NL(u0)  on the residual with an L1 penalty on g.
Report the branch firing amplitude ||g NL(u)|| / ||A u|| and the held-out capture.

Outputs: figures/phys_wp2_meter.png, results/phys_wp2.json
"""
import os, sys, json, time
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from data import EPS, load, pairs
from models import dmd, Gated, train_1step, evaluate

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
FIG = os.path.join(ROOT, "figures"); RES = os.path.join(ROOT, "results")
EPOCHS = 1500; LAM = 1e-3; WD = 3e-5


def run():
    rows = []
    for eps in EPS:
        tr, te, _ = load(eps)
        Xtr, Ytr = pairs(tr); Xte, Yte = pairs(te)
        A = dmd(Xtr, Ytr)
        t0 = time.time()
        m = train_1step(Gated(A, "mlp", seed=0), Xtr, Ytr, epochs=EPOCHS, lam=LAM, wd=WD, seed=0)
        ev = evaluate(m, Xte, Yte)
        ev["eps"] = eps
        if ev["res_dmd"] < 1e-6: ev["capture"] = 0.0          # nothing to capture at eps = 0
        rows.append(ev)
        print(f"eps={eps:3.1f}  DMD residual={ev['res_dmd']:.4f}  after branch={ev['res_after']:.4f}  "
              f"capture={100*ev['capture']:5.1f}%  firing={ev['firing']:.4f}  g={ev['g']:+.3f}  "
              f"({time.time()-t0:.0f}s)")
    return rows


def plot(rows):
    e = [r["eps"] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot(e, [r["res_dmd"] for r in rows], "o-", color="#31557F", label="DMD (linear) residual")
    ax[0].plot(e, [r["res_after"] for r in rows], "s-", color="#B0173A", label="after gated branch")
    ax[0].set_xlabel(r"nonlinearity $\varepsilon$"); ax[0].set_ylabel("held-out relative residual")
    ax[0].set_title("nonlinearity meter"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].plot(e, [r["firing"] for r in rows], "o-", color="#B0173A")
    ax[1].set_xlabel(r"nonlinearity $\varepsilon$"); ax[1].set_ylabel(r"$\|g\,NL(u)\| / \|Au\|$")
    ax[1].set_title("how much the branch fires"); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "phys_wp2_meter.png"), dpi=160); plt.close(fig)


if __name__ == "__main__":
    rows = run(); plot(rows)
    json.dump(dict(rows=rows, epochs=EPOCHS, lam=LAM, wd=WD),
              open(os.path.join(RES, "phys_wp2.json"), "w"), indent=1)
    print("done")
