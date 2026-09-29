"""WP4 + WP4b — what the reading is worth: extrapolation, and hardening by rollout training.

At eps = 1.0, one-step models sharing the same frozen DMD head:
  linear      DMD only
  mlp         + black-box MLP branch
  quad        + the discovered u u_x term, one explicit step (linear map on [u, u_x, u u_x])
  quadint4    + the same term sub-stepped 4 times (structured integrator, shared weights)
trained only on one-step pairs, then rolled out 30 steps (a horizon never seen) from
in-distribution test initial conditions and from higher-amplitude (OOD, amplitude 1.6) ones;
plus an amplitude sweep (1.0 .. 1.6, fresh initial conditions) that locates where each
explicit surrogate leaves its stability margin.

WP4b: rollout fine-tuning (K = 5 unrolled steps, gradient clipping) on top of the converged
one-step models, 3 seeds (seeds vary the network initialisation only; data and test initial
conditions are fixed), plus a physical check: viscous Burgers must dissipate energy.

Outputs: figures/phys_wp4_rollout.png, figures/phys_wp4_amplitude.png,
         figures/phys_wp4b_hardening.png, results/phys_wp4.json
"""
import os, sys, json, time
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from data import load, pairs, NSNAP, DELTA, DT, SUB, AMP_OOD
from common import integrate, random_ic
from models import dmd, Gated, train_1step, finetune_rollout, rollout, N

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
FIG = os.path.join(ROOT, "figures"); RES = os.path.join(ROOT, "results")
EPS = 1.0; STEPS = 30; SEEDS = [0, 1, 2]; KINDS = ["mlp", "quad", "quadint4"]
EPOCHS = 1500; LAM = 1e-3; WD = 3e-5
AMPS = [1.0, 1.2, 1.4, 1.6]; DIVERGED = 10.0          # relative error above this = diverged
LABEL = {"linear": "linear only (DMD)", "mlp": "+ black-box branch", "quad": r"+ discovered $u\,u_x$, one step",
         "quadint4": r"+ discovered $u\,u_x$, 4 sub-steps"}
COL = {"linear": "#31557F", "mlp": "#888888", "quad": "#E08A9B", "quadint4": "#7A0F28"}


def rollout_np(A, u0, steps):
    u = u0.copy(); out = [u.copy()]
    for _ in range(steps):
        u = u @ A.T; out.append(u.copy())
    return np.array(out)


def error_curve(pred, truth):
    with np.errstate(over="ignore", invalid="ignore"):
        e = np.array([np.linalg.norm(pred[s] - truth[s]) / np.linalg.norm(truth[s]) for s in range(len(pred))])
    e[~np.isfinite(e)] = np.inf
    return e


def energy_growth_fraction(pred):
    with np.errstate(over="ignore", invalid="ignore"):
        E = (pred ** 2).sum(-1)
        d = np.diff(E, axis=0)
    d = d[np.isfinite(d)]
    return float((d > 1e-9).mean()) if d.size else float("nan")


def fresh_truth(amp, n=10, seed=5000):
    rng = np.random.default_rng(seed + int(round(amp * 100)))
    T = (NSNAP - 1) * DELTA
    return np.array([integrate(random_ic(rng, amp=amp), EPS, T, dt=DT, sub=SUB) for _ in range(n)])


def run():
    tr, te, ood = load(EPS)
    Xtr, Ytr = pairs(tr)
    A = dmd(Xtr, Ytr)
    truth_id = np.transpose(te[:, :STEPS + 1], (1, 0, 2))
    truth_ood = np.transpose(ood[:, :STEPS + 1], (1, 0, 2))
    curves = {"linear": dict(id=error_curve(rollout_np(A, te[:, 0], STEPS), truth_id),
                             ood=error_curve(rollout_np(A, ood[:, 0], STEPS), truth_ood))}
    res = {"linear": {"id": [curves["linear"]["id"][-1]], "ood": [curves["linear"]["ood"][-1]],
                      "energy_id": [energy_growth_fraction(rollout_np(A, te[:, 0], STEPS))],
                      "energy_ood": [energy_growth_fraction(rollout_np(A, ood[:, 0], STEPS))]}}
    for kind in KINDS:
        for stage in ["1step", "rollout"]:
            res[f"{kind}_{stage}"] = {"id": [], "ood": [], "energy_id": [], "energy_ood": []}
    amp_truth = {a: fresh_truth(a) for a in AMPS}
    amp_res = {"linear": {}}
    for a, t in amp_truth.items():
        tt = np.transpose(t[:, :STEPS + 1], (1, 0, 2))
        amp_res["linear"][a] = float(error_curve(rollout_np(A, t[:, 0], STEPS), tt)[-1])
    for seed in SEEDS:
        for kind in KINDS:
            t0 = time.time()
            m = train_1step(Gated(A, kind, seed=seed), Xtr, Ytr, epochs=EPOCHS, lam=LAM, wd=WD, seed=seed)
            if seed == 0:                                       # amplitude sweep, one-step models
                amp_res[kind] = {}
                for a, t in amp_truth.items():
                    tt = np.transpose(t[:, :STEPS + 1], (1, 0, 2))
                    amp_res[kind][a] = float(error_curve(rollout(m, t[:, 0], STEPS), tt)[-1])
            for stage in ["1step", "rollout"]:
                if stage == "rollout":
                    m = finetune_rollout(m, tr, K=5, epochs=300, lr=3e-4, clip=1.0, seed=seed)
                p_id = rollout(m, te[:, 0], STEPS); p_ood = rollout(m, ood[:, 0], STEPS)
                c_id, c_ood = error_curve(p_id, truth_id), error_curve(p_ood, truth_ood)
                key = f"{kind}_{stage}"
                res[key]["id"].append(c_id[-1]); res[key]["ood"].append(c_ood[-1])
                res[key]["energy_id"].append(energy_growth_fraction(p_id))
                res[key]["energy_ood"].append(energy_growth_fraction(p_ood))
                curves.setdefault(key, []).append(dict(id=c_id, ood=c_ood))
                print(f"seed {seed} {key:17s} err@30 id={c_id[-1]:.3f} ood={c_ood[-1]:.3f}  "
                      f"energy-growth steps id={100*res[key]['energy_id'][-1]:.1f}% "
                      f"ood={100*res[key]['energy_ood'][-1]:.1f}%  ({time.time()-t0:.0f}s)", flush=True)
    return A, res, curves, amp_res


def stat(v):
    v = np.array(v, dtype=float); fin = v[np.isfinite(v) & (v < DIVERGED)]
    return dict(mean=float(fin.mean()) if fin.size else float("inf"), std=float(fin.std()) if fin.size else float("nan"),
                n=int(len(v)), diverged=int(len(v) - fin.size))


def summarize(res):
    return {k: {c: stat(v[c]) for c in ["id", "ood", "energy_id", "energy_ood"] if len(v[c])} for k, v in res.items()}


def _plot_curves(ax, curves_by_key, c, spec):
    steps = np.arange(STEPS + 1)
    for key, col, lab in spec:
        arr = np.array([d[c] for d in curves_by_key[key]]) if key != "linear" else np.array([curves_by_key["linear"][c]])
        arr = np.where(np.isfinite(arr), arr, DIVERGED)
        mu, sd = arr.mean(0), arr.std(0)
        ax.plot(steps, mu, "-", color=col, label=lab)
        if len(arr) > 1:
            ax.fill_between(steps, mu - sd, mu + sd, color=col, alpha=.2)
    ax.set_ylim(0, 1.5); ax.set_xlabel("rollout step"); ax.set_ylabel("relative $L_2$ error (clipped at 1.5)")
    ax.grid(alpha=.3); ax.legend(fontsize=7)


def plot_wp4(curves, amp_res):
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.7))
    spec = [("linear", COL["linear"], LABEL["linear"]), ("mlp_1step", COL["mlp"], LABEL["mlp"]),
            ("quad_1step", COL["quad"], LABEL["quad"]), ("quadint4_1step", COL["quadint4"], LABEL["quadint4"])]
    one = {k: ([curves[k][0]] if k != "linear" else curves[k]) for k in curves}
    _plot_curves(ax[0], one, "id", spec); ax[0].set_title("in-distribution rollout (seed 0)")
    _plot_curves(ax[1], one, "ood", spec); ax[1].set_title(f"unseen amplitude {AMP_OOD:g} (seed 0)")
    for k in ["linear", "mlp", "quad", "quadint4"]:
        v = np.array([amp_res[k][a] for a in AMPS]); v = np.where(np.isfinite(v) & (v < DIVERGED), v, np.nan)
        ax[2].plot(AMPS, v, "o-", color=COL[k], label=LABEL[k])
        for a, x in zip(AMPS, v):
            if np.isnan(x): ax[2].annotate("diverged", (a, 1.42), color=COL[k], fontsize=7, ha="center", va="top")
    ax[2].set_ylim(0, 1.5); ax[2].set_xlabel("initial-condition amplitude (training: 1.0)"); ax[2].set_ylabel("error at step 30")
    ax[2].set_title("stability margin in amplitude"); ax[2].grid(alpha=.3); ax[2].legend(fontsize=7, loc="center right")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "phys_wp4_rollout.png"), dpi=160); plt.close(fig)


def plot_wp4b(curves):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.7))
    spec = [("mlp_1step", "#BBBBBB", "black box, one-step"), ("mlp_rollout", "#555555", "black box, + rollout"),
            ("quadint4_1step", "#E08A9B", r"$u\,u_x$ integrator, one-step"), ("quadint4_rollout", "#7A0F28", r"$u\,u_x$ integrator, + rollout")]
    _plot_curves(ax[0], curves, "id", spec); ax[0].set_title("in-distribution rollout (3 seeds)")
    _plot_curves(ax[1], curves, "ood", spec); ax[1].set_title("unseen amplitude (3 seeds)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "phys_wp4b_hardening.png"), dpi=160); plt.close(fig)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "replot":
        w = json.load(open(os.path.join(RES, "phys_wp4.json")))
        curves = {k: (v if k == "linear" else [{c: np.array(d[c], dtype=float) for c in d} for d in v]) for k, v in w["curves"].items()}
        curves["linear"] = {c: np.array(v, dtype=float) for c, v in curves["linear"].items()}
        amp_res = {k: {float(a): (np.inf if x is None else x) for a, x in d.items()} for k, d in w["amplitude_sweep"].items()}
        plot_wp4(curves, amp_res); plot_wp4b(curves); print("replotted"); sys.exit()
    A, res, curves, amp_res = run()
    plot_wp4(curves, amp_res); plot_wp4b(curves)
    summ = summarize(res)
    for k, v in summ.items():
        print(f"{k:17s} id {v['id']['mean']:.3f}±{v['id']['std']:.3f} (div {v['id']['diverged']})  "
              f"ood {v['ood']['mean']:.3f}±{v['ood']['std']:.3f} (div {v['ood']['diverged']})")
    print("amplitude sweep (err@30):", {k: {str(a): round(x, 3) for a, x in d.items()} for k, d in amp_res.items()})
    tojson = lambda o: (None if isinstance(o, float) and not np.isfinite(o) else np.asarray(o).tolist())
    json.dump(dict(summary=summ, amplitude_sweep={k: {str(a): (x if np.isfinite(x) else None) for a, x in d.items()} for k, d in amp_res.items()},
                   seeds=SEEDS, eps=EPS, steps=STEPS, amps=AMPS,
                   curves={k: (v if k == "linear" else [{c: np.asarray(d[c]).tolist() for c in d} for d in v]) for k, v in curves.items()}),
              open(os.path.join(RES, "phys_wp4.json"), "w"), indent=1, default=tojson)
    print("done")
