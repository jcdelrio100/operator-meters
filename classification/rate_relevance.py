"""Rate-distortion versus rate-relevance — selecting K = 16 dictionary coefficients by ENERGY
(what a codec keeps) versus by class F-score (what the task needs), same pooled dictionary,
same head.  Compression toward the signal and compression toward the label are different
objectives; the gap between the two bars is the information a lossy front-end throws away.

Outputs: figures/rate_relevance.png, results/rate_relevance.json
"""
import os, sys, json
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(HERE, "flob_lib"))
import superstructure_multimodal as MM
FIG = os.path.join(ROOT, "figures"); RES = os.path.join(ROOT, "results")
torch.set_num_threads(2)
TASKS = ["1D global (Fourier)", "1D localized (wavelet)", "hidden basis (rotated)"]
SEEDS = [0, 1, 2]; K = 16


def select(P1, ytr, P2, nc, by):
    if by == "energy":
        score = (P1 ** 2).mean(0)
    else:
        score = MM.fscore(P1, ytr, nc)
    idx = np.argsort(score)[::-1][:K]
    A, B = P1[:, idx], P2[:, idx]; mu, sd = A.mean(0), A.std(0) + 1e-9
    return (A - mu) / sd, (B - mu) / sd


def run():
    out = {}
    for name in TASKS:
        out[name] = {"energy": [], "fscore": []}
        for s in SEEDS:
            torch.manual_seed(s)
            Xtr, ytr, Xte, yte, nc, reps = MM.load(name, s)
            P1 = np.concatenate([f(Xtr) for f in reps.values()], 1); P2 = np.concatenate([f(Xte) for f in reps.values()], 1)
            for by in ["energy", "fscore"]:
                A, B = select(P1, ytr, P2, nc, by)
                out[name][by].append(MM.fit_head(A, ytr, B, yte, nc)[0])
        print(f"{name:24s} energy-selected {np.mean(out[name]['energy']):.3f}  "
              f"task-selected {np.mean(out[name]['fscore']):.3f}", flush=True)
    return out


def plot(out):
    fig, ax = plt.subplots(figsize=(6.5, 3.6)); x = np.arange(len(TASKS)); w = 0.36
    ax.bar(x - w / 2, [np.mean(out[t]["energy"]) for t in TASKS], w, yerr=[np.std(out[t]["energy"]) for t in TASKS],
           color="#31557F", capsize=3, label="K = 16 by energy (rate–distortion)")
    ax.bar(x + w / 2, [np.mean(out[t]["fscore"]) for t in TASKS], w, yerr=[np.std(out[t]["fscore"]) for t in TASKS],
           color="#B0173A", capsize=3, label="K = 16 by class F-score (rate–relevance)")
    ax.set_xticks(x); ax.set_xticklabels(TASKS, fontsize=8); ax.set_ylabel("test accuracy"); ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, loc="lower left"); ax.grid(axis="y", alpha=.3)
    ax.set_title("same dictionary, same head: what you keep decides")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "rate_relevance.png"), dpi=160); plt.close(fig)


if __name__ == "__main__":
    out = run(); plot(out)
    json.dump(dict(out=out, K=K, seeds=SEEDS), open(os.path.join(RES, "rate_relevance.json"), "w"), indent=1)
    print("done")
