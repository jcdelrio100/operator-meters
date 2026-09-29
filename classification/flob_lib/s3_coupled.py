"""
S3 = coupled superstructure: pool S1 (dictionary) AND S2 (learned transform) into one
self-organizing head with a GATE on the S2 branch (designs 1 + 3).

    y = Lin_dict(x_dict) + g_nl * NLblock(x_dict) + g_s2 * Lin_s2(x_learn)
    loss = CE + lambda(|g_nl| + |g_s2|)

Both branches carry K=16 F-ANOVA-selected features. The learned transform U is
warm-started from the dictionary's best fixed transform (DFT / per-axis DFT) and
task-trained (its 'prior learning stage'). g_s2 starts open; the L1 prunes it, so on
classical modalities S3 collapses to S1 (g_s2 -> 0) and only engages S2 where it helps.
g_s2 is a free 'did we need a learned transform?' diagnostic. S3 should be >= max(S1,S2).
"""
import sys, os, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(__file__)
for p in (".", "../tasks", "../src"):
    sys.path.insert(0, os.path.join(HERE, p))
import superstructure_multimodal as MM
from learned_transform import UnitaryClassifier
from structured_learned import MultilinearClassifier
from warmstart import dft_matrix, warm_start_flat, warm_start_separable, train_ws

SEEDS = 5; K = 16
FLAT = {"1D global (Fourier)", "1D localized (wavelet)", "hidden basis (rotated)", "XOR in hidden basis"}


def learned_branch(name, Xtr, ytr, Xte, yte, nc, seed):
    if name in FLAT:
        F1 = Xtr.astype(np.float64); F2 = Xte.astype(np.float64); d = F1.shape[1]
        m = UnitaryClassifier(d, nc, seed=seed); warm_start_flat(m.U, dft_matrix(d))
    else:
        shape = Xtr.shape[1:]; mu = Xtr.mean(0, keepdims=True); sd = Xtr.std(0, keepdims=True) + 1e-9
        F1 = ((Xtr - mu) / sd).astype(np.float64); F2 = ((Xte - mu) / sd).astype(np.float64)
        m = MultilinearClassifier(shape, nc, seed=seed)
        warm_start_separable(m.U, [dft_matrix(int(x)) for x in shape])
    train_ws(m, F1, ytr, F2, yte, epochs=150)                       # Stage A: task-train the transform
    with torch.no_grad():
        Z1 = m.features(torch.tensor(F1)).numpy(); Z2 = m.features(torch.tensor(F2)).numpy()
    return MM.std_select(Z1, ytr, Z2, nc)


class S3Head(nn.Module):
    def __init__(self, Kd, Kl, C, h=32):
        super().__init__()
        self.lin_d = nn.Linear(Kd, C)
        self.nl = nn.Sequential(nn.Linear(Kd, h), nn.ReLU(), nn.Linear(h, C))
        self.lin_s2 = nn.Linear(Kl, C)
        self.g_nl = nn.Parameter(torch.tensor(1.0)); self.g_s2 = nn.Parameter(torch.tensor(1.0))
        self.Kd, self.Kl, self.C, self.h = Kd, Kl, C, h
    def forward(self, xd, xl):
        return self.lin_d(xd) + self.g_nl * self.nl(xd) + self.g_s2 * self.lin_s2(xl)


def fit_s3(Ad, Al, ytr, Bd, Bl, yte, nc, lam=0.25, ep=400, lr=5e-3, eps=0.15):
    Ad = torch.tensor(Ad, dtype=torch.float32); Al = torch.tensor(Al, dtype=torch.float32)
    Bd = torch.tensor(Bd, dtype=torch.float32); Bl = torch.tensor(Bl, dtype=torch.float32)
    yt = torch.tensor(ytr); ye = torch.tensor(yte)
    net = S3Head(Ad.shape[1], Al.shape[1], nc)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    for _ in range(ep):
        net.train(); opt.zero_grad()
        loss = Fn.cross_entropy(net(Ad, Al), yt) + lam * (net.g_nl.abs() + net.g_s2.abs())
        loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        acc = (net(Bd, Bl).argmax(1) == ye).float().mean().item()
        g_s2 = float(net.g_s2.abs()); g_nl = float(net.g_nl.abs())
    params = net.Kd * nc + nc
    if g_nl > eps: params += net.Kd * net.h + net.h + net.h * nc + nc
    if g_s2 > eps: params += net.Kl * nc + nc
    return acc, g_s2, params


TASKS = ["1D global (Fourier)", "1D localized (wavelet)", "video (motion)", "texture (28x28)",
         "text (topic)", "XOR in hidden basis", "hidden basis (rotated)"]


JS = os.path.join(HERE, "s3_results.json")


def run(tasks=None):
    tasks = tasks or TASKS
    out = json.load(open(JS)) if os.path.exists(JS) else {}
    print(f"{SEEDS} seeds\n  {'task':24s} {'S1':>13s} {'S2warm':>13s} {'S3':>13s} {'g_s2':>6s}")
    for name in tasks:
        S1, S2, S3, G = [], [], [], []
        for s in range(SEEDS):
            Xtr, ytr, Xte, yte, nc, reps = MM.load(name, s)
            Ad, Bd, _ = MM.s1_feats(Xtr, ytr, Xte, yte, nc, reps)
            S1.append(MM.fit_head(Ad, ytr, Bd, yte, nc)[0])
            Al, Bl = learned_branch(name, Xtr, ytr, Xte, yte, nc, s)
            S2.append(MM.fit_head(Al, ytr, Bl, yte, nc)[0])
            acc, g, _ = fit_s3(Ad, Al, ytr, Bd, Bl, yte, nc)
            S3.append(acc); G.append(g)
        out[name] = {"S1": S1, "S2": S2, "S3": S3, "g_s2": G, "seeds": SEEDS}
        def ms(v): return f"{np.mean(v):.3f}+/-{np.std(v):.3f}"
        print(f"  {name:24s} {ms(S1):>13s} {ms(S2):>13s} {ms(S3):>13s} {np.mean(G):>6.2f}")
        json.dump(out, open(JS, "w"), indent=1)


def plot():
    out = json.load(open(JS)); names = [t for t in TASKS if t in out]
    x = np.arange(len(names)); w = 0.16
    m = lambda t, k: float(np.mean(out[t][k])); sd = lambda t, k: float(np.std(out[t][k]))
    have_ref = all(("SNN" in out[t] and "deep" in out[t]) for t in names)
    Y0 = 0.5                                    # cut y-axis: bars are near-saturated
    fig, ax = plt.subplots(figsize=(14, 5.2))
    series = [("S1", "#555555", "S1 Pooled dictionary"),
              ("S2", "#d62728", "S2 Learned (warm-start)"),
              ("S3", "#1f77b4", "S3 Coupled (dict + gated learned)")]
    if have_ref:
        series = [("SNN", "#9467bd", "SNN (raw input, reference)"),
                  ("deep", "#e377c2", "Deep MLP (raw input, reference)")] + series
    off = {k: (i - (len(series) - 1) / 2) * w for i, (k, _, _) in enumerate(series)}
    for k, col, lab in series:
        hatch = "//" if k in ("SNN", "deep") else None
        ax.bar(x + off[k], [m(t, k) for t in names], w, yerr=[sd(t, k) for t in names],
               capsize=2, color=col, label=lab, hatch=hatch, edgecolor="white", linewidth=0.3)
    for i, t in enumerate(names):
        ax.annotate(f"g={m(t,'g_s2'):.2f}", (x[i] + off['S3'], m(t, "S3") + sd(t, "S3") + 0.006),
                    fontsize=7, ha="center", va="bottom", color="#1f77b4")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8.5)
    ax.set_ylabel("Test accuracy"); ax.set_ylim(Y0, 1.02); ax.axhline(Y0, color="k", lw=0.6)
    ax.set_title(f"S3 coupled superstructure vs S1/S2 and SNN/deep references "
                 f"(mean +/- std, {out[names[0]]['seeds']} seeds; y-axis starts at {Y0}; gate g_s2 shown)")
    ax.legend(fontsize=8, loc="lower left", ncol=2); ax.grid(axis="y", alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(HERE, "s3_coupled.png"), dpi=130); print("saved s3_coupled.png")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "plot": plot()
    elif len(sys.argv) > 2 and sys.argv[1] == "run": run(sys.argv[2].split(";"))
    else: run()
