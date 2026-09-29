"""Classification gate law — the gate g_s2 of the S3 coupling as a meter of information.

Controlled family  rotated(theta):  the class-separating coordinates live in a hidden
orthogonal basis Q(theta) = F expm(theta * log(F^T Q)),  a geodesic from the real Fourier
basis F (theta = 0: the fixed dictionary sees the classes perfectly) to a random orthogonal
basis Q (theta = 1: nothing classical exposes them).  Same class construction as the
'hidden basis' task of the companion study.

For every theta and seed:
  T_dict    K = 16 F-score-selected dictionary features (S1, fixed)
  T_learned K = 16 features of the learned unitary warm-started at the DFT (S2)
  G  = I(T_learned; Y) - I(T_dict; Y)   in bits, with I(T;Y) ~ H(Y) - CE_test(head on T)/ln 2
  g_s2 = the S3 protective gate on the learned branch (L1-penalised)
Then the four 1-D tasks of the companion study are placed on the same axes, and a rectified
linear law  g = a + b * max(G, 0)  is fitted.

Outputs: figures/gap_law.png, results/gap_law.json
"""
import os, sys, json, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn
from scipy.linalg import logm, expm
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(HERE, "flob_lib"))
import superstructure_multimodal as MM
import s3_coupled as S3
FIG = os.path.join(ROOT, "figures"); RES = os.path.join(ROOT, "results")
os.makedirs(FIG, exist_ok=True); os.makedirs(RES, exist_ok=True)
torch.set_num_threads(2)

N = 128; THETAS = [0.0, 0.2, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9, 1.0]; SEEDS = [0, 1, 2]
REAL_TASKS = ["1D global (Fourier)", "1D localized (wavelet)", "hidden basis (rotated)", "XOR in hidden basis"]


# ------------------------------------------------------------- the rotated family
def real_fourier(n):
    B = [np.ones(n) / np.sqrt(n)]
    t = np.arange(n)
    for k in range(1, n // 2):
        B.append(np.cos(2 * np.pi * k * t / n) * np.sqrt(2 / n)); B.append(np.sin(2 * np.pi * k * t / n) * np.sqrt(2 / n))
    B.append(np.cos(np.pi * t) / np.sqrt(n))
    return np.array(B).T                                         # columns = basis vectors


_F = real_fourier(N)
_Q, _ = np.linalg.qr(np.random.default_rng(12345).normal(size=(N, N)))
_S = logm(_F.T @ _Q); _S = 0.5 * (_S - _S.T).real               # skew generator of the geodesic


def basis(theta):
    return _F @ expm(theta * _S)


def rotated_task(theta, seed, n_per_class=250, nc=4, active=3, noise=0.35):
    """Class c is separated by `active` coordinates in basis(theta)."""
    B = basis(theta)
    fix = np.random.default_rng(777)
    coords = fix.permutation(N)[:nc * active].reshape(nc, active)
    rng = np.random.default_rng(seed); X, y = [], []
    for c in range(nc):
        for _ in range(n_per_class):
            z = noise * rng.standard_normal(N)
            z[coords[c]] += rng.uniform(1.5, 2.5, active) * rng.choice([-1.0, 1.0], active)
            X.append(B @ z); y.append(c)
    X = np.array(X); X = (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-9)
    y = np.array(y)
    rng2 = np.random.default_rng(seed + 1); idx = rng2.permutation(len(X)); nte = int(0.3 * len(X))
    te, tr = idx[:nte], idx[nte:]
    return X[tr], y[tr], X[te], y[te], nc


# ------------------------------------------------------------- information estimate
def mi_bits(A, ytr, B, yte, nc, ep=300, lr=5e-3):
    """I(T;Y) in bits, estimated as H(Y) - CE_test of a small head trained on T (a lower bound)."""
    A = torch.tensor(A, dtype=torch.float32); B = torch.tensor(B, dtype=torch.float32)
    yt = torch.tensor(ytr); ye = torch.tensor(yte)
    net = nn.Sequential(nn.Linear(A.shape[1], 32), nn.ReLU(), nn.Linear(32, nc))
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    for _ in range(ep):
        opt.zero_grad(); Fn.cross_entropy(net(A), yt).backward(); opt.step()
    with torch.no_grad():
        ce = Fn.cross_entropy(net(B), ye).item()
    p = np.bincount(yte, minlength=nc) / len(yte); HY = -(p[p > 0] * np.log(p[p > 0])).sum()
    return float(np.clip(HY - ce, 0, HY) / np.log(2))       # a lower bound; clipped to [0, H(Y)]


def one_point(Xtr, ytr, Xte, yte, nc, seed, name="hidden basis (rotated)"):
    torch.manual_seed(seed)
    Ad, Bd, _ = MM.s1_feats(Xtr, ytr, Xte, yte, nc, MM.reps1d)
    Al, Bl = S3.learned_branch(name, Xtr, ytr, Xte, yte, nc, seed)
    I_d = mi_bits(Ad, ytr, Bd, yte, nc); I_l = mi_bits(Al, ytr, Bl, yte, nc)
    acc, g, _ = S3.fit_s3(Ad, Al, ytr, Bd, Bl, yte, nc)
    return dict(I_dict=I_d, I_learned=I_l, G=I_l - I_d, g_s2=g, acc_s3=acc)


def run():
    rows = []
    for th in THETAS:
        for s in SEEDS:
            t0 = time.time()
            r = one_point(*rotated_task(th, s), s); r.update(theta=th, seed=s, task="rotated")
            rows.append(r)
            print(f"theta={th:.1f} seed={s}  I_dict={r['I_dict']:.2f}  I_learned={r['I_learned']:.2f}  "
                  f"G={r['G']:+.2f} bits  g_s2={r['g_s2']:.3f}  ({time.time()-t0:.0f}s)", flush=True)
    for name in REAL_TASKS:
        for s in SEEDS:
            t0 = time.time()
            Xtr, ytr, Xte, yte, nc, _ = MM.load(name, s)
            r = one_point(Xtr, ytr, Xte, yte, nc, s, name); r.update(theta=None, seed=s, task=name)
            rows.append(r)
            print(f"{name:24s} seed={s}  G={r['G']:+.2f} bits  g_s2={r['g_s2']:.3f}  ({time.time()-t0:.0f}s)", flush=True)
    return rows


def fit_law(rows):
    """Fit on the rotated family only; the four benchmark tasks are held out and scored."""
    from scipy.optimize import curve_fit
    rot = [r for r in rows if r["task"] == "rotated"]; real = [r for r in rows if r["task"] != "rotated"]
    Gp = lambda rs: np.maximum(np.array([r["G"] for r in rs]), 0); gg = lambda rs: np.array([r["g_s2"] for r in rs])
    b, a = np.polyfit(Gp(rot), gg(rot), 1)
    lin = lambda x: a + b * x
    sat_f = lambda x, c, G0: c * (1 - np.exp(-x / G0))
    (c, G0), _ = curve_fit(sat_f, Gp(rot), gg(rot), p0=[0.6, 0.5])
    sat = lambda x: sat_f(x, c, G0)
    rc = lambda f, rs: float(np.corrcoef(f(Gp(rs)), gg(rs))[0, 1])
    shut = [r["g_s2"] for r in rows if r["G"] <= 0]
    return dict(linear=dict(a=float(a), b=float(b), r_family=rc(lin, rot), r_heldout=rc(lin, real), r_all=rc(lin, rows)),
                saturating=dict(c=float(c), G0=float(G0), r_family=rc(sat, rot), r_heldout=rc(sat, real), r_all=rc(sat, rows)),
                n=int(len(rows)), n_family=int(len(rot)), n_heldout=int(len(real)),
                gate_max_when_G_le_0=float(max(shut)), n_G_le_0=int(len(shut)))


def plot(rows, law):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    rot = [r for r in rows if r["task"] == "rotated"]
    sc = ax.scatter([r["G"] for r in rot], [r["g_s2"] for r in rot], c=[r["theta"] for r in rot],
                    cmap="viridis", s=36, label=r"rotated($\theta$) family (colour = $\theta$)")
    mk = {"1D global (Fourier)": "s", "1D localized (wavelet)": "^", "hidden basis (rotated)": "D", "XOR in hidden basis": "P"}
    for name in REAL_TASKS:
        rr = [r for r in rows if r["task"] == name]
        ax.scatter([r["G"] for r in rr], [r["g_s2"] for r in rr], marker=mk[name], s=55,
                   facecolor="none", edgecolor="#B0173A", label=name)
    xs = np.linspace(min(r["G"] for r in rows) - 0.05, max(r["G"] for r in rows) + 0.05, 200); xp = np.maximum(xs, 0)
    L, S = law["linear"], law["saturating"]
    ax.plot(xs, L["a"] + L["b"] * xp, "--", color="black", lw=0.9,
            label=rf"rectified linear $g \approx {L['a']:.2f} + {L['b']:.2f}[G]_+$ (r = {L['r_family']:.2f} / {L['r_heldout']:.2f} held-out)")
    ax.plot(xs, S["c"] * (1 - np.exp(-xp / S["G0"])), "-", color="black", lw=1.1,
            label=rf"rectified saturating $g \approx {S['c']:.2f}\,(1-e^{{-[G]_+/{S['G0']:.2f}}})$ (r = {S['r_family']:.2f} / {S['r_heldout']:.2f})")
    plt.colorbar(sc, ax=ax, label=r"$\theta$ (0 = Fourier, 1 = random basis)")
    ax.set_xlabel(r"information gap $G = I(T_{learned};Y) - I(T_{dict};Y)$  [bits]")
    ax.set_ylabel(r"gate $g_{s2}$ on the learned branch"); ax.grid(alpha=.3); ax.legend(fontsize=6.5, loc="lower right")
    ax.set_title("the classification gate pays for information")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "gap_law.png"), dpi=160); plt.close(fig)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "replot":
        rows = json.load(open(os.path.join(RES, "gap_law.json")))["rows"]
    else:
        rows = run()
    law = fit_law(rows); print("law:", law)
    plot(rows, law)
    json.dump(dict(rows=rows, law=law, thetas=THETAS, seeds=SEEDS), open(os.path.join(RES, "gap_law.json"), "w"), indent=1)
    print("done")
