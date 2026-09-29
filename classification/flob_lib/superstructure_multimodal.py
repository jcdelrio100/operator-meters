"""
Multimodal break-even: Superstructure 1 (pooled dictionary) vs Superstructure 2
(learned transform), same GatedResidual head (article 1), across complex modalities
to locate where the learned transform starts to win.

Usage:
  python superstructure_multimodal.py run  <task1;task2;...>   # append results to JSON
  python superstructure_multimodal.py plot                     # draw English charts

S2 learns a unitary on the flattened input (PCA-reduced to <=128 dims only when the
flattened dimension is larger, for compute; noted as a scalability step, not a
predefined transform). Charts in English.
"""
import sys, os, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(__file__)
for p in (".", "../tasks", "../src"):
    sys.path.insert(0, os.path.join(HERE, p))
import wave_extractors as W
from learned_transform import UnitaryClassifier, train as train_uni
import classic_tasks

JSON = os.path.join(HERE, "multimodal_results.json")
K, SEEDS, HEAD_EP, STAGE1_EP, DMAX = 16, 2, 300, 150, 128
N = 128


# ---------------- generators (from article 1) ----------------
def loc(n, seed, ncl=4, noise=0.25):
    rng = np.random.default_rng(seed); t = np.arange(N); X = np.zeros((n, N)); y = rng.integers(0, ncl, n)
    info = [13, 19, 27, 37]; sig = 7.0
    for i in range(n):
        c = y[i]; cd = rng.uniform(N*0.2, N*0.8); X[i] += 4*np.sin(2*np.pi*5*t/N)*np.exp(-((t-cd)**2)/(2*sig**2))
        ci = rng.uniform(N*0.2, N*0.8); X[i] += 0.8*np.sin(2*np.pi*info[c]*t/N)*np.exp(-((t-ci)**2)/(2*sig**2)); X[i] += rng.normal(0, noise, N)
    return X, y, ncl
def glob(n, seed, ncl=4, noise=0.7):
    rng = np.random.default_rng(seed); t = np.arange(N); X = np.zeros((n, N)); y = rng.integers(0, ncl, n)
    bands = [[5,6,7],[17,18,19],[30,31,32],[44,45,46]]
    for i in range(n):
        for f in bands[y[i]]: X[i] += np.sin(2*np.pi*f*t/N + rng.uniform(0, 6.28))
        X[i] += rng.normal(0, noise, N)
    return X, y, ncl
def texture(n, seed, S=28, ncl=4, noise=0.5):
    rng = np.random.default_rng(seed); X = np.zeros((n, S, S)); y = rng.integers(0, ncl, n)
    yy, xx = np.meshgrid(np.arange(S), np.arange(S), indexing='ij'); angs = [0,45,90,135]
    for i in range(n):
        a = np.deg2rad(angs[y[i]]); f = rng.uniform(0.18,0.30); ph = rng.uniform(0,6.28); proj = xx*np.cos(a)+yy*np.sin(a)
        X[i] = np.sin(2*np.pi*f*proj+ph) + rng.normal(0, noise, (S,S))
    return X, y, ncl
def seqt(n, seed, L=60, V=24, ncl=4):
    rng = np.random.default_rng(seed); base = np.full((ncl, V), 1.0); blk = V//ncl
    for c in range(ncl): base[c, c*blk:(c+1)*blk] = 8.0
    base = base/base.sum(1, keepdims=True); X = np.zeros((n, L, V)); y = np.zeros(n, int)
    for i in range(n):
        c = i % ncl; toks = rng.choice(V, size=L, p=base[c])
        for tt, tok in enumerate(toks): X[i, tt, tok] = 1.0
        y[i] = c
    idx = rng.permutation(n); return X[idx], y[idx], ncl
def vid(n, seed, T=12, Hh=16, Ww=16, ncl=4):
    rng = np.random.default_rng(seed); X = np.zeros((n, T, Hh, Ww)); y = np.zeros(n, int)
    dirs = [(1,0.5),(-1,0.5),(0.5,1),(0.5,-1)]; yy, xx = np.meshgrid(np.arange(Hh), np.arange(Ww), indexing='ij')
    for i in range(n):
        c = i % ncl; vy, vx = dirs[c]; y0, x0 = rng.uniform(3, Hh-3), rng.uniform(3, Ww-3)
        for tt in range(T):
            cy = (y0+vy*tt) % Hh; cx = (x0+vx*tt) % Ww; X[i, tt] = np.exp(-((yy-cy)**2+(xx-cx)**2)/4)
        X[i] += rng.normal(0, 0.2, (T, Hh, Ww)); y[i] = c
    idx = rng.permutation(n); return X[idx], y[idx], ncl
def digits_split(seed):
    from sklearn.datasets import load_digits
    d = load_digits(); X = d.data.astype('float64').reshape(-1,8,8); y = d.target.astype('int64')
    X = (X-X.mean())/(X.std()+1e-6); rng = np.random.default_rng(seed); idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]; return X[:1200], y[:1200], X[1200:], y[1200:], 10
def xor_hidden(n, seed, ncl=2, noise=0.35):
    """XOR of two coordinates hidden in a random orthogonal basis: non-linear AND
    non-classical. Dictionary cannot expose the coords; a learned rotation can, and
    the non-linear head then solves the XOR."""
    fix = np.random.default_rng(999); Q, _ = np.linalg.qr(fix.normal(size=(N, N))); c0, c1 = fix.permutation(N)[:2]
    rng = np.random.default_rng(seed); X, y = [], []
    for _ in range(n):
        z = noise*rng.standard_normal(N); a = rng.choice([-1.0,1.0]); b = rng.choice([-1.0,1.0])
        z[c0] += 2*a; z[c1] += 2*b; X.append(Q.T@z); y.append(int(a*b < 0))
    X = np.array(X); X = (X-X.mean(1,keepdims=True))/(X.std(1,keepdims=True)+1e-9)
    return X, np.array(y), ncl


# ---------------- S1 dictionaries (modality-appropriate) ----------------
def f_mp(X):
    Xf = np.fft.rfft(X, axis=1); m = np.abs(Xf); p = np.angle(Xf)
    return np.concatenate([m, m*np.cos(p), m*np.sin(p)], 1)
reps1d = {"Fourier": f_mp, "DWT": lambda X: W.dwt_features(X, "db4"), "CWT": W.cwt_mag}
img_reps = {"Fourier2D": lambda X: np.abs(np.fft.rfft2(X, axes=(1,2))).reshape(len(X),-1),
            "pixels": lambda X: X.reshape(len(X),-1)}
seq_reps = {"Fourier-tok": lambda X: (lambda Xf: np.concatenate([np.abs(Xf).reshape(len(X),-1),
            (np.abs(Xf)*np.cos(np.angle(Xf))).reshape(len(X),-1), (np.abs(Xf)*np.sin(np.angle(Xf))).reshape(len(X),-1)],1))(np.fft.rfft(X,axis=1)),
            "bag-of-words": lambda X: X.mean(1)}
vid_reps = {"Fourier3D": lambda X: np.abs(np.fft.rfftn(X, axes=(1,2,3))).reshape(len(X),-1),
            "per-frame": lambda X: X.mean(1).reshape(len(X),-1)}


def load(name, seed):
    if name == "1D global (Fourier)":         Xtr,ytr,nc = glob(900,seed); Xte,yte,_ = glob(500,seed+100); return Xtr,ytr,Xte,yte,nc,reps1d
    if name == "1D localized (wavelet)":      Xtr,ytr,nc = loc(900,seed);  Xte,yte,_ = loc(500,seed+100);  return Xtr,ytr,Xte,yte,nc,reps1d
    if name == "image digits (real)":         Xtr,ytr,Xte,yte,nc = digits_split(seed); return Xtr,ytr,Xte,yte,nc,img_reps
    if name == "texture (28x28)":             Xtr,ytr,nc = texture(900,seed); Xte,yte,_ = texture(500,seed+100); return Xtr,ytr,Xte,yte,nc,img_reps
    if name == "text (topic)":                Xtr,ytr,nc = seqt(1600,seed); Xte,yte,_ = seqt(800,seed+100); return Xtr,ytr,Xte,yte,nc,seq_reps
    if name == "video (motion)":              Xtr,ytr,nc = vid(500,seed);  Xte,yte,_ = vid(400,seed+100);  return Xtr,ytr,Xte,yte,nc,vid_reps
    if name == "hidden basis (rotated)":      Xtr,ytr,Xte,yte,nc = classic_tasks.hidden_basis(seed=seed); return Xtr,ytr,Xte,yte,nc,reps1d
    if name == "XOR in hidden basis":         Xtr,ytr,nc = xor_hidden(1600,seed); Xte,yte,_ = xor_hidden(800,seed+100); return Xtr,ytr,Xte,yte,nc,reps1d


# ---------------- shared head ----------------
class GatedResidual(nn.Module):
    def __init__(self, K, C, blocks=2, h=32):
        super().__init__(); self.lin = nn.Linear(K, C)
        self.blocks = nn.ModuleList([nn.Sequential(nn.Linear(K,h), nn.ReLU(), nn.Linear(h,C)) for _ in range(blocks)])
        self.gates = nn.Parameter(torch.ones(blocks)); self.K, self.C, self.h = K, C, h
    def forward(self, x):
        o = self.lin(x)
        for g, b in zip(self.gates, self.blocks): o = o + g*b(x)
        return o
    def lin_params(self): return self.K*self.C+self.C
    def block_params(self): return self.K*self.h+self.h+self.h*self.C+self.C

def fit_head(A, ytr, B, yte, nc, lam=0.25, ep=HEAD_EP, lr=5e-3, eps=0.15):
    A = torch.tensor(A, dtype=torch.float32); B = torch.tensor(B, dtype=torch.float32)
    yt = torch.tensor(ytr); ye = torch.tensor(yte); net = GatedResidual(A.shape[1], nc)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4); curve = []
    for e in range(ep):
        net.train(); opt.zero_grad(); (Fn.cross_entropy(net(A), yt)+lam*net.gates.abs().sum()).backward(); opt.step()
        if e % 10 == 0 or e == ep-1:
            net.eval()
            with torch.no_grad(): curve.append((e, (net(B).argmax(1)==ye).float().mean().item()))
    net.eval()
    with torch.no_grad():
        acc = (net(B).argmax(1)==ye).float().mean().item(); gates = net.gates.abs().tolist()
    eff = net.lin_params() + sum(net.block_params() for g in gates if g > eps)
    return acc, eff, curve


def fscore(F, y, nc):
    n, D = F.shape; grand = F.mean(0); ssb = np.zeros(D); ssw = np.zeros(D)
    for c in range(nc):
        Fc = F[y==c]; ssb += len(Fc)*(Fc.mean(0)-grand)**2; ssw += ((Fc-Fc.mean(0))**2).sum(0)
    return (ssb/(nc-1))/(ssw/(n-nc)+1e-9)
def std_select(P1, ytr, P2, nc):
    idx = np.argsort(fscore(P1, ytr, nc))[::-1][:min(K, P1.shape[1])]
    A, B = P1[:, idx], P2[:, idx]; mu, sd = A.mean(0), A.std(0)+1e-9
    return (A-mu)/sd, (B-mu)/sd


def s1_feats(Xtr, ytr, Xte, yte, nc, reps):
    P1 = np.concatenate([f(Xtr) for f in reps.values()], 1); P2 = np.concatenate([f(Xte) for f in reps.values()], 1)
    return (*std_select(P1, ytr, P2, nc), 0)

def s2_feats(Xtr, ytr, Xte, yte, nc, seed):
    F1 = Xtr.reshape(len(Xtr), -1).astype(np.float64); F2 = Xte.reshape(len(Xte), -1).astype(np.float64)
    pca = 0
    if F1.shape[1] > DMAX:                       # scalability reduction (data-driven)
        mu = F1.mean(0); _, _, Vt = np.linalg.svd(F1-mu, full_matrices=False); Vt = Vt[:DMAX]
        F1 = (F1-mu)@Vt.T; F2 = (F2-mu)@Vt.T; pca = 1
    d = F1.shape[1]
    sd = F1.std(0)+1e-9; F1 = F1/sd; F2 = F2/sd
    torch.manual_seed(seed); m = UnitaryClassifier(d, nc, seed=seed)
    train_uni(m, F1, ytr, F2, yte, epochs=STAGE1_EP, sparse=4.0)
    with torch.no_grad():
        Z1 = m.features(torch.tensor(F1)).numpy(); Z2 = m.features(torch.tensor(F2)).numpy()
    A, B = std_select(Z1, ytr, Z2, nc)
    return A, B, d*d, pca


def run(tasks):
    out = json.load(open(JSON)) if os.path.exists(JSON) else {}
    for name in tasks:
        r = {"S1": {"acc": [], "tot": []}, "S2": {"acc": [], "tot": []}, "curve": {}}
        for s in range(SEEDS):
            Xtr, ytr, Xte, yte, nc, reps = load(name, s)
            A, B, tp = s1_feats(Xtr, ytr, Xte, yte, nc, reps)
            a1, e1, c1 = fit_head(A, ytr, B, yte, nc)
            r["S1"]["acc"].append(a1); r["S1"]["tot"].append(tp+e1)
            A, B, tp, pca = s2_feats(Xtr, ytr, Xte, yte, nc, s)
            a2, e2, c2 = fit_head(A, ytr, B, yte, nc)
            r["S2"]["acc"].append(a2); r["S2"]["tot"].append(tp+e2)
            if s == 0: r["curve"] = {"S1": c1, "S2": c2, "pca": pca}
        out[name] = r
        print(f"{name:26s} S1={np.mean(r['S1']['acc']):.3f}  S2={np.mean(r['S2']['acc']):.3f}  "
              f"| par S1={int(np.median(r['S1']['tot']))} S2={int(np.median(r['S2']['tot']))}")
        json.dump(out, open(JSON, "w"), indent=1)


def plot():
    out = json.load(open(JSON))
    names = list(out)
    # sort by (S2 - S1) so the break-even ordering is visible
    names.sort(key=lambda t: np.mean(out[t]["S2"]["acc"]) - np.mean(out[t]["S1"]["acc"]))
    x = np.arange(len(names)); w = 0.38
    s1 = [np.mean(out[t]["S1"]["acc"]) for t in names]; s1e = [np.std(out[t]["S1"]["acc"]) for t in names]
    s2 = [np.mean(out[t]["S2"]["acc"]) for t in names]; s2e = [np.std(out[t]["S2"]["acc"]) for t in names]
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.bar(x-w/2, s1, w, yerr=s1e, capsize=3, color="#555555", label="S1 Pooled dictionary")
    ax.bar(x+w/2, s2, w, yerr=s2e, capsize=3, color="#d62728", label="S2 Learned transform")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Test accuracy"); ax.set_ylim(0.5, 1.03); ax.axhline(0.5, color="k", lw=0.6)
    ax.set_title("Break-even: Pooled Dictionary vs Learned Transform across modalities (y-axis starts at 0.5)")
    ax.legend(); ax.grid(axis="y", alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(HERE, "multimodal_breakeven.png"), dpi=130)
    # params
    fig, ax = plt.subplots(figsize=(11, 4.2))
    p1 = [np.median(out[t]["S1"]["tot"]) for t in names]; p2 = [np.median(out[t]["S2"]["tot"]) for t in names]
    ax.bar(x-w/2, p1, w, color="#555555", label="S1 Pooled dictionary")
    ax.bar(x+w/2, p2, w, color="#d62728", label="S2 Learned transform")
    ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Total trainable parameters (log)"); ax.set_title("Total parameters: transform + head")
    ax.legend(); ax.grid(axis="y", alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(HERE, "multimodal_params.png"), dpi=130)
    print("saved multimodal_breakeven.png, multimodal_params.png")
    print("\nSummary (sorted by S2-S1):")
    for t in names:
        print(f"  {t:26s} S1={np.mean(out[t]['S1']['acc']):.3f}  S2={np.mean(out[t]['S2']['acc']):.3f}  "
              f"delta={np.mean(out[t]['S2']['acc'])-np.mean(out[t]['S1']['acc']):+.3f}")


if __name__ == "__main__":
    if sys.argv[1] == "run": run(sys.argv[2].split(";"))
    elif sys.argv[1] == "plot": plot()
