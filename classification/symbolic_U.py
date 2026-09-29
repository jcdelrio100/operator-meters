"""Reading the learned unitary U symbolically — the classification counterpart of WP3.

Three transforms on the '1D global (Fourier)' task, N = 128:
  (a) U warm-started at the DFT and left untrained            (sanity: must read as the DFT)
  (b) U trained with a compression (sparsity) prior + CE      (compression side)
  (c) U trained by cross-entropy alone, from a random start   (classification-driven)
Reading = fit the closed form  U[j,k] = exp(-i a j k) / sqrt(N)  after gauge canonicalisation
(rows sorted by dominant frequency; row phase fixed) — the score is |<U, model>| / (||U|| ||model||)
maximised over a; the DFT has a = 2 pi / N and score 1.  A second, weaker reading: the
fraction of rows that are pure sinusoids (novelty = 1 - max_j |<row, DFT_j>|).

Outputs: figures/symbolic_U.png, results/symbolic_U.json
"""
import os, sys, json
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(HERE, "flob_lib"))
import superstructure_multimodal as MM
from learned_transform import UnitaryClassifier, train as train_uni
from warmstart import dft_matrix, warm_start_flat
FIG = os.path.join(ROOT, "figures"); RES = os.path.join(ROOT, "results")
torch.set_num_threads(2)
N = 128


def canonicalise(U):
    """Sort rows by dominant |DFT| frequency and fix each row's phase at its dominant bin."""
    F = dft_matrix(N)
    C = U @ F.conj().T                     # row j of U expressed in the DFT basis
    dom = np.abs(C).argmax(1)
    order = np.argsort(dom, kind="stable")
    U = U[order]; C = C[order]; dom = dom[order]
    ph = np.exp(-1j * np.angle(C[np.arange(N), dom]))
    return U * ph[:, None], dom


def closed_form_fit(U):
    """max_a |<M_a, U>| / (||M_a|| ||U||) with M_a[j,k] = exp(-i a j k)/sqrt(N); vectorised grid + refinement."""
    from scipy.optimize import minimize_scalar
    j = np.arange(N); jk = np.outer(j, j).ravel().astype(float); u = U.ravel(); nu = np.linalg.norm(u)
    def score_vec(avec):
        out = np.empty(len(avec))
        for i0 in range(0, len(avec), 256):
            a = avec[i0:i0 + 256]
            M = np.exp(1j * a[:, None] * jk[None, :]) / np.sqrt(N)          # conj(M_a) rows
            out[i0:i0 + 256] = np.abs(M @ u) / (np.linalg.norm(M, axis=1) * nu)
        return out
    grid = np.linspace(0, 2 * np.pi, 20001)[1:]
    sc = score_vec(grid); i = int(sc.argmax())
    lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
    r = minimize_scalar(lambda a: -score_vec(np.array([a]))[0], bounds=(lo, hi), method="bounded", options=dict(xatol=1e-11))
    return (float(-r.fun), float(r.x)) if -r.fun > sc[i] else (float(sc[i]), float(grid[i]))


def novelty(U):
    F = dft_matrix(N)
    return 1 - np.abs(U @ F.conj().T).max(1)


def get_U(kind, seed=0):
    Xtr, ytr, Xte, yte, nc, _ = MM.load("1D global (Fourier)", seed)
    F1 = Xtr.astype(np.float64); F2 = Xte.astype(np.float64)
    torch.manual_seed(seed); m = UnitaryClassifier(N, nc, seed=seed)
    if kind == "dft":
        warm_start_flat(m.U, dft_matrix(N))
    elif kind == "sparse":
        warm_start_flat(m.U, dft_matrix(N)); train_uni(m, F1, ytr, F2, yte, epochs=150, sparse=4.0)
    elif kind == "ce":
        train_uni(m, F1, ytr, F2, yte, epochs=150)
    with torch.no_grad():
        return m.U.matrix().numpy()


def run():
    out = {}
    for kind, label in [("dft", "DFT, untrained"), ("sparse", "compression prior + CE"), ("ce", "CE alone, random start")]:
        U, dom = canonicalise(get_U(kind))
        s, a = closed_form_fit(U); nv = novelty(U)
        out[kind] = dict(label=label, fit=s, a=a, a_over_2pi_N=a / (2 * np.pi / N),
                         novelty_median=float(np.median(nv)), frac_pure=float((nv < 0.1).mean()), U_real=np.real(U).tolist())
        print(f"{label:26s} closed-form fit={s:.3f}  a/(2π/N)={a/(2*np.pi/N):.3f}  median novelty={np.median(nv):.3f}  "
              f"pure-sinusoid rows={100*(nv<0.1).mean():.0f}%", flush=True)
    return out


def plot(out):
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.8))
    for i, kind in enumerate(["dft", "sparse", "ce"]):
        o = out[kind]
        ax[i].imshow(np.array(o["U_real"]), cmap="RdBu", vmin=-0.12, vmax=0.12, aspect="auto")
        ax[i].set_title(f"{o['label']}\nfit to $e^{{-iajk}}/\\sqrt{{N}}$: {o['fit']:.2f},  "
                        f"$a/(2\\pi/N)$ = {o['a_over_2pi_N']:.2f}", fontsize=9)
        ax[i].set_xlabel("k"); ax[i].set_ylabel("row j (canonicalised)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "symbolic_U.png"), dpi=150); plt.close(fig)


if __name__ == "__main__":
    out = run(); plot(out)
    json.dump({k: {kk: vv for kk, vv in v.items() if kk != "U_real"} for k, v in out.items()},
              open(os.path.join(RES, "symbolic_U.json"), "w"), indent=1)
    print("done")
