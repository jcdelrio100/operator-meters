"""
Warm-start the learned transform FROM a fixed transform.

U = exp(C - C^H). To make exp(C-C^H) = U0 (a given unitary) we need the skew-Hermitian
generator G = log(U0); then setting C = G/2 gives C - C^H = G (since G^H = -G). So the
learned transform *starts* exactly at U0 (e.g. the DFT) and can only improve from there
under task training -> learned >= fixed by construction.
"""
import numpy as np, torch


def dft_matrix(n):
    j = np.arange(n)
    return np.exp(-2j * np.pi * np.outer(j, j) / n) / np.sqrt(n)


def skew_log(U0):
    """G skew-Hermitian with exp(G) = U0. Principal matrix logarithm (robust to the
    DFT's degenerate spectrum, where an eig-based log fails)."""
    from scipy.linalg import logm
    G = logm(U0.astype(np.complex128))
    return 0.5 * (G - G.conj().T)          # project to skew-Hermitian (kills numerical drift)


def _set_generator(Cr_param, Ci_param, U0):
    G = skew_log(U0) / 2.0
    with torch.no_grad():
        Cr_param.copy_(torch.tensor(G.real, dtype=torch.float64))
        Ci_param.copy_(torch.tensor(G.imag, dtype=torch.float64))


def warm_start_flat(uni, U0):
    """uni : LearnableUnitary (single Cr/Ci)."""
    _set_generator(uni.Cr, uni.Ci, U0)


def warm_start_separable(uni, U0_list):
    """uni : MultilinearUnitary (Cr/Ci ParameterLists), one U0 per axis."""
    for k, U0 in enumerate(U0_list):
        _set_generator(uni.Cr[k], uni.Ci[k], U0)


def train_ws(model, Xtr, ytr, Xte, yte, epochs=200, lr=0.02, head_lr=0.01):
    """End-to-end CE, NO weight decay on the transform (so it is not pulled back to 0
    and away from the warm-start)."""
    import torch.nn as nn
    Xt = torch.tensor(Xtr); yt = torch.tensor(ytr).long(); Xe = torch.tensor(Xte); ye = torch.tensor(yte).long()
    tr = [p for n, p in model.named_parameters() if n.startswith("U.")]
    rest = [p for n, p in model.named_parameters() if not n.startswith("U.")]
    opt = torch.optim.Adam([{"params": tr, "lr": lr, "weight_decay": 0.0},
                            {"params": rest, "lr": head_lr, "weight_decay": 1e-4}])
    lf = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train(); opt.zero_grad(); lf(model(Xt), yt).backward(); opt.step()
    model.eval()
    with torch.no_grad():
        return (model(Xe).argmax(1) == ye).float().mean().item()
