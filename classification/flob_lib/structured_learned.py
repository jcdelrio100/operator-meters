"""
Structure-aware learned transform: a SEPARABLE (multilinear) unitary, one learned
unitary per tensor axis, applied as mode products. It is the learned analogue of
2-D/3-D Fourier (which is a separable per-axis DFT) and respects the 2-D/3-D/
sequential structure that a single flattened unitary ignores.

For an input of shape (d1,...,dm): U = U_1 (x) ... (x) U_m, each U_k = exp(C_k - C_k^H)
unitary and learned. Parameters: sum_k d_k^2  (<< (prod d_k)^2 of the full unitary),
and mode products are cheap. For a 1-D input this reduces to the plain learned unitary.
"""
import numpy as np, torch, torch.nn as nn


class MultilinearUnitary(nn.Module):
    def __init__(self, shape, scale=0.05, seed=0):
        super().__init__()
        self.shape = tuple(int(d) for d in shape)
        g = torch.Generator().manual_seed(seed)
        self.Cr = nn.ParameterList(); self.Ci = nn.ParameterList()
        for d in self.shape:
            self.Cr.append(nn.Parameter(torch.randn(d, d, generator=g, dtype=torch.float64) * scale))
            self.Ci.append(nn.Parameter(torch.randn(d, d, generator=g, dtype=torch.float64) * scale))

    def mats(self):
        Us = []
        for cr, ci in zip(self.Cr, self.Ci):
            C = torch.complex(cr, ci); Us.append(torch.matrix_exp(C - C.conj().t()))
        return Us

    @staticmethod
    def _mode(Xc, U, axis):
        Xm = Xc.movedim(axis, -1); sh = Xm.shape
        Y = Xm.reshape(-1, sh[-1]) @ U.transpose(0, 1)      # (U x)_i = sum_j U_ij x_j
        return Y.reshape(sh).movedim(-1, axis)

    def coeffs(self, X):
        Xc = X.to(torch.complex128)
        for ax, U in enumerate(self.mats()):
            Xc = self._mode(Xc, U, ax + 1)
        return Xc

    def features(self, X):
        return self.coeffs(X).abs().reshape(X.shape[0], -1)

    def n_free(self):
        return int(sum(d * d for d in self.shape))


class MultilinearClassifier(nn.Module):
    def __init__(self, shape, nc, seed=0):
        super().__init__()
        self.U = MultilinearUnitary(shape, seed=seed)
        d = int(np.prod(shape))
        self.bn = nn.BatchNorm1d(d, dtype=torch.float64)
        self.head = nn.Linear(d, nc).double()

    def features(self, X): return self.U.features(X)
    def forward(self, X): return self.head(self.bn(self.features(X)))
