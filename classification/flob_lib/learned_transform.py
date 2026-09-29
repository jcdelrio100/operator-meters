"""
learned_transform.py — Transformada UNITARIA APRENDIDA desde cero.

No se elige ninguna base (ni Fourier, ni wavelet, ni FrFT): se parametriza una
matriz unitaria U generica y se APRENDE por gradiente la que mejor representa la
senal para la tarea. U se mantiene unitaria en todo momento porque se genera como
la exponencial de una matriz anti-hermitica:

    U = exp(C - C^H),   C compleja libre  ->  U U^H = I  siempre.

Rasgo = |U x| (magnitud de los coeficientes, invariante a fase, como Fourier pero
con la base aprendida). Un termino L1 opcional empuja a 'K componentes limpios'
(representacion compacta). Es el caso general del grupo unitario del que la FrFT
era una geodesica de 1 parametro.
"""
import numpy as np
import torch
import torch.nn as nn


class LearnableUnitary(nn.Module):
    def __init__(self, N, scale=0.05, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.Cr = nn.Parameter(torch.randn(N, N, generator=g, dtype=torch.float64) * scale)
        self.Ci = nn.Parameter(torch.randn(N, N, generator=g, dtype=torch.float64) * scale)

    def matrix(self):
        C = torch.complex(self.Cr, self.Ci)
        H = C - C.conj().t()                 # anti-hermitica
        return torch.matrix_exp(H)           # unitaria

    def coeffs(self, x):
        U = self.matrix()
        return x.to(torch.complex128) @ U.t()   # (B,N) = U x por fila


class StructuredUnitary(LearnableUnitary):
    """Unitaria ESTRUCTURADA: el generador anti-hermitico se restringe a una BANDA
    (|i-j| <= bandwidth), asi la base se describe con << N^2 parametros. exp() sigue
    siendo unitaria aunque el generador sea bandeado."""
    def __init__(self, N, bandwidth, scale=0.05, seed=0):
        super().__init__(N, scale, seed)
        self.bandwidth = int(bandwidth)
        m = (np.abs(np.subtract.outer(np.arange(N), np.arange(N))) <= bandwidth)
        self.register_buffer("mask", torch.tensor(m.astype(np.float64)))

    def matrix(self):
        C = torch.complex(self.Cr * self.mask, self.Ci * self.mask)
        H = C - C.conj().t()
        return torch.matrix_exp(H)

    def n_free(self):
        N = self.Cr.shape[0]; b = self.bandwidth
        return N + 2 * sum(N - d for d in range(1, b + 1))   # DOF reales del generador


class UnitaryClassifier(nn.Module):
    def __init__(self, N, nc, scale=0.05, seed=0, bandwidth=None):
        super().__init__()
        self.U = (LearnableUnitary(N, scale, seed) if bandwidth is None
                  else StructuredUnitary(N, bandwidth, scale, seed))
        self.bn = nn.BatchNorm1d(N, dtype=torch.float64)
        self.head = nn.Linear(N, nc).double()

    def features(self, x):
        return self.U.coeffs(x).abs()        # |U x|

    def forward(self, x):
        return self.head(self.bn(self.features(x)))


def _tv(U):
    """Variacion total de los atomos (filas de U) a lo largo del tiempo: prior de
    SUAVIDAD -> atomos oscilatorios limpios en vez de ruido."""
    d = U[:, 1:] - U[:, :-1]
    return d.abs().mean()


def _concentration(feat):
    """||z||_1/||z||_2 por muestra (bajo = disperso/concentrado). Invariante a escala."""
    l1 = feat.sum(1); l2 = torch.sqrt((feat ** 2).sum(1)) + 1e-9
    return (l1 / l2).mean()


def train(model, Xtr, ytr, Xte, yte, epochs=300, lr=0.02, head_lr=0.01,
          l1=0.0, tv=0.0, sparse=0.0):
    Xtr = torch.tensor(Xtr); ytr = torch.tensor(ytr).long()
    Xte = torch.tensor(Xte); yte = torch.tensor(yte).long()
    U_params = list(model.U.parameters())
    rest = list(model.bn.parameters()) + list(model.head.parameters())
    opt = torch.optim.Adam([{"params": U_params, "lr": lr},
                            {"params": rest, "lr": head_lr}], weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()
    best = 0.0
    for _ in range(epochs):
        model.train(); opt.zero_grad()
        feat = model.features(Xtr)
        out = model.head(model.bn(feat))
        loss = lossf(out, ytr)
        if sparse > 0:                        # HIBRIDO: sparsity (limpieza) + discriminacion
            loss = loss + sparse * _concentration(feat)
        if l1 > 0:                            # compacidad: pocas componentes activas
            fn = feat / (feat.sum(1, keepdim=True) + 1e-9)
            loss = loss + l1 * fn.abs().mean()
        if tv > 0:                            # suavidad de los atomos (limpieza)
            loss = loss + tv * _tv(model.U.matrix())
        loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            best = max(best, (model(Xte).argmax(1) == yte).float().mean().item())
    return best


def train_sparsifying(uni, X, epochs=300, lr=0.02, tv=0.0):
    """NO SUPERVISADO: aprende U unitaria que hace la senal DISPERSA (K coeficientes
    limpios) minimizando ||Ux||_1/||Ux||_2 (invariante de escala; U unitaria conserva
    la energia, asi que esto concentra). Sin etiquetas. Modelo clasico de sparsifying
    transform learning sobre la variedad unitaria."""
    Xt = torch.tensor(X)
    opt = torch.optim.Adam(uni.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        z = uni.coeffs(Xt).abs()
        l1 = z.sum(1); l2 = torch.sqrt((z ** 2).sum(1)) + 1e-9
        loss = (l1 / l2).mean()
        if tv > 0:
            loss = loss + tv * _tv(uni.matrix())
        loss.backward(); opt.step()
    return uni


def atom_flatness(U, y_scores_idx):
    """Planitud espectral media (0=tono puro/limpio, 1=ruido blanco) de los atomos
    indicados. Metrica de 'limpieza' de la base aprendida."""
    import numpy as np
    fl = []
    for i in y_scores_idx:
        s = np.abs(np.fft.rfft(U[i].real)) ** 2 + 1e-12
        fl.append(np.exp(np.log(s).mean()) / s.mean())
    return float(np.mean(fl))
