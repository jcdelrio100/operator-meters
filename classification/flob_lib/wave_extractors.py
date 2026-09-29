"""
wave_extractors.py
Wavelet feature extractors for the compact-multiresolution study.
Two bases: orthogonal DWT (Daubechies) and CWT/scattering (Morlet).
Plus shared utilities: ANOVA F-score / energy selection and a linear readout.
"""
import numpy as np
import pywt

# ---------------- orthogonal DWT (Daubechies) ----------------
def dwt_coeffs(X, wavelet='db4', level=None):
    """Flat DWT coefficient vector per sample (real, multiresolution). (n, D)."""
    c0 = pywt.wavedec(X[0], wavelet, level=level)
    sizes = [len(c) for c in c0]
    out = np.zeros((X.shape[0], sum(sizes)), dtype=np.float64)
    for i in range(X.shape[0]):
        cs = pywt.wavedec(X[i], wavelet, level=level)
        out[i] = np.concatenate(cs)
    return out

def dwt_features(X, wavelet='db4', level=None):
    """For a linear readout: the raw DWT coefficients (signed)."""
    return dwt_coeffs(X, wavelet, level)

# ---------------- Morlet CWT / scattering ----------------
def _morlet_bank(N, freqs, nc=5):
    t = np.arange(N) - N // 2
    kers = []
    for f in freqs:
        s = nc * N / (2 * np.pi * f)
        psi = np.exp(2j * np.pi * f * t / N) * np.exp(-t**2 / (2 * s**2))
        kers.append(np.fft.fft(np.fft.ifftshift(psi), N))
    return np.array(kers)  # (F, N) in freq domain

def cwt_mag(X, freqs=(4, 6, 8, 12, 16, 20, 24), nc=5, pool=4):
    """|CWT| magnitude, time-pooled. Localized (not shift-invariant). (n, F*N/pool)."""
    N = X.shape[1]; K = _morlet_bank(N, freqs, nc); Xf = np.fft.fft(X, axis=1)
    feats = []
    for k in range(len(freqs)):
        C = np.fft.ifft(Xf * K[k][None, :], axis=1); mag = np.abs(C)
        p = mag[:, :N // pool * pool].reshape(X.shape[0], N // pool, pool).mean(2)
        feats.append(p)
    return np.concatenate(feats, 1)

def scattering1d(X, freqs=(4, 6, 8, 12, 16, 20, 24), nc=5):
    """First-order scattering: time-averaged modulus -> translation invariant. (n, F)."""
    N = X.shape[1]; K = _morlet_bank(N, freqs, nc); Xf = np.fft.fft(X, axis=1)
    feats = []
    for k in range(len(freqs)):
        C = np.fft.ifft(Xf * K[k][None, :], axis=1)
        feats.append(np.abs(C).mean(1, keepdims=True))  # S1 = mean_t |x*psi|
    return np.concatenate(feats, 1)

# ---------------- selection + readout (shared) ----------------
def anova_scores_1d(F, y, ncl):
    """F-score per column (atom) of feature matrix F (n, D)."""
    n, D = F.shape; sc = np.zeros(D)
    grand = F.mean(0)
    for c in range(ncl):
        pass
    ssb = np.zeros(D); ssw = np.zeros(D)
    for c in range(ncl):
        Fc = F[y == c]
        if len(Fc) == 0:
            continue
        ssb += len(Fc) * (Fc.mean(0) - grand) ** 2
        ssw += ((Fc - Fc.mean(0)) ** 2).sum(0)
    return (ssb / (ncl - 1)) / (ssw / (n - ncl) + 1e-9)

def energy_scores_1d(F):
    return np.abs(F).mean(0)

def softmax(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)

def linear_readout(Ftr, ytr, Fte, yte, ncl, epochs=400, lr=0.2, l2=1e-4):
    mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-9
    A, B = (Ftr - mu) / sd, (Fte - mu) / sd
    rng = np.random.default_rng(0)
    W = rng.normal(0, 0.01, (A.shape[1], ncl)); b = np.zeros(ncl)
    Y = np.zeros((len(ytr), ncl)); Y[np.arange(len(ytr)), ytr] = 1; n = len(ytr)
    for _ in range(epochs):
        p = softmax(A @ W + b); g = (p - Y) / n
        W -= lr * (A.T @ g + l2 * W); b -= lr * g.sum(0)
    return float((softmax(B @ W + b).argmax(1) == yte).mean())
