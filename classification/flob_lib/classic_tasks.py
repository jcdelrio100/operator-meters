"""
classic_tasks.py
Tareas de contraste (no chirp) para el estudio integrado: verifican que el
auto-enrutado del diccionario elige la transformada 'correcta' por regimen.
  - narrowband_tones: estructura GLOBAL de banda estrecha -> Fourier gana, DWT falla.
  - localized_bump:    estructura LOCAL (posicion) -> wavelet/CWT ganan.
Formato: train_fn() -> (Xtr, ytr, Xte, yte, nc).
"""
import numpy as np

N_DEFAULT = 128


def _normalize(X):
    X = X - X.mean(1, keepdims=True)
    return X / (X.std(1, keepdims=True) + 1e-9)


def _split(X, y, seed, test_frac=0.3):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X)); ntest = int(len(X) * test_frac)
    te, tr = idx[:ntest], idx[ntest:]
    return X[tr], y[tr], X[te], y[te], int(np.unique(y).size)


def narrowband_tones(N=N_DEFAULT, n_per_class=300, tones=(20.0, 23.0),
                     noise=0.4, seed=0):
    """Tonos globales de frecuencias muy proximas (misma banda de octava).
    Fourier los resuelve; la DWT diadica no."""
    rng = np.random.default_rng(seed); t = np.arange(N)
    X, y = [], []
    for cls, f in enumerate(tones):
        for _ in range(n_per_class):
            ff = f + rng.uniform(-0.3, 0.3)
            ph = rng.uniform(0, 2 * np.pi); amp = rng.uniform(0.8, 1.2)
            s = amp * np.cos(2 * np.pi * ff * t / N + ph) + noise * rng.standard_normal(N)
            X.append(s); y.append(cls)
    return _split(_normalize(np.array(X)), np.array(y), seed + 1)


def localized_bump(N=N_DEFAULT, n_per_class=300, carrier=16.0, width=9.0,
                   noise=0.3, seed=0):
    """Paquete de onda localizado; la CLASE = posicion (primera/segunda mitad).
    La posicion es local -> wavelet/CWT la capturan; la magnitud de Fourier no."""
    rng = np.random.default_rng(seed); t = np.arange(N)
    X, y = [], []
    halves = [(0.15, 0.40), (0.60, 0.85)]
    for cls, (lo, hi) in enumerate(halves):
        for _ in range(n_per_class):
            pos = rng.uniform(lo, hi) * N
            ph = rng.uniform(0, 2 * np.pi); amp = rng.uniform(0.8, 1.2)
            env = np.exp(-((t - pos) ** 2) / (2 * width ** 2))
            s = amp * env * np.cos(2 * np.pi * carrier * t / N + ph)
            s = s + noise * rng.standard_normal(N)
            X.append(s); y.append(cls)
    return _split(_normalize(np.array(X)), np.array(y), seed + 1)


def hidden_basis(N=N_DEFAULT, n_per_class=250, nc=4, active=3, noise=0.35, seed=0):
    """La clase se separa por unas pocas coordenadas activas en una base ORTOGONAL
    ALEATORIA fija Q, DESCONOCIDA (no es Fourier ni wavelet). En el dominio de la
    senal (y en Fourier/wavelet) es una mezcla densa; solo una transformada que
    redescubra Q expone los K componentes limpios discriminativos."""
    fix = np.random.default_rng(12345)                 # Q y coords FIJOS (base oculta)
    Q, _ = np.linalg.qr(fix.normal(size=(N, N)))
    coords = fix.permutation(N)[:nc * active].reshape(nc, active)
    rng = np.random.default_rng(seed); X, y = [], []
    for c in range(nc):
        for _ in range(n_per_class):
            z = noise * rng.standard_normal(N)
            z[coords[c]] += rng.uniform(1.5, 2.5, active) * rng.choice([-1.0, 1.0], active)
            X.append(Q.T @ z); y.append(c)
    X = np.array(X); X = (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-9)
    return _split(X, np.array(y), seed + 1)


TASKS = {
    "narrowband_tones": narrowband_tones,
    "localized_bump": localized_bump,
    "hidden_basis": hidden_basis,
}

if __name__ == "__main__":
    for name, fn in TASKS.items():
        Xtr, ytr, Xte, yte, nc = fn()
        print(f"{name:18s} Xtr={Xtr.shape} nc={nc} balance={np.bincount(ytr)}")
