"""Shared numerics for the physics experiments (1-D, periodic, N = 64).

Two controlled families:
  * heterogeneous diffusion  u_t = d/dx( a(x) du/dx ),  a = 1 + beta * f(x)
    discretized as L = -G^T diag(a_f) G  (G: forward difference, a_f on faces),
    one-step propagator P = expm(L dt).
  * viscous Burgers          u_t = nu u_xx - eps u u_x
    integrated with RK4 and 2/3 dealiasing (eps = 0 is the heat equation).

Diagonality metric (paper Eq. 1):  D(B, P) = ||offdiag(B P B^T)||_F / ||B P B^T||_F
for a basis B with orthonormal rows.  D = 0 <=> a per-mode (diagonal) propagator in B
reproduces the dynamics exactly.
"""
import numpy as np
from scipy.linalg import expm, eigh

N = 64
LX = 2 * np.pi
DX = LX / N
X = np.arange(N) * DX            # cell centres
XF = X + DX / 2                  # cell faces (between i and i+1)


# ---------------------------------------------------------------- operators
def forward_diff():
    G = np.zeros((N, N))
    for i in range(N):
        G[i, i] = -1.0 / DX
        G[i, (i + 1) % N] = 1.0 / DX
    return G


G = forward_diff()


def L_of_coeff(a_faces):
    """Divergence-form diffusion operator for a coefficient sampled on faces."""
    return -G.T @ np.diag(a_faces) @ G


def propagator(L, dt):
    return expm(L * dt)


def real_fourier_basis():
    """Orthonormal real Fourier basis (rows): DC, cos k, sin k, ..., Nyquist."""
    B = [np.ones(N) / np.sqrt(N)]
    for k in range(1, N // 2):
        B.append(np.cos(k * X) * np.sqrt(2.0 / N))
        B.append(np.sin(k * X) * np.sqrt(2.0 / N))
    B.append(np.cos((N // 2) * X) / np.sqrt(N))
    return np.array(B)


FOURIER = real_fourier_basis()


def eigenbasis(L):
    """Orthonormal eigenbasis (rows) of a symmetric operator."""
    w, V = eigh((L + L.T) / 2)
    return V.T, w


def offdiag_fraction(B, P):
    M = B @ P @ B.T
    off = M - np.diag(np.diag(M))
    return np.linalg.norm(off) / np.linalg.norm(M)


def alignment(B_test, B_ref=FOURIER):
    """Mean over rows of B_test of the energy captured by the best single frequency of
    the real Fourier basis (cos/sin pair pooled).  Equals 1 when every vector of B_test is
    a pure sinusoid, whatever the phase."""
    C = B_test @ B_ref.T                          # overlaps
    E = C ** 2
    # pool the cos/sin pair of each frequency
    pooled = [E[:, 0]]
    for k in range(1, N // 2):
        pooled.append(E[:, 2 * k - 1] + E[:, 2 * k])
    pooled.append(E[:, -1])
    pooled = np.array(pooled).T
    return float(pooled.max(1).mean())


# ---------------------------------------------------------------- coefficient fields
def field_from_harmonics(coeffs, x):
    """a(x) = 1 + sum_m c_m cos(m x) + s_m sin(m x);  coeffs = [c1, s1, c2, s2, ...]."""
    a = np.ones_like(x)
    for m in range(len(coeffs) // 2):
        a = a + coeffs[2 * m] * np.cos((m + 1) * x) + coeffs[2 * m + 1] * np.sin((m + 1) * x)
    return a


def truth_field(kind="well"):
    """Fixed, smooth, normalized heterogeneity profile f(x) with max|f| = 0.24, so that
    a = 1 + beta f stays positive up to beta = 4.
      'well' : harmonics 1..3  (inside a 6-parameter family)
      'mis'  : harmonics 1..3 plus a strong 5th harmonic (outside that family)"""
    if kind == "well":
        c = np.array([0.60, 0.00, 0.30, 0.20, 0.10, 0.00])
    elif kind == "mis":
        c = np.array([0.45, 0.00, 0.25, 0.15, 0.00, 0.00, 0.00, 0.00, 0.55, 0.20])
    else:
        raise ValueError(kind)
    f = field_from_harmonics(c, X) - 1.0
    ff = field_from_harmonics(c, XF) - 1.0
    s = 0.24 / np.abs(np.concatenate([f, ff])).max()
    return f * s, ff * s


# ---------------------------------------------------------------- Burgers
NU = 0.05
K = np.fft.fftfreq(N, d=DX) * 2 * np.pi          # wavenumbers
DEALIAS = (np.abs(K) < (2.0 / 3.0) * K.max())


def spectral_deriv(u, order=1):
    return np.real(np.fft.ifft(((1j * K) ** order) * np.fft.fft(u)))


def burgers_rhs(u, eps, nu=NU):
    uh = np.fft.fft(u)
    ux = np.real(np.fft.ifft(1j * K * uh))
    uxx = np.real(np.fft.ifft(-(K ** 2) * uh))
    nl = np.fft.fft(u * ux) * DEALIAS
    return nu * uxx - eps * np.real(np.fft.ifft(nl))


def rk4_step(u, dt, eps, nu=NU):
    k1 = burgers_rhs(u, eps, nu)
    k2 = burgers_rhs(u + 0.5 * dt * k1, eps, nu)
    k3 = burgers_rhs(u + 0.5 * dt * k2, eps, nu)
    k4 = burgers_rhs(u + dt * k3, eps, nu)
    return u + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def integrate(u0, eps, T, dt=2e-3, nu=NU, sub=1):
    """Integrate to time T; return snapshots every `sub` RK4 steps (including u0)."""
    n = int(round(T / dt))
    u = u0.copy()
    out = [u.copy()]
    for i in range(1, n + 1):
        u = rk4_step(u, dt, eps, nu)
        if i % sub == 0:
            out.append(u.copy())
    return np.array(out)


def random_ic(rng, amp=1.0, kmax=4):
    """Smooth random initial condition: a few low harmonics with random phases."""
    u = np.zeros(N)
    for k in range(1, kmax + 1):
        u += rng.normal() * np.cos(k * X) / k + rng.normal() * np.sin(k * X) / k
    u -= u.mean()
    return amp * u / np.abs(u).max()


def rel_l2(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))
