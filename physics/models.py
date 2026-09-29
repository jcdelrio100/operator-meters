"""Frozen-DMD head + nonlinear branch designs, shared by WP2, WP2b, WP4, WP4b.

  u_hat_1 = A u_0 + g * NL(u_0)

A  : closed-form least-squares one-step propagator (DMD), frozen.
g  : scalar gate, initialised at 1 and L1-penalized so that it must earn its keep — the instrument.
     (starting at g = 0 makes g * NL a dead saddle: neither factor receives gradient.)
NL : one of
     'mlp'    one-step multilayer perceptron              (black box, no depth)
     'resint4','resint8'  residual integrator, k stages   u <- u + h f_i(u)   (neural-ODE-like)
     'quad'   structure-matched quadratic branch: linear map on [u, u_x, u*u_x]
     'quadint4'  structured integrator: the SAME quadratic term (shared W) applied in 4
                 sub-steps  v <- v + (1/4) W[v, v_x, v v_x]  — structure plus depth, no extra
                 parameters; the analogue of sub-stepping an explicit integrator.
"""
import os, sys
import numpy as np
import torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(__file__))
from common import N, K

torch.set_num_threads(2)
KT = torch.tensor(K, dtype=torch.float32)


def dmd(X, Y, ridge=1e-8):
    """Least-squares A with Y ~ A X (rows are samples): A = Y^T X (X^T X + ridge I)^-1."""
    XtX = X.T @ X + ridge * np.eye(X.shape[1])
    return np.linalg.solve(XtX, X.T @ Y).T


def spectral_dx_torch(u):
    uh = torch.fft.fft(u, dim=-1)
    return torch.real(torch.fft.ifft(1j * KT * uh, dim=-1))


class MLP(nn.Module):
    def __init__(self, h=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(N, h), nn.Tanh(), nn.Linear(h, N))
    def forward(self, u): return self.net(u)


class ResInt(nn.Module):
    """k residual stages with their own small nets: u <- u + (1/k) f_i(u); output u_k - u_0."""
    def __init__(self, stages=4, h=64):
        super().__init__()
        self.f = nn.ModuleList([nn.Sequential(nn.Linear(N, h), nn.Tanh(), nn.Linear(h, N))
                                for _ in range(stages)])
        self.h = 1.0 / stages
    def forward(self, u):
        v = u
        for f in self.f:
            v = v + self.h * f(v)
        return v - u


class Quad(nn.Module):
    """Structure-matched branch: features [u, u_x, u*u_x] -> linear map to N."""
    def __init__(self):
        super().__init__()
        self.W = nn.Linear(3 * N, N, bias=False)
        nn.init.normal_(self.W.weight, std=1e-2)     # not zero: g=0 & W=0 is a dead saddle
    def forward(self, u):
        ux = spectral_dx_torch(u)
        return self.W(torch.cat([u, ux, u * ux], -1))


class QuadInt(nn.Module):
    """Structured residual integrator: k sub-steps of one shared quadratic term; output v_k - u."""
    def __init__(self, stages=4):
        super().__init__()
        self.W = nn.Linear(3 * N, N, bias=False)
        nn.init.normal_(self.W.weight, std=1e-2)
        self.k = stages
    def forward(self, u):
        v = u
        for _ in range(self.k):
            vx = spectral_dx_torch(v)
            v = v + (1.0 / self.k) * self.W(torch.cat([v, vx, v * vx], -1))
        return v - u


def make_branch(kind):
    return {"mlp": MLP, "resint4": lambda: ResInt(4), "resint8": lambda: ResInt(8),
            "quad": Quad, "quadint4": lambda: QuadInt(4)}[kind]()


class Gated(nn.Module):
    """Frozen linear head A plus gated nonlinear branch."""
    def __init__(self, A, kind="mlp", g0=1.0, seed=0):
        super().__init__()
        torch.manual_seed(seed)                      # reproducible branch initialisation
        self.register_buffer("A", torch.tensor(A, dtype=torch.float32))
        self.nl = make_branch(kind)
        self.g = nn.Parameter(torch.tensor(float(g0)))
    def linear(self, u): return u @ self.A.T
    def branch(self, u): return self.g * self.nl(u)
    def forward(self, u): return self.linear(u) + self.branch(u)


def train_1step(model, X, Y, epochs=1500, lr=1e-3, lam=1e-3, wd=3e-5, seed=0, verbose=False):
    """Train only the branch (A frozen) on one-step pairs with an L1 penalty on the gate."""
    torch.manual_seed(seed)
    Xt = torch.tensor(X, dtype=torch.float32); Yt = torch.tensor(Y, dtype=torch.float32)
    params = [p for n, p in model.named_parameters()]
    opt = torch.optim.Adam(params, lr=lr, weight_decay=wd)
    for ep in range(epochs):
        opt.zero_grad()
        pred = model(Xt)
        loss = ((pred - Yt) ** 2).mean() / ((Yt) ** 2).mean() + lam * model.g.abs()
        loss.backward(); opt.step()
        if verbose and ep % 300 == 0:
            print(f"    ep {ep:5d} loss {loss.item():.4e} g {model.g.item():+.4f}")
    return model


def finetune_rollout(model, traj, K=5, epochs=300, lr=3e-4, clip=1.0, seed=0):
    """Fine-tune the branch on K-step rollouts from every snapshot (gradient clipping)."""
    torch.manual_seed(seed)
    T = torch.tensor(traj, dtype=torch.float32)          # [n, S, N]
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    S = T.shape[1]
    for ep in range(epochs):
        opt.zero_grad()
        u = T[:, :S - K].reshape(-1, N)
        loss = 0.0
        for k in range(1, K + 1):
            u = model(u)
            tgt = T[:, k:S - K + k].reshape(-1, N)
            loss = loss + ((u - tgt) ** 2).mean() / (tgt ** 2).mean()
        (loss / K).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()
    return model


@torch.no_grad()
def rollout(model, u0, steps):
    u = torch.tensor(u0, dtype=torch.float32)
    out = [u.numpy().copy()]
    for _ in range(steps):
        u = model(u); out.append(u.numpy().copy())
    return np.array(out)                                  # [steps+1, n, N]


@torch.no_grad()
def evaluate(model, X, Y):
    """Held-out: DMD residual, residual after branch, capture, gate firing amplitude."""
    Xt = torch.tensor(X, dtype=torch.float32); Yt = torch.tensor(Y, dtype=torch.float32)
    lin = model.linear(Xt); br = model.branch(Xt)
    r_lin = ((Yt - lin) ** 2).sum().item()
    r_all = ((Yt - lin - br) ** 2).sum().item()
    ynorm = (Yt ** 2).sum().item()
    return dict(res_dmd=float(np.sqrt(r_lin / ynorm)),
                res_after=float(np.sqrt(r_all / ynorm)),
                capture=float(1 - r_all / max(r_lin, 1e-30)),
                firing=float(br.norm() / lin.norm()),
                g=float(model.g.item()))
