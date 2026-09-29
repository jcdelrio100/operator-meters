"""DMD = data-driven Laplace transform — a numerical check.

For the linear heat equation (eps = 0) the exact solution is a superposition of exponential
modes  u(x,t) = sum_m c_m e^{-nu m^2 t} phi_m(x).  DMD fits the best linear one-step map A on
snapshot pairs; its eigenvalues mu_j give continuous-time exponents  lambda_j = log(mu_j)/Delta,
i.e. the poles of the Laplace transform of the dynamics.  Here we check that the recovered
lambda_j equal -nu m^2 for the modes present in the data, to machine precision.

Outputs: results/dmd_laplace_check.json (and a printed table)
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from data import load, pairs, DELTA
from models import dmd
from common import NU

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")

tr, te, _ = load(0.0)
X, Y = pairs(tr)
A = dmd(X, Y)
mu = np.linalg.eigvals(A)
lam = np.log(mu.astype(complex)) / DELTA
# keep the modes actually excited by the data (the initial conditions hold harmonics 1..4)
order = np.argsort(-np.abs(mu))
rows = []
for j in order[:8]:
    l = lam[j]
    m = np.sqrt(max(-l.real, 0) / NU)
    rows.append(dict(mu=float(np.real(mu[j])), lam_real=float(l.real), lam_imag=float(l.imag),
                     m_est=float(m), m_round=int(round(m)), truth=float(-NU * round(m) ** 2),
                     abs_err=float(abs(l.real + NU * round(m) ** 2))))
print(f"{'mu':>10s} {'lambda=log(mu)/Delta':>22s} {'m':>4s} {'-nu m^2':>10s} {'|err|':>10s}")
for r in rows:
    print(f"{r['mu']:10.6f} {r['lam_real']:22.8f} {r['m_round']:4d} {r['truth']:10.6f} {r['abs_err']:10.2e}")
excited = [r for r in rows if r["m_round"] <= 4]
worst = max(r["abs_err"] for r in excited)
print(f"max |lambda + nu m^2| over the excited modes m<=4: {worst:.2e}")
json.dump(dict(rows=rows, max_abs_err_excited=worst, delta=DELTA, nu=NU),
          open(os.path.join(RES, "dmd_laplace_check.json"), "w"), indent=1)
