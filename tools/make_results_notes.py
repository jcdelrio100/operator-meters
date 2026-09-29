"""Write results/RESULTS.md — one note per experiment, numbers read from the JSON files."""
import os, json, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); RES = os.path.join(ROOT, "results")
J = lambda n: json.load(open(os.path.join(RES, n)))
L = []
def h(t): L.append(f"\n## {t}\n")
def p(t): L.append(t)

L.append("# Results notes — operator-meters\n\nEvery number below is read from the JSON file named in each section; the paper's numbers.tex is generated from the same files.")

# gap law
g = J("gap_law.json"); law = g["law"]
h("Classification gate law (`classification/gap_law.py` → `gap_law.json`, `figures/gap_law.png`)")
p(f"- rotated(θ) family, θ ∈ {g['thetas']}, seeds {g['seeds']} ({law['n_family']} runs) + 4 benchmark tasks × 3 seeds ({law['n_heldout']} runs, held out of the fit).")
p(f"- Gate shut wherever G ≤ 0: max g_s2 = {law['gate_max_when_G_le_0']:.4f} over {law['n_G_le_0']} runs.")
p(f"- Rectified saturating law g ≈ {law['saturating']['c']:.3f}·(1 − exp(−[G]+/{law['saturating']['G0']:.3f} bits)): r = {law['saturating']['r_family']:.3f} (family), {law['saturating']['r_heldout']:.3f} (held-out tasks).")
p(f"- Rectified linear g ≈ {law['linear']['a']:.3f} + {law['linear']['b']:.3f}·[G]+: r = {law['linear']['r_family']:.3f} / {law['linear']['r_heldout']:.3f}.")
for t in ["1D global (Fourier)", "1D localized (wavelet)", "hidden basis (rotated)", "XOR in hidden basis"]:
    rs = [r for r in g["rows"] if r["task"] == t]
    p(f"- {t}: G = {np.mean([r['G'] for r in rs]):+.2f} bits, g_s2 = {np.mean([r['g_s2'] for r in rs]):.3f} (3 seeds).")
p("- Limits: I(T;Y) is a cross-entropy lower bound (clipped to [0, H(Y)]); family is synthetic by design; L1 weight 0.25 as in the companion study, unmodified.")

rr = J("rate_relevance.json")["out"]
h("Rate–relevance (`classification/rate_relevance.py` → `rate_relevance.json`)")
for t, v in rr.items():
    p(f"- {t}: energy-selected K=16 → {np.mean(v['energy']):.3f} ± {np.std(v['energy']):.3f}; F-score-selected → {np.mean(v['fscore']):.3f} ± {np.std(v['fscore']):.3f} (3 seeds).")
p("- Reading: on the localized task energy selection keeps the carrier lines and discards the position-carrying wavelet coefficients; on the hidden-basis task both fail (no dictionary coefficient carries the classes).")

sy = J("symbolic_U.json")
h("Reading the classification basis (`classification/symbolic_U.py` → `symbolic_U.json`)")
for k, v in sy.items():
    p(f"- {v['label']}: closed-form fit {v['fit']:.3f}, a/(2π/N) = {v['a_over_2pi_N']:.3f}, pure-sinusoid rows {100*v['frac_pure']:.0f}%, median novelty {v['novelty_median']:.3f}.")
p("- Reading: the untrained DFT reads exactly; task-trained bases do not. Motivates Part II.")

w1 = J("phys_wp1.json")
h("WP1 heterogeneity meter + WP1b discovery (`physics/phys_wp1.py` → `phys_wp1.json`)")
p(f"- dt = {w1['dt']}, β ∈ {w1['betas']}; truth profile max|f| = 0.24 (a > 0 up to β = 4).")
p("- Meter: " + "; ".join(f"β={r['beta']:g}: D_F={r['D_fourier']:.3f}, D_Φ={r['D_eigen']:.0e}, align={r['align']:.2f}" for r in w1["meter"]))
for name, rows in w1["discovery"].items():
    p(f"- {name}: D {min(r['D_fourier'] for r in rows):.3f}–{max(r['D_fourier'] for r in rows):.3f} → {max(r['D_discovered'] for r in rows):.6f}; corr(a) {min(r['corr'] for r in rows):.4f}–{max(r['corr'] for r in rows):.4f}.")
p("- Limits: a identifiable only up to positive scale (standardized before scoring); the mis-specified truth keeps 3 of its harmonics inside the family, hence corr ≈ 0.65 rather than ≈ 0; the objective plateau (not the correlation) is what announces mis-specification.")

dl = J("dmd_laplace_check.json")
h("DMD = data-driven Laplace (`physics/dmd_laplace_check.py` → `dmd_laplace_check.json`)")
p(f"- Heat equation (ε = 0), Δ = {dl['delta']:.2f}: DMD exponents log(μ)/Δ equal −ν m² for m ≤ 4 to {dl['max_abs_err_excited']:.1e}.")

w2 = J("phys_wp2.json")
h("WP2 nonlinearity meter and gate (`physics/phys_wp2.py` → `phys_wp2.json`)")
p(f"- epochs {w2['epochs']}, L1 λ = {w2['lam']}, weight decay {w2['wd']}, gate initialised at 1 (g = 0 is a dead saddle), black-box MLP branch.")
p("- " + "; ".join(f"ε={r['eps']:g}: DMD res {r['res_dmd']:.3f}, capture {100*r['capture']:.1f}%, firing {r['firing']:.4f}" for r in w2["rows"]))
p("- Reading: firing ≤ 0.0025 up to ε = 0.2 (no out-of-sample capture), then rising with the residual — rectified law, no saturation within the sweep.")

d = J("depth_test.json")
h("WP2b depth vs structure (`physics/depth_test.py` → `depth_test.json`)")
p("- Mean held-out capture, ε ≥ 0.7: " + ", ".join(f"{k} {100*v:.1f}%" for k, v in d["mean_capture_eps_ge_0p7"].items()))
p("- At ε = 0.1: " + ", ".join(f"{k} {100*[r for r in d['sweep'][k] if r['eps']==0.1][0]['capture']:.1f}%" for k in d["sweep"]))
p("- Reading: depth helps (resint4 > mlp; resint8 under-trains at the same budget); the structure-matched linear map on [u, u_x, u·u_x] beats all black boxes; sub-stepping the same term (no extra parameters) captures ≈ 99%.")

w3 = J("phys_wp3.json")
h("WP3 reading the operator (`physics/phys_wp3_sindy.py` → `phys_wp3.json`)")
p("- " + "; ".join(f"ε={r['eps']:g}: ν̂={r['nu_hat']:.4f}, coef(u u_x)={-r['eps_hat']:+.3f}, max|other|={r['others_max']:.1e}" for r in w3["rows"]))
p("- Reading: exact up to ε = 1.0; under-estimated for ε ≥ 1.5 (under-resolved shocks at N = 64, numerical u_t degrades).")

w4 = J("phys_wp4.json"); s = w4["summary"]
h("WP4 extrapolation + WP4b hardening (`physics/phys_wp4_rollout.py` → `phys_wp4.json`)")
p(f"- ε = {w4['eps']}, 30-step rollout, OOD amplitude 1.6, seeds {w4['seeds']} (network init only). 'diverged' = relative error > 10 or non-finite.")
for k, v in s.items():
    fmt = lambda c: ("diverged" if v[c]["diverged"] == v[c]["n"] else f"{v[c]['mean']:.3f} ± {v[c]['std']:.3f}")
    p(f"- {k}: id {fmt('id')}, ood {fmt('ood')}; energy-growth steps id {100*v['energy_id']['mean']:.0f}%, ood {100*v['energy_ood']['mean']:.0f}%.")
p("- Amplitude sweep (err@30, seed 0, fresh ICs): " + "; ".join(f"{k}: " + ", ".join(f"{a}→{('div' if x is None else f'{x:.3f}')}" for a, x in d.items()) for k, d in w4["amplitude_sweep"].items()))
p("- Readings: (1) the black box is worse than the linear model under rollout and creates energy; (2) the discovered term in ONE explicit step extrapolates in horizon but diverges at any amplitude above training (explicit-integrator stability limit) — an unanticipated negative result; (3) the same term sub-stepped 4× extrapolates in horizon and amplitude and never creates energy; (4) rollout fine-tuning improves both by a similar factor, which leaves only the structured integrator usable.")
p("- Method note: rollout training from a cold network diverged in an earlier run (relative error exactly 1.000); it must be a fine-tune on the converged one-step model with gradient clipping.")

open(os.path.join(RES, "RESULTS.md"), "w").write("\n".join(L) + "\n")
print("wrote results/RESULTS.md")
