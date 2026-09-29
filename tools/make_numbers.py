"""Read every results/*.json and write paper/numbers.tex — the macros the paper text uses.
Run after the experiments; a number that changes in a JSON changes in the paper."""
import os, json, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results"); OUT = os.path.join(ROOT, "paper", "numbers.tex")
J = lambda n: json.load(open(os.path.join(RES, n)))
M = {}
def mac(name, val): M[name] = val
pct = lambda x: f"{100*x:.1f}\\%"
f3 = lambda x: f"{x:.3f}"
pm = lambda s: f"{s['mean']:.3f} $\\pm$ {s['std']:.3f}" if s["diverged"] == 0 else ("diverged" if s["diverged"] == s["n"] else f"{s['mean']:.3f} ({s['diverged']}/{s['n']} diverged)")

# ---------------- classification
g = J("gap_law.json"); L = g["law"]["linear"]; S = g["law"]["saturating"]
mac("gapA", f"{L['a']:.2f}"); mac("gapB", f"{L['b']:.2f}"); mac("gapR", f"{L['r_family']:.2f}"); mac("gapRheld", f"{L['r_heldout']:.2f}")
mac("gapC", f"{S['c']:.2f}"); mac("gapGzero", f"{S['G0']:.2f}"); mac("gapRsat", f"{S['r_family']:.2f}"); mac("gapRsatHeld", f"{S['r_heldout']:.2f}")
mac("gapN", str(g["law"]["n"])); mac("gapNrot", str(g["law"]["n_family"])); mac("gapNreal", str(g["law"]["n_heldout"]))
mac("gapGshut", f"{g['law']['gate_max_when_G_le_0']:.3f}"); mac("gapNshut", str(g["law"]["n_G_le_0"]))
rows = g["rows"]
def task_mean(name, key): return float(np.mean([r[key] for r in rows if r["task"] == name]))
mac("gapHiddenG", f"{task_mean('hidden basis (rotated)','G'):.2f}"); mac("gapHiddenGate", f"{task_mean('hidden basis (rotated)','g_s2'):.2f}")
mac("gapXorG", f"{task_mean('XOR in hidden basis','G'):.2f}"); mac("gapXorGate", f"{task_mean('XOR in hidden basis','g_s2'):.2f}")
mac("gapGmax", f"{max(r['G'] for r in rows):.2f}")
rr = J("rate_relevance.json")["out"]
for key, name in [("Glob", "1D global (Fourier)"), ("Loc", "1D localized (wavelet)"), ("Hid", "hidden basis (rotated)")]:
    mac(f"rr{key}Energy", f3(np.mean(rr[name]["energy"]))); mac(f"rr{key}Task", f3(np.mean(rr[name]["fscore"])))
sy = J("symbolic_U.json")
mac("symDftFit", f"{sy['dft']['fit']:.3f}"); mac("symDftA", f"{sy['dft']['a_over_2pi_N']:.3f}")
mac("symSparseFit", f"{sy['sparse']['fit']:.3f}"); mac("symCeFit", f"{sy['ce']['fit']:.3f}"); mac("symCePure", pct(sy["ce"]["frac_pure"]))
mac("symSparsePure", pct(sy["sparse"]["frac_pure"]))

# ---------------- physics WP1
w1 = J("phys_wp1.json"); mt = w1["meter"]; d = w1["discovery"]
mac("wpOneDmax", f3(max(r["D_fourier"] for r in mt))); mac("wpOneAlignMin", f"{min(r['align'] for r in mt):.2f}")
mac("wpOneDeigMax", "$" + f"{max(r['D_eigen'] for r in mt):.0e}".replace("e-", "\\cdot 10^{-") + "}$")
well, mis, re_ = d["well-specified"], d["mis-specified"], d["re-specified"]
mac("wpOneDrange", f"{min(r['D_fourier'] for r in well):.3f}–{max(r['D_fourier'] for r in well):.3f}")
mac("wpOneDiscD", f"{max(r['D_discovered'] for r in well+re_):.6f}")
mac("wpOneCorr", f"{min(r['corr'] for r in well+re_):.4f}")
mac("wpOneMisD", f"{max(r['D_fourier'] for r in mis):.3f} $\\to$ {max(r['D_discovered'] for r in mis):.3f} at $\\beta=4$")
mac("wpOneMisCorr", f"{min(r['corr'] for r in mis):.2f}–{max(r['corr'] for r in mis):.2f}")
dl = J("dmd_laplace_check.json"); e = dl["max_abs_err_excited"]
mac("dmdLaplaceErr", "$" + f"{e:.0e}".replace("e-", "\\cdot 10^{-") + "}$")
import importlib.util, sys
sys.path.insert(0, os.path.join(ROOT, "physics")); import data as D
mac("wpDelta", f"{D.DELTA:.2f}"); mac("wpNtr", str(D.NTR)); mac("wpNte", str(D.NTE)); mac("wpAmpOod", f"{D.AMP_OOD:g}")

# ---------------- physics WP2
w2 = J("phys_wp2.json")["rows"]
mac("wpTwoResMax", f3(max(r["res_dmd"] for r in w2)))
shut = [r for r in w2 if r["eps"] <= 0.2]; mac("wpTwoFireShut", f"{max(r['firing'] for r in shut):.3f}"); mac("wpTwoEpsThresh", "0.2")
mac("wpTwoFireMax", f3(max(r["firing"] for r in w2)))
caps = [r["capture"] for r in w2 if r["eps"] >= 0.7]; mac("wpTwoCapRange", f"{100*min(caps):.0f}–{100*max(caps):.0f}\\%")
mac("wpTwoCapOne", pct([r for r in w2 if r["eps"] == 1.0][0]["capture"]))
mac("wpTwoFireOne", f3([r for r in w2 if r["eps"] == 1.0][0]["firing"]))
mac("wpTwoFireFour", f3([r for r in w2 if r["eps"] == 0.4][0]["firing"]))
dt = J("depth_test.json")["mean_capture_eps_ge_0p7"]
for k, n in [("mlp", "Mlp"), ("resint4", "ResFour"), ("resint8", "ResEight"), ("quad", "Quad"), ("quadint4", "QuadInt")]:
    mac(f"wpDepth{n}", pct(dt[k]))
dts = J("depth_test.json")["sweep"]
mac("wpDepthQuadIntEpsOne", pct([r for r in dts["quadint4"] if r["eps"] == 0.1][0]["capture"]))
mac("wpDepthMlpEpsOne", pct([r for r in dts["mlp"] if r["eps"] == 0.1][0]["capture"]))

# ---------------- physics WP3
w3 = J("phys_wp3.json")["rows"]; by = {r["eps"]: r for r in w3}
mac("wpSindyNuOne", "0.05"); mac("wpSindyNuHat", f3(by[1.0]["nu_hat"]))
mac("wpSindyEpsList", ", ".join(f"{-by[e]['eps_hat']:.3f}" for e in [0.2, 0.4, 0.7, 1.0]))
mac("wpSindyEpsTwo", f"{-by[2.0]['eps_hat']:.3f}"); mac("wpSindyNuTwo", f3(by[2.0]["nu_hat"]))
mac("wpSindyEpsOneHalf", f"{-by[1.5]['eps_hat']:.3f}")

# ---------------- physics WP4 / WP4b
w4 = J("phys_wp4.json"); s = w4["summary"]; amp = w4["amplitude_sweep"]
c0 = {k: (w4["curves"][k] if k == "linear" else w4["curves"][k][0]) for k in w4["curves"]}
def last(k, c):
    v = c0[k][c][-1]; return "diverged" if v is None or not np.isfinite(v) or v > 10 else f3(v)
mac("wpRollLinId", last("linear", "id")); mac("wpRollLinOod", last("linear", "ood"))
mac("wpRollMlpId", last("mlp_1step", "id")); mac("wpRollMlpOod", last("mlp_1step", "ood"))
mac("wpRollQuadId", last("quad_1step", "id")); mac("wpRollQuadOod", last("quad_1step", "ood"))
mac("wpRollQuadIntId", last("quadint4_1step", "id")); mac("wpRollQuadIntOod", last("quadint4_1step", "ood"))
mac("wpRollGainId", f"{s['linear']['id']['mean']/s['quadint4_1step']['id']['mean']:.0f}")
mac("wpRollGainOod", f"{s['linear']['ood']['mean']/s['quadint4_1step']['ood']['mean']:.0f}")
mac("wpRollGainMlpId", f"{s['mlp_1step']['id']['mean']/s['quadint4_1step']['id']['mean']:.0f}")
mac("wpRollGainMlpOod", f"{s['mlp_1step']['ood']['mean']/s['quadint4_1step']['ood']['mean']:.0f}")
mac("wpAmpQuadIntMax", f3(max(v for v in amp["quadint4"].values() if v is not None)))
mac("wpAmpLinMax", f3(max(v for v in amp["linear"].values() if v is not None)))
for k, n in [("mlp_1step", "MlpOne"), ("mlp_rollout", "MlpRoll"), ("quadint4_1step", "QuadOne"), ("quadint4_rollout", "QuadRoll"), ("quad_1step", "QuadStepOne"), ("quad_rollout", "QuadStepRoll")]:
    mac(f"wpHard{n}Id", pm(s[k]["id"])); mac(f"wpHard{n}Ood", pm(s[k]["ood"]))
mac("wpHardGain", f"{s['quadint4_1step']['id']['mean']/s['quadint4_rollout']['id']['mean']:.1f}")
mac("wpHardGainOod", f"{s['quadint4_1step']['ood']['mean']/s['quadint4_rollout']['ood']['mean']:.1f}")
mac("wpHardMlpGain", f"{s['mlp_1step']['id']['mean']/s['mlp_rollout']['id']['mean']:.1f}")
mac("wpHardMlpGainOod", f"{s['mlp_1step']['ood']['mean']/s['mlp_rollout']['ood']['mean']:.1f}")
eg = lambda k, c: 100 * s[k][c]["mean"]
mac("wpEnergyMlpIdMean", f"{eg('mlp_1step','energy_id'):.0f}\\%"); mac("wpEnergyMlpOodMean", f"{eg('mlp_1step','energy_ood'):.0f}\\%")
mac("wpEnergyMlpRollId", f"{eg('mlp_rollout','energy_id'):.0f}\\%"); mac("wpEnergyMlpRollOod", f"{eg('mlp_rollout','energy_ood'):.0f}\\%")
mac("wpEnergyQuadIntId", f"{eg('quadint4_1step','energy_id'):.0f}\\%"); mac("wpEnergyQuadIntOod", f"{eg('quadint4_1step','energy_ood'):.0f}\\%")
mac("wpEnergyQuadStepOod", f"{eg('quad_1step','energy_ood'):.0f}\\%")
mac("wpEnergyLinId", f"{eg('linear','energy_id'):.0f}\\%")

with open(OUT, "w") as f:
    f.write("% generated by tools/make_numbers.py — do not edit\n")
    for k, v in M.items():
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print(f"wrote {len(M)} macros to {OUT}")
for k, v in M.items(): print(f"  {k:24s} {v}")
