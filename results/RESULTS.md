# Results notes — operator-meters

Every number below is read from the JSON file named in each section; the paper's numbers.tex is generated from the same files.

## Classification gate law (`classification/gap_law.py` → `gap_law.json`, `figures/gap_law.png`)

- rotated(θ) family, θ ∈ [0.0, 0.2, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9, 1.0], seeds [0, 1, 2] (36 runs) + 4 benchmark tasks × 3 seeds (12 runs, held out of the fit).
- Gate shut wherever G ≤ 0: max g_s2 = 0.0016 over 18 runs.
- Rectified saturating law g ≈ 0.594·(1 − exp(−[G]+/0.304 bits)): r = 0.994 (family), 0.982 (held-out tasks).
- Rectified linear g ≈ 0.081 + 0.348·[G]+: r = 0.912 / 0.962.
- 1D global (Fourier): G = -0.03 bits, g_s2 = 0.000 (3 seeds).
- 1D localized (wavelet): G = -0.03 bits, g_s2 = 0.001 (3 seeds).
- hidden basis (rotated): G = +1.34 bits, g_s2 = 0.555 (3 seeds).
- XOR in hidden basis: G = +0.31 bits, g_s2 = 0.261 (3 seeds).
- Limits: I(T;Y) is a cross-entropy lower bound (clipped to [0, H(Y)]); family is synthetic by design; L1 weight 0.25 as in the companion study, unmodified.

## Rate–relevance (`classification/rate_relevance.py` → `rate_relevance.json`)

- 1D global (Fourier): energy-selected K=16 → 1.000 ± 0.000; F-score-selected → 1.000 ± 0.000 (3 seeds).
- 1D localized (wavelet): energy-selected K=16 → 0.265 ± 0.016; F-score-selected → 0.999 ± 0.001 (3 seeds).
- hidden basis (rotated): energy-selected K=16 → 0.584 ± 0.049; F-score-selected → 0.660 ± 0.048 (3 seeds).
- Reading: on the localized task energy selection keeps the carrier lines and discards the position-carrying wavelet coefficients; on the hidden-basis task both fail (no dictionary coefficient carries the classes).

## Reading the classification basis (`classification/symbolic_U.py` → `symbolic_U.json`)

- DFT, untrained: closed-form fit 1.000, a/(2π/N) = 1.000, pure-sinusoid rows 100%, median novelty -0.000.
- compression prior + CE: closed-form fit 0.088, a/(2π/N) = 1.003, pure-sinusoid rows 0%, median novelty 0.688.
- CE alone, random start: closed-form fit 0.030, a/(2π/N) = 1.010, pure-sinusoid rows 0%, median novelty 0.780.
- Reading: the untrained DFT reads exactly; task-trained bases do not. Motivates Part II.

## WP1 heterogeneity meter + WP1b discovery (`physics/phys_wp1.py` → `phys_wp1.json`)

- dt = 0.05, β ∈ [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]; truth profile max|f| = 0.24 (a > 0 up to β = 4).
- Meter: β=0: D_F=0.000, D_Φ=2e-15, align=1.00; β=0.5: D_F=0.024, D_Φ=3e-15, align=0.70; β=1: D_F=0.047, D_Φ=4e-15, align=0.56; β=1.5: D_F=0.069, D_Φ=6e-15, align=0.48; β=2: D_F=0.091, D_Φ=4e-15, align=0.43; β=3: D_F=0.133, D_Φ=4e-15, align=0.35; β=4: D_F=0.178, D_Φ=9e-15, align=0.30
- well-specified: D 0.024–0.178 → 0.000000; corr(a) 1.0000–1.0000.
- mis-specified: D 0.020–0.190 → 0.132882; corr(a) 0.6458–0.6751.
- re-specified: D 0.020–0.190 → 0.000000; corr(a) 1.0000–1.0000.
- Limits: a identifiable only up to positive scale (standardized before scoring); the mis-specified truth keeps 3 of its harmonics inside the family, hence corr ≈ 0.65 rather than ≈ 0; the objective plateau (not the correlation) is what announces mis-specification.

## DMD = data-driven Laplace (`physics/dmd_laplace_check.py` → `dmd_laplace_check.json`)

- Heat equation (ε = 0), Δ = 0.17: DMD exponents log(μ)/Δ equal −ν m² for m ≤ 4 to 4.0e-10.

## WP2 nonlinearity meter and gate (`physics/phys_wp2.py` → `phys_wp2.json`)

- epochs 1500, L1 λ = 0.001, weight decay 3e-05, gate initialised at 1 (g = 0 is a dead saddle), black-box MLP branch.
- ε=0: DMD res 0.000, capture 0.0%, firing 0.0006; ε=0.1: DMD res 0.009, capture -2.2%, firing 0.0015; ε=0.2: DMD res 0.018, capture 0.3%, firing 0.0025; ε=0.4: DMD res 0.036, capture 14.1%, firing 0.0126; ε=0.7: DMD res 0.059, capture 32.3%, firing 0.0313; ε=1: DMD res 0.082, capture 44.4%, firing 0.0519; ε=1.5: DMD res 0.126, capture 49.2%, firing 0.0897; ε=2: DMD res 0.173, capture 54.6%, firing 0.1304
- Reading: firing ≤ 0.0025 up to ε = 0.2 (no out-of-sample capture), then rising with the residual — rectified law, no saturation within the sweep.

## WP2b depth vs structure (`physics/depth_test.py` → `depth_test.json`)

- Mean held-out capture, ε ≥ 0.7: mlp 45.1%, resint4 68.8%, resint8 63.5%, quad 90.3%, quadint4 98.9%
- At ε = 0.1: mlp -2.2%, resint4 -1.1%, resint8 -1.0%, quad 79.5%, quadint4 81.8%
- Reading: depth helps (resint4 > mlp; resint8 under-trains at the same budget); the structure-matched linear map on [u, u_x, u·u_x] beats all black boxes; sub-stepping the same term (no extra parameters) captures ≈ 99%.

## WP3 reading the operator (`physics/phys_wp3_sindy.py` → `phys_wp3.json`)

- ε=0.1: ν̂=0.0500, coef(u u_x)=-0.100, max|other|=0.0e+00; ε=0.2: ν̂=0.0500, coef(u u_x)=-0.200, max|other|=0.0e+00; ε=0.4: ν̂=0.0500, coef(u u_x)=-0.399, max|other|=0.0e+00; ε=0.7: ν̂=0.0500, coef(u u_x)=-0.699, max|other|=0.0e+00; ε=1: ν̂=0.0497, coef(u u_x)=-0.988, max|other|=0.0e+00; ε=1.5: ν̂=0.0458, coef(u u_x)=-1.359, max|other|=0.0e+00; ε=2: ν̂=0.0378, coef(u u_x)=-1.537, max|other|=8.4e-02
- Reading: exact up to ε = 1.0; under-estimated for ε ≥ 1.5 (under-resolved shocks at N = 64, numerical u_t degrades).

## WP4 extrapolation + WP4b hardening (`physics/phys_wp4_rollout.py` → `phys_wp4.json`)

- ε = 1.0, 30-step rollout, OOD amplitude 1.6, seeds [0, 1, 2] (network init only). 'diverged' = relative error > 10 or non-finite.
- linear: id 0.697 ± 0.000, ood 0.860 ± 0.000; energy-growth steps id 0%, ood 0%.
- mlp_1step: id 1.152 ± 0.249, ood 2.907 ± 0.409; energy-growth steps id 14%, ood 33%.
- mlp_rollout: id 0.531 ± 0.063, ood 1.508 ± 0.124; energy-growth steps id 2%, ood 11%.
- quad_1step: id 0.064 ± 0.001, ood diverged; energy-growth steps id 0%, ood 66%.
- quad_rollout: id 0.033 ± 0.001, ood diverged; energy-growth steps id 0%, ood 36%.
- quadint4_1step: id 0.037 ± 0.002, ood 0.096 ± 0.009; energy-growth steps id 0%, ood 0%.
- quadint4_rollout: id 0.024 ± 0.009, ood 0.047 ± 0.004; energy-growth steps id 0%, ood 0%.
- Amplitude sweep (err@30, seed 0, fresh ICs): linear: 1.0→0.625, 1.2→0.789, 1.4→0.762, 1.6→1.034; mlp: 1.0→0.771, 1.2→1.736, 1.4→1.501, 1.6→3.195; quad: 1.0→0.068, 1.2→div, 1.4→div, 1.6→div; quadint4: 1.0→0.043, 1.2→0.097, 1.4→0.067, 1.6→0.120
- Readings: (1) the black box is worse than the linear model under rollout and creates energy; (2) the discovered term in ONE explicit step extrapolates in horizon but diverges at any amplitude above training (explicit-integrator stability limit) — an unanticipated negative result; (3) the same term sub-stepped 4× extrapolates in horizon and amplitude and never creates energy; (4) rollout fine-tuning improves both by a similar factor, which leaves only the structured integrator usable.
- Method note: rollout training from a cold network diverged in an earlier run (relative error exactly 1.000); it must be a fine-tune on the converged one-step model with gradient clipping.
