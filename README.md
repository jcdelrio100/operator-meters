# When Does Fourier Diagonalize the Physics?

**Calibrated meters for fixed versus learned representations in classification and in operator learning, and what the learned part is worth.**
Juan Carlos del Río Romero — independent research — ORCID 0009-0009-2908-4168. Working draft, September 2026.

The paper is `paper/operator_meters.pdf`. Every number in it is read from `results/*.json` by `tools/make_numbers.py`; the human-readable notes are in `results/RESULTS.md`.

## What is in the paper

One gate — a scalar on a learned branch, initialised open, pushed shut by an L1 penalty — read as a meter in two settings that share the architecture of the companion study *Fixed, Learned, or Both?* but invert its objective.

* **Part I, classification.** The gate follows the information the learned branch adds about the label, in bits, with a rectified saturating law (r = 0.99 on a family whose gap is controlled by design, 0.98 on held-out benchmark tasks). Energy-based (codec-style) selection discards the discriminative coefficients (0.27 vs 0.999 on a localized task). The task-trained basis cannot be read back symbolically.
* **Part II, physics (1-D PDEs).** The off-diagonality of the fixed Fourier basis is a heterogeneity meter; a Sturm–Liouville family searched without an oracle recovers the physical coefficient (corr 1.0000) and flags its own mis-specification; the DMD residual (a data-driven Laplace transform) is a nonlinearity meter and the gated branch fires with the same rectified law; structure beats depth; sparse regression names the term exactly; the named term extrapolates in horizon and amplitude — but only when integrated (sub-stepped), not applied in one explicit step, which diverges above the training amplitude.

## Reproduce

```
pip install -r requirements.txt
python physics/data.py                 # Burgers trajectory cache (~7 min, CPU)
python physics/phys_wp1.py             # heterogeneity meter + discovery        (~5 min)
python physics/dmd_laplace_check.py    # DMD exponents = Laplace poles          (seconds)
python physics/phys_wp2.py             # nonlinearity meter + gate              (~1 min)
python physics/depth_test.py           # depth vs structure                     (~25 min)
python physics/phys_wp3_sindy.py       # reading the operator                   (~1 min)
python physics/phys_wp4_rollout.py     # extrapolation + hardening, 3 seeds     (~10 min)
python classification/gap_law.py       # gate law in bits                       (~45 min)
python classification/rate_relevance.py
python classification/symbolic_U.py    # (~3 min)
python tools/make_numbers.py           # -> paper/numbers.tex
python tools/make_results_notes.py     # -> results/RESULTS.md
cd paper && pdflatex operator_meters && pdflatex operator_meters
```

`classification/flob_lib/` vendors, unmodified, the S1/S2/S3 code of the companion repository *fixed-learned-both* (MIT). `recovered/` archives the material of the first (10-page, single-domain) draft of this paper and of the parked geometry study, kept for provenance.

## Licence
MIT.

## Cite
Archived at Zenodo: https://doi.org/10.5281/zenodo.23071565 (concept DOI, resolves to the latest version).
See `CITATION.cff` for a citable entry.