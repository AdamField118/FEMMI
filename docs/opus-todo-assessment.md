# Opus TODO: item-by-item disposition

This records the completed calibration patch. The subsequent production-mapper
cleanup removes the compatibility defaults mentioned below; see [Catalogue mapper](mapper.md).

Base: `8cb095ccd6333b057a7c931935b52bf42d035450` (main, rechecked during the task).
Source: the recovered `Pasted text.txt`, containing 17 numbered items.
Author for the delivered patch: **AdamField118 <adfield@wpi.edu>**.

All 17 questions are worth addressing. The code, synthetic experiments, and
numerical measurements are included. Two external-data experiments remain
blocked by missing inputs, as stated below; neither is presented as completed.
No changes were committed or pushed, and no branch was created.

| # | Assessment and action | Outcome / remaining limit |
|---:|---|---|
| 1 | Essential. Jointly search KS grid and smoothing on held-out catalogues, expanding winning edges. | Calibrated KS is in every baseline/scope comparison; all candidate scores are saved. Historical API anchors remain explicitly labeled. |
| 2 | Essential. Replace marginal “sigma” with paired differences and intervals. | Pairing verifies catalogue hashes and scenario/density/seed identity; duplicates and ambiguous prefixes fail. Missing pairs, wins, ties, SE, t intervals, and exact sign p values are recorded. |
| 3 | Essential. Jointly calibrate λ and physical correlation length per FEM kind. | Shared `M + length² K` family; independent P3/Argyris/HCT searches, adaptive expansion and local refinement. Finite search does not prove a global optimum. |
| 4 | Essential. Regenerate comparisons after all changes. | Raw candidates, held-out catalogues, maps, paired statistics, figures, and tables are retained. Historical headline sections are replaced, not mixed with the corrected results. |
| 5 | Essential to interpreting density. Sweep component noise 0.05/0.10/0.20/0.30 at fixed count density 10. | Every noise scenario is recalibrated. No survey performance is inferred from count density alone. |
| 6 | Worth doing. Off-centre halo, two halos, and changed mass/concentration. | Separate held-out calibration and six evaluation seeds per scenario. This is a small set of scope tests, not a complete halo population study. |
| 7 | Worth doing. Radius 1.5/3/6 arcmin at count density 5. | Count follows gross field area; area and density are not conflated. |
| 8 | Essential. HCT requires its own calibration. | HCT gets an independent joint λ/length search in every scenario, rather than inheriting Argyris's setting. |
| 9 | Essential for catalogues. Apply nonbinary weights consistently. | Mean-one lognormal weights, noise σ/√w, weighted FEM and KS, and weighted effective density are exercised. Legacy tuple/unweighted-path inconsistencies are repaired. |
| 10 | Worth doing, **external benchmark blocked**. MassiveNuS map files were not supplied. | Added a split-directory config and content-hash overlap rejection. Fixed the loader to preserve the actual patch mean instead of shifting the minimum to zero. Synthetic loader tests pass; no MassiveNuS science result is claimed. Independent lines of sight and the correct angular/redshift metadata are still required. |
| 11 | Worth doing as an opt-in numerical experiment. A Gram condition number alone was insufficient. | Assemble dual P0 single-layer and continuous P1 hypersingular operators, singular quadrature, constant stabilization, and both mixed-Gram inverse factors. Test the full positive spectrum under refinement and perturbed spacing. This does not automatically precondition the production high-order coupled system. |
| 12 | Worth measuring; an unconditional DOF gate is **not justified**. | Measure actual SuperLU factor/solve and ILU/GMRES setup/solve costs through 9409 DOFs and five RHS, checking residuals. Cold setup and repeated solves have different trade-offs; the production builder already owns an LU. No universal crossover or automatic switch is claimed. Larger target-hardware measurements remain appropriate. |
| 13 | Worth remeasuring; keep ACA off. | Warm dense/ACA assembly timings and matrix errors cover cubic/quintic circles and catalogue rings after CPU acceleration. Dense wins in the tested cases. Raw repeated timings are retained; hardware/load dependence is explicit. |
| 14 | Worth doing, **real-cluster experiment blocked**. Registered lensing/shear and Chandra data were not supplied. | Implement/test annular Fourier cross-coherence, explicit resolution/FOV censoring, and sample propagation without Fisher linearization. Actual registration, PSF/window treatment, posterior validation, and comparison with a parametric ensemble remain necessary. This is diagnostic infrastructure, not a replication of Cerini et al. |
| 15 | Worth doing. Replace irregular letter suffixes with one ordered scheme. | Sections are now 18.3.1–18.3.18; source/doc references updated. `math-section-map.json` records the checked old-to-new mapping, including the old h-before-g order. |
| 16 | Worth doing after results. Surface the corrected comparison and limits. | README, examples, mathematical discussion and MkDocs calibration page point to generated results and reproducible commands. Unsupported density factors, mask-advantage statements and stale significance claims are retired. |
| 17 | Worth doing. Separate concerns while preserving imports. | `femmi.density` is now a package with `sampling`, `runners`, `sweep`, and `stats`, preserving public reexports. |

## Additional correctness findings necessary for fair calibration

- Generating C¹ noise on guard vertices changed the second component's random
  stream. Catalogues are now generated before meshes and shared by every arm.
- Prematurely stopped L-BFGS fits can make a parameter or element look better
  or worse. Quadratic MAP is solved with prior-preconditioned CG and independently
  checked residuals, rather than accepting an optimizer success flag alone.
- Mixed C¹ value/derivative units caused coupled-LU roundoff: one test catalogue
  gave a fresh normal-equation residual near 7e-6 despite CG success. Diagonal
  congruence equilibration reduced it to about 3e-9; a separate refinement
  calculation agreed with the reconstruction. Forward and transpose scaling
  are matched, preserving adjoints and the MAP equation.
- Clustering could move sources back into masked holes. Cluster perturbations
  now respect both the outer field and every mask.
- A MassiveNuS minimum shift is not restoration of a simulation's mass sheet.
  Truth construction now retains the sampled map's actual mean.

## Evidence and reproduction

`docs/calibration.md` defines the observation, solver and statistical contract.
`benchmarks/calibration/configs.json` defines the 16 synthetic scenarios.
`benchmarks/calibration/RESULTS.md` and `audit.json` report the accepted results
and failures/edges. `examples/paper/report_calibration.py` regenerates summaries
without choosing parameters from evaluation truth.

`examples/diagnostics/numerical_followup.py` and
`benchmarks/calibration/numerics/numerics.json` contain the BC, iterative and
ACA measurements. Timing under concurrent experiment load is illustrative;
replay serially on publication hardware before making a speed claim.

`benchmarks/calibration/VALIDATION.md` records the executed test commands and
patch validation. The release still needs the separately planned full SMPy
benchmark, real-data evaluation, and publication-level replication.
