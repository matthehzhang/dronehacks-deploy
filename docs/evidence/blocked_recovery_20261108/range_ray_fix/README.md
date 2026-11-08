# Range-ray horizon correction

The local mapping path previously discarded valid returns beyond its 3 m mapping horizon twice: once in local scan extraction and again in `LocalGrid.integrate`. A finite 5 m or 8 m measurement can support bounded pass-through evidence before that horizon without placing an obstacle at 3 m. Missing and nonfinite returns cannot provide that evidence.

The intended regression boundaries are finite far-ray preservation, uncertainty-limited free carving, no artificial horizon obstacle, unchanged nearby-hit selection, rejection of nonreturns, and existing per-cell expiry. Adding many far points to an angular bin must not move the near-return quantile to the far surface or turn an insufficient near sample into a free-space certificate.

Candidate `50978085fedb6b3f56e1b23cb904167c15383eec` was copied to the active drone package before testing. The connection then dropped before saved-launcher update/hash verification; native staging resumed once the user reconnected USB-C. Device evidence records the exact state, rather than treating an rsync completion as a native build pass.

Validation completed on the laptop:

- Focused ray/age regressions: 36 passed.
- Full package suite: 576 passed, 2 expected failures; 195.24 seconds.
- Before/after extraction and mapping: old 5 m and 8 m scenes supplied zero beams/free cells; fixed scenes supplied 38 beams and 1532/1608 confidently free cells respectively, with no occupied cells. The 2.5 m scene is unchanged (1230 free, 36 occupied).
- Single-ray probe: far readings free 59 cells through 2.925 m, no artificial occupied endpoint; invalid and highly uncertain readings clear nothing; free evidence expires after 8 seconds.
- Patch replay reproduces all 11 adapted upstream files.
- Prior course/map assets and refnav authority/control code are unchanged by this fix.

Two simulations reuse DIVERSE01/DIVERSE02 and the unchanged prior, sensor profile and impairments. Their results are separate evidence; the source defect and regression results alone do not prove that this defect caused a specific real-flight landing.

## Focused validation

After the parent confirmed the active drone package rsync succeeded, 36 targeted tests passed on candidate `50978085fedb6b3f56e1b23cb904167c15383eec`. Device reconnection/final verification remained with the device agent. The offline probe used the actual production integrator and scan extractor:

| Input | Free cells after three updates | Occupied cells | Furthest free cell centre |
|---|---:|---:|---:|
| Valid 5 m return | 59 | 0 | 2.925 m |
| Valid 8 m return | 59 | 0 | 2.925 m |
| 1 m near return | 19 | 1 | 0.925 m |
| Exact 3 m return | 59 | 1 | 2.925 m |
| Zero / NaN / infinity / excessive uncertainty | 0 | 0 | — |

All free evidence expired after eight seconds. Adding 1,000 far points preserved the original 0.6 m near-return quantile. Tests also cover the actual camera origin, rotated offsets, no origin failing closed for far rays, the three-sigma plus cell uncertainty margin, and a single close point vetoing far clearing in its angular bin.

This establishes bounded ray interpretation, not successful navigation. Simulation results are recorded separately under `sim/` using the unchanged DIVERSE01/DIVERSE02 geometry.


The 5097808 comparison flights did not pass: DIVERSE01 and DIVERSE02 landed after
18.9 and 18.8 simulation seconds respectively, with zero checkpoints and zero points.
Both exhausted bounded recovery with `no observed-free rejoin beyond recovery lookahead`.
This fixes a demonstrated sensing-to-map defect but does not establish successful
mismatch recovery. Native flight traces were intact; an auxiliary observer started
after the flights and cannot provide missing in-flight statistics. See `sim/` for
source/world verification and outcome evidence.


Native device validation completed after USB-C reconnection: all 159 package files
match 5097808 in both active and external saved launcher copies. The main workspace
build passed in 6.10 seconds with one package; actual localizer constructors passed
at fit thresholds 0.35 and 0.42; the actual navigator constructed five legs with
three preplanned paths and three retained references; launch argument resolution
passed. The existing launcher was preserved. No physical restart or flight command
was issued. The last PX4 link check was absent, so these checks do not establish
physical flight readiness.
