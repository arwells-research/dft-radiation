# Success Register (Phase Evolution)

This register lists the **only items** that are considered “successes” for the
purpose of visibility in `tools/phase-evolution/`.

Anything not listed here is **non-canonical**, exploratory, or incomplete and
must not be treated as a validated DFT instrument.

An entry must include:

- a **checkable claim**,
- explicit **assumptions**,
- a minimal **reproduction path**,
- a clearly defined **expected output**,
- and an explicit **failure interpretation** (what breaks if the result fails).

---

## Template (copy for new entries)

## S-XXXX: <short title>

**Claim:**
<one sentence; must be checkable>

**Assumptions:**
- <A1>
- <A2>

**Repro:**
- Command(s): <...>
- Inputs: <...>
- Outputs: <...>

**Expected output:**
- <figure / CSV names and what constitutes “pass”>

**Failure interpretation:**
- If <observable mismatch>, then <assumption or projection rule is invalid>.

---

## Provenance header convention (CSV outputs)

All CSV outputs produced by registered S-000x toys begin with **comment header**
lines (each line starts with `#`) of the form:

    # key=value

At minimum, these keys are recorded:

- `git_commit` (or `nogit`)
- `run_utc`
- `toy_file`
- `source_sha256` (SHA-256 of the toy script text)

The CSV data begins immediately after these comment lines. To load in pandas:

    pd.read_csv(path, comment="#")

---

## Registered entries

## S-0001: κ₂ scaling classification (OU vs fractional Gaussian long-memory)

**Claim:**
For phase accumulation Δϕ(t)=∫ω(s)ds, the ensemble coherence envelope
c(t)=|⟨exp(iΔϕ(t))⟩| exhibits κ₂-driven scaling families that are audit-classifiable:
OU → κ₂(t)∝t (α≈1), fGn(H) → κ₂(t)∝t^{2H} (α≈2H).

**Assumptions:**
- ω is treated as a projection-driven stationary process for the purpose of this toy (no fundamental stochastics claimed).
- Envelope is evaluated by ensemble averaging at fixed t: c(t)=|⟨exp(iΔϕ(t))⟩|.
- Cumulant-dominant (Gaussian) regime: log(-log c(t)) vs log t is approximately linear over an audited window.

**Repro:**
- Command(s): `python toys/s0001_k2_scaling_demo.py`
- Inputs: none (seeded; parameters declared in-file DemoConfig)
- Outputs:
  - `toys/outputs/s0001_envelopes.csv`
  - `toys/outputs/s0001_envelopes.png`
  - `toys/outputs/s0001_scaling_diagnostic.png`
  - `toys/outputs/s0001_audit.csv`

**Expected output:**
- Audit banners report PASS for:
  - OU slope within tolerance of 1.0 in a declared late-time window
  - fGn slope within tolerance of 2H in a declared window
- Script exits with code 0 and writes the four artifacts above.

**Failure interpretation:**
- If OU slope cannot be made ≈1 in late time, the “diffusive long-time OU regime” is not being reached (grid/τc/ensemble too small) or the diagnostic fit is invalid.
- If fGn slope deviates persistently from 2H, the fGn generator or envelope computation is inconsistent with the claimed long-memory covariance structure.
- If r² thresholds are not met, the diagnostic window is not in a scaling regime (insufficient time-span or insufficient ensemble size).

---

## S-0002: κ₂ predictive closure and adjacency sensitivity (OU vs fGn)

**Claim:**
The coherence envelope c(t)=|⟨exp(iΔϕ(t))⟩| is predictively reconstructed by the
discrete κ₂ functional computed from the empirical autocovariance **if and only if**
the underlying process retains long-range adjacency (fractional Gaussian noise);
short-memory OU fails predictive closure without additional structural truncation.

**Assumptions:**
- Phase accumulation is defined as Δϕ(t)=∫₀ᵗ ω(s)ds, discretized as a cumulative sum with fixed Δt.
- Measured and predicted envelopes use **identical centering** of ω (declared in-file via `center_method`).
- κ₂ is computed via the discrete covariance sum:
  κ₂[j] = Δt² [ R(0)(j+1) + 2∑ₖ₌₁ʲ (j+1−k)R(k) ],
  not via a continuum approximation.
- Autocovariance lag support for fGn is sufficient to cover the audited time window (no artificial truncation within the window).
- Bartlett tapering, when enabled, is treated as an **estimator modification**, not part of the closure claim.

**Repro:**
- Command(s): `python toys/s0002_k2_prediction_demo.py`
- Inputs: none (seeded; parameters declared in-file via `DemoConfig`)
- Outputs:
  - `toys/outputs/s0002_envelope_compare.csv`
  - `toys/outputs/s0002_envelope_compare.png`
  - `toys/outputs/s0002_error_diagnostic.png`
  - `toys/outputs/s0002_audit.csv`

**Expected output:**
- Three audited subcases are produced:
  - **OU**: FAIL (large median/p95 error and strong positive drift).
  - **fGn_RAW** (no taper): PASS — median|Δlog c|, p95|Δlog c|, and drift slope all within declared tolerances over the audit window.
  - **fGn_TAPER** (Bartlett): FAIL or marginal — small absolute error but measurable positive drift due to taper-induced bias.
- Script exits with code 0 **iff fGn_RAW passes**.
- Audit CSV records all three subcases explicitly.

**Failure interpretation:**
- If **fGn_RAW** fails predictive closure, the κ₂ functional computed from the empirical autocovariance is insufficient to reconstruct the envelope even under long-memory adjacency, invalidating the claimed closure mechanism.
- If **OU** unexpectedly passes without structural truncation, the claim that short-memory processes require adjacency-aware modification is invalid.
- If **fGn_TAPER** shows time-dependent drift, this indicates estimator-induced bias rather than failure of κ₂ itself, and does not invalidate the closure claim.

---

## S-0003: Boundary of Gaussian sufficiency (κ₃ phase-bias diagnostic)

**Claim:**
When the phase-driving process exhibits a nonzero third cumulant (κ₃ ≠ 0) while
κ₂ is held fixed by construction, the **complex coherence** ⟨exp(iΔϕ(t))⟩ develops a
**sign-sensitive phase bias** (odd-cumulant signature) that is auditable and reversible
under κ₃→−κ₃, while the **magnitude envelope** c(t)=|⟨exp(iΔϕ(t))⟩| remains
κ₂-closure-consistent in the audited regime. This establishes a concrete boundary:
**κ₂ predicts magnitude**, but **κ₃ breaks phase-neutral Gaussian sufficiency**.

**Assumptions:**
- Phase accumulation is defined as Δϕ(t)=∫₀ᵗ ω(s)ds, discretized as a cumulative sum with fixed Δt.
- The κ₂ functional and magnitude-envelope prediction are identical to those used in S-0001 and S-0002:
  - κ₂(t) from empirical autocovariance R(k) via the discrete formula
  - c_pred(t)=exp(−κ₂(t)/2)
- Non-Gaussianity is introduced explicitly and minimally via a controlled κ₃ ≠ 0 perturbation **at the innovation level**, while preserving κ₂ (second-order structure) **by construction**.
- No claim is made that non-Gaussianity is fundamental; it is treated purely as a diagnostic perturbation.
- Over the audited window, κ₃ is detectable primarily through **phase bias** of the complex mean, not necessarily through magnitude-envelope deviation.

**Repro:**
- Command(s): `python toys/s0003_k3_gaussian_boundary_demo.py`
- Inputs:
  - Declared base process parameters (Gaussian reference)
  - Declared κ₃ injection mechanism and amplitude (innovation-level κ₃ with Var=1; optional κ₂-preserving mixture)
- Outputs:
  - `toys/outputs/s0003_envelope_compare.csv` (includes c_meas, c_pred, κ₂, and phase angle)
  - `toys/outputs/s0003_envelope_compare.png` (measured vs κ₂-predicted magnitude envelopes)
  - `toys/outputs/s0003_error_diagnostic.png` (|Δlog c(t)| diagnostic for magnitude closure)
  - `toys/outputs/s0003_phase_bias.png` (angle⟨exp(iΔϕ(t))⟩ phase diagnostic)
  - `toys/outputs/s0003_audit.csv` (closure + phase signature + residual checks)

**Expected output:**
Subcases are fixed: **GAUSS / K3+ / K3−**.

- **GAUSS (κ₃=0):**
  - κ₂ predictive closure for the **magnitude** envelope holds within declared tolerances over the audit window (**PASS**).
  - Phase bias is near zero (no coherent sign).

- **K3+ (κ₃>0):**
  - κ₂ predictive closure for the **magnitude** envelope still holds within tolerances (**PASS**).
  - Phase bias is **positive** with coherent sign across log-t bins (audited).

- **K3− (κ₃<0):**
  - κ₂ predictive closure for the **magnitude** envelope still holds within tolerances (**PASS**).
  - Phase bias is **negative** with coherent sign across log-t bins (audited).

Residual (structured, sign-sensitive) audit must PASS:
- median_sign(K3+) = −median_sign(K3−), both nonzero
- sign_fraction(K3±) ≥ declared `residual_sign_bin_frac` (coherence across bins)

**Failure interpretation:**
- If **GAUSS fails κ₂ magnitude closure**, the κ₂ prediction pipeline (autocovariance → discrete κ₂ → exp(−κ₂/2)) is invalid in this regime (or parameters/window insufficient).
- If **K3± do not show sign-sensitive phase bias**, κ₃ injection is not surviving to the complex envelope (or injection construction is incorrect).
- If **K3± are sign-sensitive but not coherent**, κ₃ exists but is too weak/noisy for certification under the current ensemble/window (increase n_trajectories and/or κ₃ amplitude).
- If **K3± break κ₂ magnitude closure**, κ₃ is coupling into magnitude under this construction (allowed as an additional observed mode, but not required for PASS).

**Interpretation boundary:**
This artifact certifies only that κ₃ can be introduced without altering κ₂ and that its observable footprint is a robust, sign-reversible **phase bias** in ⟨exp(iΔϕ(t))⟩. It does **not** claim κ₃ is fundamental, nor that κ₃ necessarily breaks κ₂-based prediction of the **magnitude** envelope in general—only that it breaks **phase-neutral Gaussian sufficiency** for the complex coherence.

---

## S-0004: Even-cumulant boundary certification (κ₄ magnitude collapse)

**Claim:**
With κ₂ held fixed by construction and a GAUSS baseline verifying pipeline identity,
introducing a controlled nonzero fourth cumulant (κ₄ ≠ 0) produces a **measured envelope
magnitude collapse** that κ₂-closure cannot explain. This certifies an auditable boundary
where **even-cumulant structure falsifies κ₂-only magnitude closure**.

**Assumptions:**
- Phase accumulation: Δϕ(t)=∫₀ᵗ ω(s)ds, discretized as a cumulative sum with fixed Δt.
- κ₂ prediction uses the same discrete κ₂ functional as S-0002:
  κ₂[j] = Δt² [ R(0)(j+1) + 2∑ₖ₌₁ʲ (j+1−k)R(k) ],
  and the Gaussian magnitude prediction c_pred(t)=exp(−κ₂(t)/2).
- GAUSS baseline must pass: sanity + identity + closure under the audited window.
- κ₄ injection is an explicit diagnostic perturbation; no claim of fundamental non-Gaussianity.

**Repro:**
- Command(s): `python toys/s0004_k4_even_cumulant_boundary_demo.py`
- Inputs: none (seeded; parameters declared in-file DemoConfig)
- Outputs:
  - `toys/outputs/s0004_envelope_compare.csv`
  - `toys/outputs/s0004_envelope_compare.png`
  - `toys/outputs/s0004_error_diagnostic.png`
  - `toys/outputs/s0004_phase_diagnostic.png`
  - `toys/outputs/s0004_audit.csv`

**Expected output:**
- **GAUSS (κ₄=0):** PASS (identity + closure + sanity in the audited window).
- **K4_* (κ₄ ≠ 0):** κ₂ magnitude closure FAIL, producing systematic collapse relative to κ₂ prediction.
- Script exits with code 0 **iff** GAUSS passes and the κ₄ failure is detected under declared certification criteria.

**Failure interpretation:**
- If GAUSS fails, the pipeline identity/audit window is invalid; S-0004 is not interpretable.
- If κ₄ does not cause measurable deviation, κ₄ injection is too weak for the current ensemble/window, or the perturbation is not surviving to Δϕ statistics.
- If κ₄ causes deviation but GAUSS also fails, you are in a numerical/noise-floor regime (discard; adjust window/ensemble or reduce sweep strength).

---

## S-0004-SENS: Boundary metric sweep and ρcrit bracketing

**Claim:**
The κ₄ boundary can be parameterized by the dimensionless ratio
ρ(t*) = |κ₄(t*)| / κ₂(t*)², and certified κ₂ failure can be bracketed by a reproducible
2-consecutive-failure rule, yielding [ρ_last_nonfail, ρ_first_fail] for the selected t*.

**Assumptions:**
- `gauss_ok` defines a validity gate: boundary inference is only performed where the GAUSS reference passes,
  preventing noise-floor misclassification.
- Classification uses four regimes: Invalid (Gray), Noise Floor (Blue), Transition (Orange), Failure (Red).
- Two consecutive Red points (sorted by ρ) define the first robust failure onset.
- The bracket is reported as:
  - `rho_last_nonfail`: nearest earlier valid Blue/Orange point
  - `rho_first_fail`: first Red in the 2-consecutive run

**Repro:**
- Command(s): `python toys/s0004_sensitivity_sigma_scale.py`
- Inputs: none (seeded; sweep range declared in-file)
- Outputs:
  - `toys/outputs/s0004_sensitivity.csv`
  - `toys/outputs/s0004_sensitivity.png`

**Expected output:**
- Sweep produces a mix of regimes (unless the chosen range does not cross the boundary).
- A bracket is reported when the 2-consecutive criterion is met.
- If no bracket is reported, the sweep did not cross the boundary (this is not a failure).

**Failure interpretation:**
- If no bracket and all points are Blue/Orange, increase sweep strength (e.g., `sigma_max` or equivalent).
- If all points are Gray, decrease sweep strength and/or increase ensemble size / adjust audit window.
- If bracket exists but transition is very wide, the boundary is weakly resolved; tighten by increasing ensemble size and/or selecting a better t* / audit window.

---

## S-0005: Cross-constraint closure (C1 ⟷ C2 ⟷ C5) under measurement kernels

**Claim:**
For Δϕ(t) = ∫ω(s)ds, the late-time scaling exponent α of
κ₂(t) = Var[Δϕ(t)] (fit from log κ₂ vs log t over a *declared* scaling window)
separates short-memory (OU) from long-memory baselines.
A C5 measurement kernel (windowing + finite detector response + additive noise)
applied to ω(t) may degrade estimator quality, but **cannot invert the regime
assignment** on any **admissible** case; i.e. it cannot “fake” a regime transition
when classification is restricted to its certified window.

**Assumptions:**
- ω(t) is treated as a stationary projection variable for this toy (no claim of
  fundamental stochastic dynamics).
- Two baselines are used:
  - OU (short-memory),
  - a long-memory surrogate with demonstrably larger κ₂ scaling exponent.
- The measurement kernel is applied in ω-space
  (moving-average window + causal exponential detector response + additive noise).
- ω is re-centered per trajectory, then integrated to Δϕ(t).
- Classification uses κ₂(t) scaling via a linear fit in (log t, log κ₂) over a
  **declared late-time audit window**.
- “Admissible” means **both regimes** satisfy declared r² and κ₂-adequacy
  thresholds in that window.
  Inadmissible cases are treated as **C5-dominated boundary cases**, not
  misclassification.

**Repro:**
- Command(s):
  - `python toys/s0005_c1_c2_c5_windowing_demo.py`
  - `python toys/s0005_c1_c2_c5_windowing_demo.py --write_outputs`
- Inputs: none (deterministic seed; parameters declared in-file via `S0005Config`)
- Outputs (untracked; written only with `--write_outputs`):
  - `toys/outputs/s0005_cases.csv`
  - `toys/outputs/s0005_audit.csv`

**Expected output:**
- Baseline audit prints show:
  - admissible high-r² κ₂ scaling fits for OU and long-memory,
  - α_LM > α_OU with gap ≥ declared minimum.
- Across the declared C5 kernel sweep:
  - some cases may become **C5-DOMINATED** (inadmissible due to bandwidth/SNR limits),
  - there are **zero MISCLASSIFIED admissible cases**.
- Script exits with code `0` iff:
  - the baseline separation is admissible,
  - ≥1 admissible swept case exists, and
  - no admissible case violates regime ordering.

**Failure interpretation:**
- If the baseline is not admissible (low r² or inadequate κ₂ in the audit window),
  the classifier is outside its certified regime; S-0005 is not interpretable.
- If any **admissible** case inverts the regime ordering, the measurement kernel
  is sufficient to fake a regime transition under the declared classifier,
  falsifying the C1/C2/C5 closure claim.
- If too few cases are admissible, the sweep is **C5-dominated**; this is a
  measurement boundary condition and is reported as **INCONCLUSIVE**, not as
  a regime transition.

---

## S-0006: Radiation ω generator families (C1 ⟷ C2 ⟷ C3 ⟷ C4)

**Claim:**
For Δϕ(t) = ∫ω(s)ds, the late-time scaling exponent α of
κ₂(t) = Var[Δϕ(t)] (fit from log κ₂ vs log t over a *declared* scaling window)
is **measurable and stable** across a finite, declared menu of ω(t) generator
families representing phenomenological bindings of C1–C4.

Linear transport-style filtering (C3) and event-driven interface injection (C4)
may alter bandwidth and increment statistics of ω, but **do not change the κ₂
scaling class** or violate pre-declared α expectations on any **admissible** case.

**Assumptions:**
- ω(t) is treated as a stationary projection variable for this toy
  (no claim of fundamental stochastic dynamics).
- Two base regimes are used:
  - OU (short-memory; diffusive κ₂ scaling),
  - fGn (long-memory; superdiffusive κ₂ scaling with fixed H).
- C3 is represented by linear, time-invariant FIR smoothing applied per trajectory.
- C4 is represented by event-driven (shot-like) injections applied directly in ω.
- ω is centered only by **global DC removal** (ensemble-wide mean);
  no per-trajectory or per-time re-centering is applied.
- Classification uses κ₂(t) scaling via a linear fit in (log t, log κ₂) over a
  **declared late-time audit window**.
- “Admissible” means each generator family satisfies declared r² and κ₂-adequacy
  thresholds in that window.
  Inadmissible cases are treated as **boundary conditions**, not misclassification.

**Repro:**
- Command(s):
  - `python toys/s0006_radiation_omega_generators.py`
  - `python toys/s0006_radiation_omega_generators.py --write_outputs`
- Inputs: none (deterministic seed; parameters and expectations declared in-file via
  `S0006Config`)
- Outputs (untracked; written only with `--write_outputs`):
  - `toys/outputs/s0006_cases.csv`
  - `toys/outputs/s0006_audit.csv`

**Expected output:**
- For each declared generator family:
  - an admissible high-r² κ₂ scaling fit exists in the audit window,
  - the fitted α lies within the **pre-declared expectation range** for that family.
- Certified families include:
  - OU,
  - OU + C3,
  - OU + C4,
  - OU + C3 + C4,
  - fGn,
  - fGn + C3,
  - fGn + C4,
  - fGn + C3 + C4.
- Script exits with code `0` iff:
  - all generator families are admissible, and
  - no admissible family violates its declared α expectation.

**Failure interpretation:**
- If any generator family is not admissible (low r² or inadequate κ₂ in the audit
  window), the result is **INCONCLUSIVE**: the scaling classifier is outside its
  certified regime for that construction.
- If any **admissible** generator violates its declared α expectation, the binding
  (C3/C4) is sufficient to fake a κ₂ scaling class, falsifying the C1/C2/C3/C4
  invariance claim.
- If all families are admissible and consistent with expectations, C3 and C4 are
  certified as **non-regime-altering bindings** under κ₂-based classification.

---

## Interpretation boundary

Entries in this register certify **diagnostic correctness and reproducible behavior only**.

They do **not** constitute claims that:
- ω(t) is fundamentally stochastic,
- OU or fGn processes are ontologically primary,
- κ₂ closure holds universally outside the audited regimes,
- observed failures imply physical impossibility rather than structural mismatch.

Each success establishes that **given the declared assumptions and constructions**, a specific observable relation:
- is testable,
- is reproducible,
- and fails in a *diagnostically interpretable* way when assumptions are violated.

No extrapolation beyond the explicitly audited window, estimator choice, or process class is implied.
