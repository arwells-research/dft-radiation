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

## S-0007: Mixture + crossover “no-wrong-claims” certification (OU ⊕ fGn; early/late windows)

**Claim:**  
For ω(t) constructed as a **declared linear mixture** ω = ω_OU + ε·ω_fGn over a
**finite declared ε grid**, the κ₂-slope classifier (log κ₂ vs log t over **two declared
windows**) is **restrained**:
it does **not** make **wrong regime claims** on any **admissible** window, and it
returns **BOUNDARY** (no claim) whenever mixture/crossover effects prevent a
clean regime identification inside the declared windows.

This success certifies **classifier restraint under mixture**, not that the mixture
must realize an fGn-dominant late-time scaling regime within the finite run horizon.

**Assumptions:**  
- ω(t) is treated as a stationary projection variable for this toy (no fundamental
  stochastics claimed).
- Mixture family is **declared and finite**: ω = ω_OU + ε·ω_fGn with ε ∈ `eps_list`
  (no scanning).
- Optional bindings are declared and finite:
  - C3: per-trajectory linear FIR smoothing,
  - C4: event-driven shot injection into ω,
  - fixed application order: **C4 then C3** (when both enabled).
- Centering is **global DC removal only**: subtract one constant mean across
  ensemble×time; **no per-trajectory** and **no per-time** centering.
- Phase accumulation is defined as Δϕ(t)=∫ω(s)ds, discretized as a cumulative sum
  with fixed Δt.
- κ₂(t)=Var(Δϕ(t)) is computed by ensemble variance at fixed t.
- Window fits are restricted to **two declared windows** (early, late); no scanning
  or adaptive windowing.
- Window admissibility uses the same criteria as S-0006:
  minimum r², κ₂ adequacy at window end, and minimum point count.
- A drift guard may be applied as an **additional admissibility veto** for mixture
  cases (ε>0) to prevent fragile “claims” under slow trend/offset artifacts; the
  ε=0 baseline is still required to be OK in both windows.

**Repro:**  
- Command(s):
  - `python toys/s0007_mixture_crossover_audit.py`
  - `python toys/s0007_mixture_crossover_audit.py --write_outputs`
- Inputs: none (deterministic seed; parameters and expectations declared in-file via
  `S0007Config`)
- Outputs (untracked; written only with `--write_outputs`):
  - `toys/outputs/s0007_cases.csv`
  - `toys/outputs/s0007_audit.csv`

**Expected output:**  
A finite declared set of cases (generator families × ε grid) is evaluated.
Each case produces **two window tags**: `early_tag` and `late_tag`.

Tag semantics are **directional** (designed to prevent false claims):
- **OK**: window admissible AND fitted α lies in the expected range.
- **BOUNDARY**: window inadmissible OR admissible but α lies outside both expected
  and forbidden ranges (interpreted as mixture/crossover/mixed scaling → *no claim*).
- **FAIL**: window admissible AND α lies in the **forbidden range**
  (i.e. a wrong regime is explicitly claimed).

Declared expectations:
- **ε = 0 (pure OU):**
  - Early window: must be **OK (OU)**.
  - Late window: must be **OK (OU)**.
  - Any **BOUNDARY** or **FAIL** ⇒ overall **FAIL**.

- **ε > 0 (mixture):**
  - Early window:
    - **OK (OU)** or **BOUNDARY** are allowed.
    - **FAIL** (claiming fGn) is forbidden.
  - Late window:
    - **OK (fGn)** is allowed when it occurs, but is **not required**.
    - **BOUNDARY** is allowed (interpreted as mixed/crossover scaling → no claim).
    - **FAIL** (claiming OU when fGn is expected) is forbidden only when the window
      is admissible and lands inside the fGn expectation range; otherwise the safe
      outcome is **BOUNDARY**.

Exit codes:
- `0` — AUDIT PASS: ε=0 baseline is OK in both windows, and there are **zero FAIL tags**
  across all admissible windows (no wrong claims).
- `2` — AUDIT FAIL: one or more admissible windows make a forbidden claim (FAIL), or
  the ε=0 baseline is not OK in both windows.

**Failure interpretation:**  
- If any **admissible** window produces a **FAIL**, the κ₂-slope classifier is capable
  of making a wrong regime claim under mixture/crossover, falsifying the restraint
  guarantee.
- If ε>0 cases are predominantly **BOUNDARY** in late windows, this indicates that the
  declared mixture does not realize a clean late-time scaling regime within the finite
  horizon / window choice; this is a **certified no-claim boundary**, not a regime
  transition.
- If ε=0 cases fail OU classification, the baseline κ₂-scaling pipeline is invalid
  under the declared discretization or audit windows.

**Interpretation boundary:**  
This success certifies **classifier restraint**, not regime separability.
It establishes that κ₂-based scaling diagnostics:
- correctly identify OU when the OU regime is clean and admissible,
- and **do not over-interpret** mixture/crossover dynamics as a false regime.

No claim is made that κ₂ scaling resolves superposition, interference,
or nonclassical linear combination at the quantum level.

---

## S-0008: Drift-limited vs diffusion-limited coherence (C2 ⟷ C5 boundary)

**Claim:**  
Within a fixed late-time audit window, the κ₂-slope classifier (log κ₂ vs log t)
distinguishes **diffusion-limited decoherence** (OU-like, α≈1) from a **drift-artifact**
construction (OU plus slow per-trajectory drift), producing **OK for diffusion**
and **BOUNDARY for drift** on admissible data without making any forbidden regime claims.

This extends the classifier-restraint program (S-0007) by certifying a C5-style
artifact axis: slow drift that can contaminate κ₂-slope inference unless the
classifier is restrained.

**Assumptions:**

- Phase accumulation remains defined as Δϕ(t)=∫₀ᵗ ω(s)ds (discrete cumulative sum).
- Two generator families are declared:
  - **Diffusion baseline:** OU process (short-memory; diffusive κ₂ scaling).
  - **Drift regime:** the same OU baseline with an added slow drift with **per-trajectory**
    slope βᵢ, i.e. ωᵢ,drift(t)=βᵢ·t with βᵢ drawn once per trajectory from a declared
    distribution (seeded), prior to global DC removal.
- Centering is **global DC removal only**, identical across regimes.
- Audit window `[t_min, t_max]` and admissibility criteria are **fixed in advance**
  (minimum r², κ₂ adequacy at window end, minimum point count).
- No adaptive fitting, scanning, or parameter tuning is permitted.

**Repro:**

- Command(s):  
  `python tools/phase-evolution/toys/s0008_drift_vs_diffusion_demo.py`
- Inputs: none (deterministic seed; parameters declared in-file via `S0008Config`)
- Outputs: none by default (see below)

Optional outputs (untracked; written only with `--write_outputs`):
- `tools/phase-evolution/toys/outputs/s0008_cases.csv`
- `tools/phase-evolution/toys/outputs/s0008_audit.csv`

**Expected output:**

Two canonical subcases are audited:

| Subcase | Expected tag | Meaning |
|--------|--------------|--------|
| **DIFFUSION** | OK | Window admissible; fitted κ₂ slope lies within OU expectation band (α≈1). |
| **DRIFT** | BOUNDARY | Window admissible but slope lies outside OU band; classifier refuses to claim. |

Exit codes:

- `0` → PASS if diffusion is correctly classified and drift is safely rejected.
- `2` → FAIL if any admissible case makes a forbidden claim (e.g., drift tagged as diffusion).
- `3` → INCONCLUSIVE if no cases are admissible.

**Failure interpretation:**

- If the diffusion baseline fails → κ₂-slope classifier is invalid in the declared audit window.
- If the drift case is tagged OK (diffusion) on admissible data → κ₂-slope diagnostics
  cannot safely separate C2 diffusion from C5-style drift artifacts, falsifying the restraint claim.
- If both cases are inadmissible → the result is **INCONCLUSIVE** (measurement/finite-horizon dominated),
  not a regime inversion.

**Interpretation boundary:**

This entry certifies only that κ₂-based scaling diagnostics:
- correctly recognize diffusion-like OU behavior when admissible, and
- refuse drift-contaminated constructions by returning **BOUNDARY** rather than making a wrong claim.

No claim is made that drift is fundamental; it is treated purely as a diagnostic boundary condition.

## S-0009: Higher-order odd-cumulant boundary (κ₅ phase-structure diagnostic)

**Claim:**
For phase accumulation Δϕ(t)=∫ω(s)ds, introducing a controlled nonzero fifth cumulant (κ₅ ≠ 0)
at the **innovation level** produces a **sign-sensitive, auditable phase bias** in the complex
coherence ⟨exp(iΔϕ(t))⟩ that reverses under κ₅→−κ₅, while the **magnitude envelope**
|⟨exp(iΔϕ(t))⟩| remains consistent with an **even-cumulant truncation** (κ₂, κ₄) over the
declared audit window.

**Assumptions:**
- Phase accumulation is defined as Δϕ(t)=∫₀ᵗ ω(s)ds, discretized as a cumulative sum with fixed Δt.
- ω(t) is generated as i.i.d. innovations (stationary, memoryless) purely for diagnostic certification;
  no claim of fundamental stochastics is made.
- Subcases are constructed to isolate κ₅ while avoiding κ₃:
  - **GAUSS** uses standard normal innovations.
  - **K5+** uses a fixed discrete innovation distribution with mean=0, Var=1, μ₃=0, μ₄=3 (⇒ κ₄=0) and μ₅>0.
  - **K5−** is constructed as the **exact sample-level sign flip** of K5+ (ω_K5− := −ω_K5+), ensuring κ₅→−κ₅
    and eliminating finite-sample sign drift as a confound.
- κ₂(t), κ₄(t), κ₅(t) are evaluated as **empirical cumulants of Δϕ(t)** (ensemble cumulants at fixed t).
- The complex coherence is evaluated as Z(t)=⟨exp(iΔϕ(t))⟩ by ensemble averaging at fixed t.
- Magnitude-closure audit compares |Z_meas(t)| to an even-cumulant truncation |Z_pred(t)| derived from
  the cumulant-truncated model (no fitting, no scanning; all cumulants are empirical):
  log Z(t) ≈ −κ₂(t)/2 + κ₄(t)/24 + i κ₅(t)/120
  (magnitude closure is evaluated on log|Z|; phase audit uses sign of sin(angle(Z)) in-window).
- Audit is performed only over a **declared fixed window**; no adaptive windowing or parameter tuning.

**Repro:**
- Command(s): `python toys/s0009_k5_higher_cumulant_boundary_demo.py`
- Inputs: none (deterministic seed; constructions declared in-file).
- Outputs (written only with `--write_outputs`):
  - `toys/outputs/s0009_envelope_compare.csv`
  - `toys/outputs/s0009_phase_bias.png`
  - `toys/outputs/s0009_error_diagnostic.png`
  - `toys/outputs/s0009_audit.csv`

**Expected output:**
Subcases are fixed: **GAUSS / K5+ / K5−**.

- **GAUSS:** magnitude-closure diagnostic passes in the declared window; phase bias is near zero.
- **K5+:** magnitude-closure diagnostic passes; phase bias has a coherent **positive** sign.
- **K5−:** magnitude-closure diagnostic passes; phase bias has a coherent **negative** sign.

Residual (sign) audit must PASS (when phase-sign admissibility holds):
- median_sin(K5+) = −median_sin(K5−), both nonzero.
- sign_fraction(K5±) ≥ declared `residual_sign_bin_frac`.

**Failure interpretation:**
- If GAUSS fails magnitude closure, the cumulant-based prediction/audit pipeline is invalid in this regime.
- If K5± do not show sign-sensitive phase bias (or are not sign-coherent), κ₅ injection is not surviving
  to Δϕ statistics at the chosen ensemble/window (insufficient signal) **or** the phase-sign observable is
  coherence-floor limited.
- If K5± break magnitude closure while GAUSS passes, the construction is coupling odd-cumulant structure
  into magnitude beyond the declared truncation boundary, falsifying the “phase-structure-only” boundary claim.
- If the complex-mean magnitude falls below the declared coherence floor across the window, the result is
  **INCONCLUSIVE** (noise-floor limited; no sign claim permitted).

**Interpretation boundary:**
This artifact certifies only that controlled κ₅ introduces a reproducible, sign-reversible **phase-structure**
boundary beyond κ₄ under the declared construction and window. It does not claim κ₅ is fundamental—only that
odd-cumulant structure can survive into Z(t)=⟨exp(iΔϕ(t))⟩ as a detectable phase bias without requiring
magnitude collapse.

---

## S-0010: κ₂-slope non-identifiability for C3 vs stationary C4 (C3 ⟷ C4 boundary)

**Claim:**
Under the DFT ω→Δϕ→κ₂ program, **late-window κ₂-slope (α) is invariant** under:

- **pure geometric transport (C3)** modeled as deterministic linear filters, and
- a declared class of **stationary, finite-variance interface injections/exchanges (C4)**

when evaluated in a fixed admissible window.

Therefore, **κ₂-slope alone is not sufficient to discriminate C3 vs stationary C4**
within this construction: transport-only and stationary interface effects can be
**κ₂-slope-indistinguishable** even when both are present and admissible.

This establishes a categorical **identifiability boundary** (refusal logic):
a change (or lack of change) in κ₂-slope must not be over-interpreted as uniquely
diagnosing C3 vs C4 without additional diagnostics beyond κ₂-slope.

**Assumptions:**
- Phase accumulation is defined as Δϕ(t)=∫₀ᵗ ω(s)ds (cumulative sum with fixed Δt).
- Baseline ω(t) is generated as OU diffusion with fixed θ and σ.
- Transport effects (C3) are modeled as deterministic linear filters applied per-trajectory
  (no randomness added by the filter itself).
- Interface effects (C4) are modeled as **stationary, finite-variance** processes acting on ω(t)
  (shot-like injection and a gain/loss variant as implemented in-file).
- κ₂(t) is computed as Var(Δϕ(t)) across the ensemble at each t, and α is the log-log slope
  of κ₂ vs t within a **declared fixed window**.
- No adaptive windows, parameter fitting, scanning, or tuning is permitted.

**Decision rule (invariance test):**
- Compute α and r² for each case in the fixed window.
- Let (α₀, r²₀) be the OU_BASE reference.
- A case is **OK** iff it is admissible and:
  - |α − α₀| ≤ `max_abs_delta_alpha`, and
  - |r² − r²₀| ≤ `max_abs_delta_r2`.
- If any required case is admissible but violates the invariance bounds → **FAIL**.
- If OU_BASE or any required case is inadmissible in-window → **INCONCLUSIVE**.

**Repro:**
- Command(s): `python toys/s0010_c3_vs_c4_k2_transport_interface_demo.py`
- Inputs: none (deterministic seeds declared in-file).
- Outputs (only with `--write_outputs`):
  - `toys/outputs/s0010_cases.csv`
  - `toys/outputs/s0010_audit.csv`

**Expected behavior:**

| Case | Interpretation | Expected behavior | Verdict |
|------|----------------|------------------|---------|
| OU_BASE | baseline diffusion | reference (α₀, r²₀) | OK |
| OU_C3_SMOOTH | geometric transport (LPF-like) | α≈α₀ (within Δ bounds) | OK |
| OU_C3_DISP | stronger dispersive-like filtering | α≈α₀ (within Δ bounds) | OK |
| OU_C4_SHOT | stationary interface injection | α≈α₀ (within Δ bounds) | OK |
| OU_C4_GAINLOSS | stationary interface exchange variant | α≈α₀ (within Δ bounds) | OK |

**Failure interpretation:**
- If a C3 transport case violates invariance → the C3 invariance claim fails in this regime.
- If a stationary C4 case violates invariance → κ₂-slope is more sensitive than claimed (boundary shifts),
  or the C4 construction is not stationary/finite-variance as assumed.
- If OU_BASE is inadmissible or coherence/variance gates fail in-window → **INCONCLUSIVE** (no claim).

**Interpretation boundary:**
This artifact certifies a **negative result**: κ₂-slope does **not** uniquely separate C3 from
stationary C4 under the declared constructions and window. It does not claim any physical ontology
for ω(t); it only establishes refusal logic for C3/C4 attribution based on κ₂-slope alone.

---

## S-0011: C3 vs C4 separability via ω non-Gaussianity (κ₄ / kurtosis) under κ₂-slope invariance

**Claim:**
Given the S-0010 identifiability boundary (κ₂-slope alone cannot separate C3 vs stationary C4),
a second, directly-audited statistic on ω can provide separability within a declared family:

- **C3 transport** modeled as deterministic linear filtering of a Gaussian OU baseline
  preserves Gaussianity (all cumulants above second vanish in expectation), so **excess kurtosis**
  of ω remains near zero.
- **C4 interface injection** modeled as stationary shot-like injection produces a
  heavy-tailed ω distribution, yielding **positive excess kurtosis** in ω,
  even when κ₂-slope in the declared window remains invariant.

Therefore, within the declared constructions and fixed window, **ω excess kurtosis**
(or equivalently κ₄(ω)) provides an auditable C3 ⟷ C4 discriminator that κ₂-slope lacks.

**Assumptions:**
- Phase accumulation is defined as Δϕ(t)=∫₀ᵗ ω(s)ds (cumulative sum with fixed Δt).
- Baseline ω(t) is generated as Gaussian OU diffusion with fixed θ and σ.
- C3 is instantiated only as deterministic linear filtering applied per-trajectory.
- C4 is instantiated as a stationary shot-like injection process acting on ω(t),
  with fixed event rate, amplitude distribution, and decay kernel.
- The audit window is fixed in absolute simulation time; no scanning/tuning.
- ω is globally DC-centered (single constant over ensemble×time) prior to statistics.

**Decision rules (fixed window):**

1) **κ₂-slope invariance gate (reuse S-0010 semantics):**
   - Compute κ₂(t)=Var(Δϕ(t)) and fit α in the declared window.
   - Compare each case to OU_BASE with declared bounds:
     - |α−α₀| ≤ `max_abs_delta_alpha`
     - |r²−r²₀| ≤ `max_abs_delta_r2`
   - If OU_BASE is inadmissible, or any required case is inadmissible → **INCONCLUSIVE**.

2) **ω non-Gaussianity separation (window-restricted ω samples):**
   - Pool ω samples within the declared window across all trajectories.
   - Compute excess kurtosis:
     g2 := E[ω⁴]/(E[ω²]²) − 3
   - Requirements:
     - **OU_BASE, OU_C3_SMOOTH, OU_C3_DISP** must satisfy:
       |g2| ≤ `max_abs_excess_kurtosis_gauss`
     - **OU_C4_SHOT** must satisfy:
       g2 ≥ `min_excess_kurtosis_shot`
   - If the variance floor is too small (E[ω²] below a declared minimum) → **INCONCLUSIVE**.

**Repro:**
- Command(s): `python toys/s0011_c3_c4_kurtosis_separation_demo.py`
- Inputs: none (deterministic seeds declared in-file).
- Outputs (only with `--write_outputs`):
  - `toys/outputs/s0011_cases.csv`
  - `toys/outputs/s0011_audit.csv`

**Expected behavior:**

| Case | Interpretation | κ₂-slope vs base | ω excess kurtosis g2 | Verdict |
|------|----------------|------------------|----------------------|---------|
| OU_BASE | Gaussian OU | invariant | ~0 | OK |
| OU_C3_SMOOTH | C3 transport | invariant | ~0 | OK |
| OU_C3_DISP | stronger C3 transport | invariant | ~0 | OK |
| OU_C4_SHOT | C4 stationary injection | invariant | >0 | OK |

**Failure interpretation:**
- If any C3 case violates the Gaussian-kurtosis bound while κ₂ invariance holds,
  the C3 “no new cumulants” assumption is violated (implementation error or non-Gaussian baseline).
- If OU_C4_SHOT fails to exceed the kurtosis minimum while κ₂ invariance holds,
  the declared C4 injection is not producing a detectable heavy-tail signature in this regime.
- If κ₂ invariance fails, S-0011 is **invalid** (it relies on S-0010’s invariance gate);
  report as **FAIL** (wrong regime) or **INCONCLUSIVE** if inadmissible.

**Interpretation boundary:**
This artifact certifies only that, under the declared Gaussian-baseline + deterministic C3
and stationary shot-like C4 constructions, **ω non-Gaussianity (κ₄ / kurtosis)** provides an
auditable C3 ⟷ C4 discriminator while κ₂-slope does not. It does not claim kurtosis is a
universal interface signature beyond the declared family.

---

## S-0012: Attempted κ₆ even-cumulant magnitude boundary (no-go under declared injection)

**Claim (as certified by this toy):**  
Under the declared construction, introducing rare symmetric “large-increment” events into i.i.d.
innovations does **not** produce a sign-controlled or reliably separated κ₆(t) between K6+ and K6−,
and does **not** induce a magnitude-envelope deviation beyond κ₂+κ₄ closure in the declared window.

Therefore, this toy certifies a **no-go / boundary-not-reached** result:
a naive symmetric rare-event injection is **insufficient** to establish a κ₆-driven
magnitude sufficiency boundary under the current audit regime.

**Observed outcome (2026-01-18 run):**
- GAUSS / K6+ / K6− all PASS the κ₂+κ₄ magnitude-closure audit.
- κ₆(t*) is nonzero (finite-sample) but is **not sign-controlled** across K6± in this construction.

**Assumptions:**
- Phase accumulation is Δϕ(t)=∫₀ᵗ ω(s)ds (cumulative sum with fixed Δt).
- ω innovations are i.i.d. and diagnostic only.
- K6± are generated by the same baseline plus rare symmetric large increments (as coded).
- κ₂(t), κ₄(t), κ₆(t) are evaluated as empirical cumulants of Δϕ(t) across the ensemble at fixed t.
- Prediction uses the even-cumulant truncation for magnitude:
      log c_pred(t) = −κ₂(t)/2 + κ₄(t)/24
- Fixed audit window; no scanning / tuning.

**Repro:**
- Command: `python toys/s0012_k6_even_cumulant_boundary_demo.py`
- Inputs: none (deterministic seed; constructions declared in-file).
- Outputs (only with `--write_outputs`):
  - `toys/outputs/s0012_cases.csv`
  - `toys/outputs/s0012_audit.csv`

**Failure interpretation:**
- If GAUSS fails → κ₂+κ₄ magnitude-closure pipeline is invalid in this regime.
- If κ₆ becomes sign-controlled and K6± depart from magnitude closure → that would indicate a
  genuine higher-order even-cumulant magnitude boundary (not observed here).
- If coherence falls below the declared floor → INCONCLUSIVE.

**Interpretation boundary:**
This artifact does **not** certify a κ₆ magnitude boundary; it certifies that this particular
κ₆-injection attempt does not reach such a boundary under the declared regime.
A future κ₆ boundary attempt must use a construction that actually controls κ₆ sign/magnitude
at the Δϕ level (likely requiring a different innovation family and/or calibrated rarity/amplitude),
and should be introduced as a new S-item rather than mutating this one.

---

## S-0013: κ₆ even-cumulant magnitude sufficiency boundary (κ₂+κ₄ truncation failure)

**Claim:**  
There exists a certified regime in which the **magnitude envelope**
c(t)=|⟨exp(iΔϕ(t))⟩| is **not explainable** by an even-cumulant truncation that retains
(κ₂, κ₄) but omits κ₆. In particular, for controlled heavy-tail innovations producing
large κ₆ at the Δϕ level, the κ₂+κ₄ magnitude predictor fails **categorically** in a
declared audit window, while the GAUSS baseline continues to pass.

This establishes a **next even-cumulant sufficiency boundary beyond S-0004**:
κ₄ is not the last magnitude-relevant cumulant; sufficiently large κ₆ breaks
κ₂+κ₄ magnitude closure.

**Assumptions:**
- Phase accumulation: Δϕ(t)=∫₀ᵗ ω(s)ds, discretized by cumulative sum with fixed Δt.
- ω innovations are i.i.d. (stationary, memoryless) for diagnostic certification only.
- κ₂(t), κ₄(t), κ₆(t) are evaluated as **empirical cumulants of Δϕ(t)** across the ensemble at fixed t,
  using central-moment cumulant formulas (κ₆ = μ₆ − 15 μ₄ μ₂ − 10 μ₃² + 30 μ₂³).
- Prediction model for magnitude uses **even-cumulant truncation excluding κ₆**:
  log c_pred(t) = −κ₂(t)/2 + κ₄(t)/24  (no fitting, no scanning).
- Audit uses a **fixed declared window**; no adaptive windowing, tuning, or rescues.
- A coherence floor guard prevents unstable log comparisons.

**Repro:**
- Command(s): `python toys/s0013_k6_magnitude_sufficiency_boundary_demo.py`
- Optional: `--k6_heavy_b <b>` (must remain defaulted to the certified value).
- Inputs: none (deterministic seeds; constructions declared in-file).
- Outputs (written only with `--write_outputs`):
  - `toys/outputs/s0013_envelope_compare.csv`
  - `toys/outputs/s0013_error_diagnostic.png`
  - `toys/outputs/s0013_audit.csv`

**Expected output (fixed subcases):**
- **GAUSS:** κ₂+κ₄ magnitude closure passes in the declared window (OK).
- **K6_MILD:** κ₂+κ₄ magnitude closure passes (OK).
- **K6_HEAVY:** κ₂+κ₄ magnitude closure fails strongly (FAIL), with κ₆(t*) large in magnitude.

Certification logic:
- GAUSS must be OK.
- K6_HEAVY must be FAIL with median|dlogc| ≥ `min_median_abs_dlogc_fail`.

**Failure interpretation:**
- If GAUSS fails → predictor / cumulant pipeline is invalid in this regime.
- If K6_HEAVY does not fail → κ₆ is not sufficiently large at the Δϕ level in the declared window
  (construction too weak / tails too rare) and the boundary is not certified.
- If K6_HEAVY fails but GAUSS also fails → regime is numerically unstable or coherence floor is violated
  (discard / treat as INCONCLUSIVE).

**Interpretation boundary:**  
This artifact certifies only that **κ₆ can drive magnitude-envelope failure** of a κ₂+κ₄ truncation
under the declared construction and window. It does not claim κ₆ is fundamental, only that even-cumulant
closure beyond κ₄ is not guaranteed and is auditable.

---

## S-0014: κ₂+κ₄ magnitude closure under C5 measurement kernels (false-pass prevention)

**Claim:**  
C5 measurement kernels (finite detector bandwidth, windowing, additive noise) can **mask**
higher-cumulant (κ₆ and/or κ₅) departures such that κ₂+κ₄ magnitude closure appears to pass
in a naïve audit window.  
This toy certifies a **no-wrong-claims boundary**: under declared kernels and fixed windows,
the audit logic must either (a) detect the contamination (FAIL) or (b) refuse to certify (BOUNDARY/INCONCLUSIVE),
but it must **not** return OK for a case whose underlying Δϕ(t) ensemble violates κ₂+κ₄ magnitude sufficiency.

**Assumptions:**
- Phase accumulation: Δϕ(t)=∫₀ᵗ ω(s)ds (discrete cumulative sum, fixed Δt).
- Underlying ω innovations are constructed deterministically (seeded) to realize:
  - **GAUSS baseline** (κₙ>2 negligible),
  - **EVEN-contaminated** case with κ₆-driven magnitude failure (per S-0013-style construction),
  - optional **ODD-contaminated** case with κ₅ phase bias but magnitude stable (per S-0009/S-0012 logic).
- C5 kernels are applied as **measurement operators** (not dynamics):
  windowing and detector low-pass filtering on ω (or equivalently on the inferred ω),
  plus additive measurement noise at declared SNR.
- κ₂ and κ₄ are computed from the **post-kernel inferred Δϕ**, and κ₂+κ₄ magnitude prediction is evaluated
  in a declared fixed window (no adaptive tuning).
- The audit includes explicit **identifiability gates** (coherence-floor, window integrity, and kernel-strength bounds)
  that prevent certification when the measurement operator can erase higher-cumulant signatures.

**Repro:**
- Command(s): `python toys/s0014_c5_kernels_hide_higher_cumulants_demo.py`
- Inputs: none (deterministic seed; declared kernels and constructions in-file).
- Outputs (only with `--write_outputs`):
  - `toys/outputs/s0014_cases.csv`
  - `toys/outputs/s0014_audit.csv`
  - `toys/outputs/s0014_error_diagnostic.png` (optional)

**Expected behavior (fixed subcases):**
- **GAUSS + mild C5:** κ₂+κ₄ magnitude closure passes (OK).
- **K6_CONTAM + mild C5:** κ₂+κ₄ magnitude closure fails (FAIL).
- **K6_CONTAM + strong C5:** naïve closure may “look OK” in magnitude, but **certification must be refused**
  via gates (BOUNDARY/INCONCLUSIVE) rather than returning OK.
- (Optional) **K5_CONTAM:** magnitude closure remains OK; phase-structure flag remains separable (links to S-0009).

**Failure interpretation:**
- If **K6_CONTAM + strong C5** returns OK → the framework permits a false sufficiency claim (hard failure).
- If GAUSS fails under mild C5 → pipeline invalid or kernel bounds too aggressive.
- If K6_CONTAM fails to fail under mild C5 → κ₆ contamination is not surviving to Δϕ statistics (construction invalid).
- If many cases become INCONCLUSIVE → gates are too strict or windows/kernels are misdeclared.

**Interpretation boundary:**  
This toy certifies only the **C5 identifiability boundary** for κ₂+κ₄ magnitude closure: when measurement kernels
are strong enough to erase higher-cumulant signatures, the correct response is **refusal (BOUNDARY/INCONCLUSIVE)**,
not a false OK. No claim is made about physical ontology of ω or about real detector models beyond the declared kernels.

---

## S-0015: Cross-trajectory coupling can mimic long-memory κ₂ scaling (C2 ⟷ C4 confusion), with refusal via variance-ratio coupling gate

**Claim:**  
A cross-trajectory coupling construction (C4-style shared-mode injection) can produce a κ₂-slope α in the long-memory band (C2-like) within a fixed late-time window even when no true C2 long-memory adjacency is present; therefore κ₂-slope alone is insufficient for C2 attribution.  
S-0015 certifies that an explicit, window-restricted **variance-ratio coupling detector** on ω prevents false C2 certification: any case that appears long-memory by κ₂-slope but exhibits detected cross-trajectory coupling must be labeled **BOUNDARY**, not **OK_LM**.

**Assumptions:**
- Phase accumulation is defined as Δϕ(t)=∫₀ᵗ ω(s)ds (discrete cumulative sum with fixed Δt).
- Centering is **global DC removal only** (one constant mean over ensemble×time), identical across cases.
- The κ₂-slope classifier is computed from κ₂(t)=Var[Δϕ(t)] by a linear fit of log κ₂ vs log t over a **declared fixed late-time window**.
- A **true long-memory baseline** is provided by fractional Gaussian noise with fixed H (no scanning).
- Cross-trajectory coupling (C4-style) is instantiated as a **shared latent ω_shared(t)** component added to every trajectory (plus independent short-memory noise).
- Coupling is detected by the **variance ratio** computed over ω within the audit window:
  - `var_mean := Var_t( mean_i ωᵢ(t) )`
  - `mean_var := mean_i Var_t( ωᵢ(t) )`
  - `r_mean := var_mean / mean_var`
  - For independent trajectories, r_mean ≈ 1/nT (up to finite-window noise); for shared-mode coupling, r_mean increases.
- Coupling decision rule is **declared and fixed**:
  - `coupling_bound = coupling_factor / n_trajectories`
  - `coupling_detected := (r_mean > coupling_bound)`
  - No tuning, scanning, or post-hoc adjustment is permitted.

**Repro:**
- Command(s): `python toys/s0015_c2_vs_c4_cross_trajectory_confusion_demo.py`
- Inputs: none (deterministic seed; parameters declared in-file via `S0015Config`).
- Outputs (only with `--write_outputs`):
  - `toys/outputs/s0015_cases.csv`
  - `toys/outputs/s0015_audit.csv`

**Expected output (fixed subcases, same fixed window):**
- **C1_OU_BASE:** admissible; α in OU band; coupling_detected = 0 → `OK_OU`.
- **C2_LM_TRUE:** admissible; α in LM band; coupling_detected = 0 → `OK_LM`.
- **C4_COUPLED:** admissible; α may fall in LM band, but coupling_detected = 1 → `BOUNDARY`.
  If the coupled case is inadmissible, result is `INCONCLUSIVE` (no claim).

Exit codes:
- `0` → PASS iff OU_BASE is `OK_OU`, LM_TRUE is `OK_LM`, and no forbidden “false OK_LM under coupling” occurs.
- `2` → FAIL if either baseline fails, or if C4_COUPLED is tagged `OK_LM` while coupling_detected = 1.
- `3` → INCONCLUSIVE if required baselines are inadmissible (no claim).

**Failure interpretation:**
- If LM_TRUE is flagged as coupled (coupling_detected=1), the coupling gate is incorrect under the declared window.
- If C4_COUPLED yields OK_LM with coupling_detected=1, classifier integrity fails (hard FAIL).
- If the coupled case is consistently inadmissible, the regime is window/horizon dominated → INCONCLUSIVE (do not tune).

**Interpretation boundary:**  
This item certifies refusal logic: κ₂-slope can be C2-like under C4 coupling, so C2 attribution requires an explicit cross-trajectory coupling guard. No ontology is claimed beyond the declared metric and window.

---

# S-0016: Non-stationary temporal curvature boundary for κ₂-slope admissibility (C1 guard)

**Claim:**  
A deterministic or slowly varying temporal curvature in ω(t) can mimic long-memory κ₂ scaling within a finite audit window; therefore κ₂-slope classification is admissible **only when temporal curvature remains below a declared bound.**

S-0016 certifies that introducing a per-trajectory quadratic temporal curvature term  
ωᵢ,curv(t) = ωᵢ(t) + γᵢ·t² produces a κ₂-slope α that appears long-memory-consistent,  
but must be refused by a **curvature admissibility gate**, yielding **BOUNDARY**, not OK_LM.

**Assumptions:**

- Phase accumulation: Δϕ(t) = ∫ ω(s) ds (discrete cumulative sum with fixed Δt).
- Centering: global DC removal only.
- Scaling classifier: α = slope of log κ₂(t) vs log t in a fixed late-time window.
- Baselines:
  - OU (short-memory; expected α≈1)
  - Long-memory Gaussian (fGn with fixed H; expected α≈2H)
- Curvature confound:
  - Each trajectory includes a deterministic curvature γᵢ·t²
  - γᵢ is drawn from a fixed symmetric distribution (no tuning).
- Curvature detector (declared, fixed; window-restricted):
  - Fit ω̄(t) (ensemble mean ω) in the audit window to a quadratic:
        ω̄(t) ≈ a + b·t + c·t²
  - Define curvature metric (quadratic-coefficient significance):
        curvature_z := |c| / SE(c)
    where SE(c) is the OLS standard error of the quadratic coefficient in the same window.
  - Decision rule:
        curvature_detected := (curvature_z > curvature_z_bound)
  - curvature_z_bound is fixed in `S0016Config` (no tuning/scanning).

**Repro:**

- Command(s): `python toys/s0016_temporal_curvature_boundary_demo.py`
- Inputs: none (deterministic seed; declared parameters).
- Outputs (only with `--write_outputs`):
  - `toys/outputs/s0016_cases.csv`
  - `toys/outputs/s0016_audit.csv`

**Expected output (fixed subcases):**

- **OU_BASE:** admissible; α≈1; curvature_detected = 0 → `OK_OU`
- **LM_TRUE:** admissible; α≈2H; curvature_detected = 0 → `OK_LM`
- **C1_CURVED:** admissible by slope alone (α may fall in LM band),
  but curvature_detected = 1 → `BOUNDARY`

Exit codes:

- `0` → PASS iff OU_BASE = OK_OU, LM_TRUE = OK_LM, and CURVED case is refused (BOUNDARY)
- `2` → FAIL if either baseline fails, or if CURVED case returns OK_LM
- `3` → INCONCLUSIVE if baselines are inadmissible

**Failure interpretation:**

- If OU_BASE or LM_TRUE are flagged as curved → curvature gate is invalid under the declared window.
- If C1_CURVED yields OK_LM → classifier integrity fails (hard FAIL).
- If CURVED is always inadmissible → the declared audit window is insufficient.

**Interpretation boundary:**  
This tool certifies a C1 admissibility constraint: κ₂-slope classification is valid only in locally stationary regimes. Temporal curvature is diagnostic, not ontological.

---

## S-0017: Cross-window κ₂-slope regime consistency (temporal stationarity guard)

**Claim:**  
κ₂-slope classification is admissible only if the inferred scaling exponent α remains
**consistent across adjacent continuation windows**.  
A process that appears long-memory–consistent in one late-time window but diverges in the next
must be labeled **BOUNDARY**, not **OK_LM**.  
S-0017 certifies that this cross-window consistency constraint prevents false regime
certification arising from temporal non-stationarity (C1 effects).

**Assumptions:**

- Phase accumulation: Δϕ(t)=∫ ω(s)ds (discrete cumulative sum, fixed Δt).
- Centering: global DC removal only (one constant mean per case).
- Scaling classifier: α = slope of log κ₂(t) vs log t in a declared audit window.
- Two **fixed continuation windows** are declared:
  - Window A: `[winA_t_min, winA_t_max]`
  - Window B: `[winB_t_min, winB_t_max]`
- A regime is admissible only if both windows yield valid slope fits and:

      |α_A − α_B| < consistency_tol

- No window scanning, adaptive selection, or post-hoc tuning is permitted.

**Repro:**

- Command(s):  
  `python toys/s0017_cross_window_regime_consistency_demo.py`
- Inputs: none (deterministic seed; parameters declared in-file via `S0017Config`).
- Outputs (only with `--write_outputs`):
  - `toys/outputs/s0017_cases.csv`
  - `toys/outputs/s0017_audit.csv`

**Expected output (fixed subcases):**

- **C1_OU_BASE:**  
  α_A and α_B fall in OU band → `OK_OU`
- **C2_LM_TRUE:**  
  α_A and α_B fall in LM band and satisfy tolerance → `OK_LM`
- **C1_CURVED:**  
  α_A and α_B diverge beyond tolerance → `BOUNDARY`

If either window is inadmissible, verdict is `INCONCLUSIVE`.

Exit codes:

- `0` → PASS iff OU_BASE = OK_OU, LM_TRUE = OK_LM, and CURVED case = BOUNDARY.
- `2` → FAIL if OU_BASE or LM_TRUE fail, or if CURVED yields OK_LM.
- `3` → INCONCLUSIVE if required baselines are inadmissible.

**Failure interpretation:**

- If OU_BASE or LM_TRUE violate cross-window consistency, the classifier is unstable.
- If CURVED yields OK_LM, the continuation admissibility gate has failed.
- If CURVED is always inadmissible, the declared continuation window is insufficient.

**Interpretation boundary:**  
This tool certifies a Σ₂ admissibility rule:  
κ₂-slope regime claims are valid only when scaling is **temporally persistent across
adjacent continuation domains**.  
This is a diagnostic refusal constraint, not an ontological claim about the underlying dynamics.

---

## S-0018: Non-Gaussian innovation masquerade boundary for κ₂-slope classification integrity (C2 ⟷ distribution guard)

**Claim:**  
A phase process can match κ₂(t) scaling (OU-like α≈1 or long-memory-like α≈2H) while being strongly
non-Gaussian at the innovation level; therefore κ₂-slope alone is insufficient for admissible
regime certification. S-0018 certifies that a window-restricted **excess-kurtosis guard** on ω
prevents false OK_* outcomes: any case matching κ₂-slope bands but exhibiting detected non-Gaussianity
must be labeled **BOUNDARY**, not OK_OU/OK_LM.

**Assumptions:**
- Phase accumulation: Δϕ(t)=∫₀ᵗ ω(s)ds (discrete cumulative sum with fixed Δt).
- Centering is **global DC removal only** (one constant mean over ensemble×time), identical across cases.
- κ₂-slope classifier is computed from κ₂(t)=Var[Δϕ(t)] by a linear fit of log κ₂ vs log t over a
  **declared fixed late-time window**.
- A **true long-memory baseline** is provided by fractional Gaussian noise with fixed H (no scanning).
- A non-Gaussian masquerade case is instantiated as OU recursion driven by a symmetric heavy-tail
  innovation mixture normalized to Var≈1 (no tuning).
- Non-Gaussianity is detected by a **window-restricted excess kurtosis** of ω:
  - Compute μ₂ and μ₄ over ω samples restricted to the audit window (flattened over i,t).
  - `g2 := μ4 / μ2^2 − 3`
  - `kurtosis_detected := (|g2| > kurtosis_bound)` where kurtosis_bound is fixed in `S0018Config`.
- Refusal rule is mandatory:
  - If `kurtosis_detected = 1`, tag is forced to **BOUNDARY** (never OK_*), regardless of κ₂-slope.

**Repro:**
- Command(s): `python toys/s0018_non_gaussian_masquerade_boundary_demo.py`
- Inputs: none (deterministic seed; parameters declared in-file via `S0018Config`).
- Outputs (only with `--write_outputs`):
  - `toys/outputs/s0018_cases.csv`
  - `toys/outputs/s0018_audit.csv`

**Expected output (fixed subcases, same fixed window):**
- **C1_OU_BASE:** admissible; α in OU band; kurtosis_detected = 0 → `OK_OU`.
- **C2_LM_TRUE:** admissible; α in LM band; kurtosis_detected = 0 → `OK_LM`.
- **NON_GAUSS:** admissible by κ₂-slope (α in OU band), but kurtosis_detected = 1 → `BOUNDARY`.
  If any case is inadmissible, result is `INCONCLUSIVE` (no claim).

Exit codes:
- `0` → PASS iff OU_BASE is `OK_OU`, LM_TRUE is `OK_LM`, and NON_GAUSS is `BOUNDARY`.
- `2` → FAIL if either baseline fails, or if NON_GAUSS is tagged `OK_OU`/`OK_LM` (forbidden false certification).
- `3` → INCONCLUSIVE if required baselines are inadmissible (no claim).

**Failure interpretation:**
- If LM_TRUE or OU_BASE yields kurtosis_detected=1, the kurtosis gate is invalid under the declared window.
- If NON_GAUSS yields OK_* while kurtosis_detected=1, classifier integrity fails (hard FAIL).
- If NON_GAUSS is consistently inadmissible, the regime is window/horizon dominated → INCONCLUSIVE (do not tune).

**Interpretation boundary:**  
This item certifies refusal logic: κ₂-slope scaling does not guarantee Gaussian admissibility.
The kurtosis guard is diagnostic (C2 distributional integrity), not ontological.

---

## S-0019: Variance-drift masquerade boundary for κ₂-slope admissibility (Σ₂ guard)

**Claim:**  
A slow, deterministic drift in the variance budget of ω(t) can preserve an apparently clean κ₂(t) scaling exponent α within a fixed audit window while invalidating regime attribution across continuation windows. Therefore κ₂-slope classification is admissible **only when the ω-variance budget is locally stationary** under a declared variance-drift gate.

S-0019 certifies that a controlled **variance-drift construction** (OU-like increments with a time-dependent amplitude envelope) can appear compatible with OU or LM κ₂-slope bands in an individual window, but must be refused (BOUNDARY) when a window-restricted variance-drift detector exceeds a declared bound.

**Assumptions:**
- Phase accumulation: Δϕ(t)=∫₀ᵗ ω(s)ds (discrete cumulative sum with fixed Δt).
- Centering: global DC removal only (one constant mean over ensemble×time), identical across cases.
- Scaling classifier: α from a linear fit of log κ₂(t) vs log t over a declared fixed window.
- Baselines:
  - OU short-memory baseline (expected α≈1).
  - Long-memory Gaussian baseline via fGn with fixed H (expected α≈2H); no scanning.
- Variance-drift confound (Σ₂-style admissibility violation):
  - ωᵢ(t) := μ + s(t)·ξᵢ(t) where ξᵢ(t) is stationary OU-like noise and s(t) is a deterministic,
    slowly varying positive envelope (no tuning), chosen so that κ₂-slope in the audit window can
    remain apparently admissible while cross-window continuation changes.
- Variance-drift detector (declared, fixed; window-restricted):
  - Let ω̄(t) := meanᵢ ωᵢ(t) and vᵢ := Varₜ(ωᵢ(t)) over the audit window.
  - Define mean_var := meanᵢ vᵢ (typical per-trajectory fluctuation scale).
  - Define v_hat(t) := Varᵢ(ωᵢ(t)) (instantaneous cross-sectional variance).
  - Fit log v_hat(t) vs t over the window to obtain a drift slope d.
  - Define drift_z := |d| / max(1e-12, 1/sqrt(n_w))  (normalized by finite-window scale).
  - Decision rule: variance_drift_detected := (drift_z > drift_z_bound), with fixed bound in config.
  - No tuning, scanning, or post-hoc adjustment is permitted.

**Repro:**
- Command(s): `python toys/s0019_variance_drift_masquerade_boundary_demo.py`
- Inputs: none (deterministic seed; parameters declared in-file via `S0019Config`).
- Outputs (only with `--write_outputs`):
  - `toys/outputs/s0019_cases.csv`
  - `toys/outputs/s0019_audit.csv`

**Expected output (fixed subcases, same fixed window):**
- **C1_OU_BASE:** admissible; α in OU band; variance_drift_detected = 0 → `OK_OU`.
- **C2_LM_TRUE:** admissible; α in LM band; variance_drift_detected = 0 → `OK_LM`.
- **C5_VAR_DRIFT:** admissible by κ₂-slope alone (α may fall in OU or LM band),
  but variance_drift_detected = 1 → `BOUNDARY` (never OK_*).
  If the drifted case is inadmissible by fit quality, result is `INCONCLUSIVE` (no claim).

Exit codes:
- `0` → PASS iff OU_BASE is `OK_OU`, LM_TRUE is `OK_LM`, and drifted case is refused (BOUNDARY).
- `2` → FAIL if either baseline fails, or if drifted case is tagged `OK_OU`/`OK_LM` while variance_drift_detected = 1.
- `3` → INCONCLUSIVE if required baselines are inadmissible (no claim).

**Failure interpretation:**
- If OU_BASE or LM_TRUE are flagged as drifted, the variance-drift gate is invalid under the declared window.
- If VAR_DRIFT yields OK_* with variance_drift_detected=1, classifier integrity fails (hard FAIL).
- If VAR_DRIFT is consistently inadmissible, the regime is window/horizon dominated → INCONCLUSIVE (do not tune).

**Interpretation boundary:**  
This item certifies a Σ₂ admissibility rule: κ₂-slope regime claims require a locally stationary variance budget for ω. The detector is diagnostic and window-restricted; no ontology is claimed beyond the declared construction.
