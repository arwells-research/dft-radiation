# Phase Evolution — ω → cumulants → envelopes

**A minimal theorem explaining why phase decoherence envelopes exist at all —
and why their form is constrained.**

This tool constructs a bridge from DFT’s **motion-first axioms** to **microscopic
phase-evolution rules** and **observable coherence envelopes**, without introducing
independent stochastic dynamics, phenomenological noise sources, or tunable
fit parameters.

The goal is not to assume envelopes — but to **derive when and why envelope
descriptions are valid, and where they provably fail**.

---

## What this tool is for

- Define a **primitive micro-variable** (ω) as a **projection** of scalar progression
  (not an independently postulated noise source).
- Treat phase as an **accumulated constraint**, expressed as an integral functional
  of ω along a trajectory.
- Derive coherence-envelope behavior from **cumulant structure** under explicit,
  auditable assumptions (stationarity, mixing, correlation decay, adjacency).
- Classify envelope families by **correlator structure**, not by ad-hoc model choice.
- Establish **certified falsification boundaries** (where κ₂ closure fails
  categorically, not just approximately).
- Provide a validation harness for future **microscopic generator derivations**
  (micro-motion → phase increments → cumulants → envelopes).

---

## What this tool is not

- Not a replacement for the canonical DFT chapter spine (01–05).
- Not a proposal of a fundamental stochastic ontology.
- Not a parameter-fitting or phenomenological envelope framework.
- Not a “works-because-it-fits” noise model library.

All stochastic structure here is **diagnostic**, not ontological.

---

## Status and visibility

This directory is visible under `tools/` **only insofar as it contains
registered, checkable successes**.

Exploratory constructions, scaffolding, or incomplete derivations do **not**
belong here.

See `success_register.md` for the authoritative success ledger,
acceptance criteria, and failure interpretations.

---

## Registered successes (current)

The following items are **the only canonical results** in this directory.

They form a logically ordered **boundary-certification program** for phase
evolution models.

### **S-0001 — κ₂ scaling classification**
Distinguishes late-time diffusive scaling (integrated OU, κ₂(t)∝t) from
long-memory scaling (fractional Gaussian noise, κ₂(t)∝t^{2H}) via audited
log-slope diagnostics.
Establishes that κ₂ scaling families are **observable and classifiable**, not assumed.

### **S-0002 — κ₂ predictive closure & adjacency sensitivity**
Demonstrates that the coherence envelope
c(t)=|⟨exp(iΔϕ(t))⟩| is reconstructible from the **empirical autocovariance**
via the discrete κ₂ functional **if and only if** long-range adjacency is retained
(fGn_RAW PASS).
Short-memory OU fails closure; estimator modifications (e.g. Bartlett taper)
introduce **auditable drift**, not genuine closure.

### **S-0003 — Boundary of Gaussian sufficiency (κ₃ diagnostic)**
With κ₂ held fixed by construction, κ₃≠0 produces a **sign-reversible,
auditable phase bias** in the complex coherence ⟨exp(iΔϕ(t))⟩, while κ₂
continues to predict the **magnitude envelope**
c(t)=|⟨exp(iΔϕ(t))⟩| within declared tolerances.
This certifies a sharp boundary: **κ₂ is sufficient for magnitude but not for
phase-neutral Gaussian sufficiency**.

### **S-0004 — Even-cumulant boundary (κ₄ magnitude collapse)**
With κ₂ held fixed and a GAUSS baseline verifying pipeline identity,
introducing κ₄≠0 produces a **systematic magnitude-envelope collapse**
that κ₂ closure cannot explain.
This establishes a **categorical failure mode**: even cumulants falsify
κ₂-only envelope models.

### **S-0004-SENS — Boundary metric & ρ₍crit₎ bracketing**
Turns S-0004 into a **dimensionless diagnostic standard** using
ρ(t*) = |κ₄(t*)| / κ₂(t*)².
A validity gate (`gauss_ok`) excludes noise-floor artifacts, and a
2-consecutive-failure rule produces a reproducible bracket
[ρ_last_nonfail, ρ_first_fail], defining a **certified Gaussian-sufficiency
boundary** rather than a heuristic threshold.

### **S-0005 — Cross-constraint closure under measurement kernels (C1 ⟷ C2 ⟷ C5)**
Demonstrates that regime classification based on **κ₂(t)=Var(Δϕ(t)) scaling**
(short-memory vs long-memory) is **stable under admissible measurement effects**
(windowing, finite detector response, additive noise).
Measurement kernels may degrade estimator quality or render cases inadmissible,
but **cannot invert regime assignment** when classification is applied strictly
within its certified scaling window.
This establishes a **C5 boundary**: failure indicates measurement-dominated
inference, not a new dynamical regime.

### **S-0006 — Radiation ω generator families under κ₂ scaling (C1 ⟷ C2 ⟷ C3 ⟷ C4)**
Demonstrates that late-time **κ₂(t)=Var(Δϕ(t)) scaling** is **stable across a finite,
declared menu of ω(t) generator families** representing phenomenological bindings
of temporal dynamics (C1), statistical memory (C2), linear transport (C3), and
interface injection (C4).
Linear transport filtering and event-driven (shot-like) interface effects may
alter bandwidth or increment statistics, but **do not change the κ₂ scaling class**
or violate declared α expectations on any admissible case.
This establishes a **C3/C4 binding boundary**: failure indicates loss of a
certified scaling window or boundary-dominated construction, not a new radiation
regime.

### **S-0007 — Mixture crossover restraint (“no-wrong-claims” certification) under κ₂ scaling**
Demonstrates that κ₂(t)=Var(Δϕ(t))–based regime classification is **restrained under
declared mixture and crossover constructions**, formed by finite linear mixtures
ω = ω_OU + ε·ω_fGn (with optional C3/C4 bindings).
Using **fixed early and late audit windows**, the classifier:
- permits **BOUNDARY (no-claim) outcomes** in crossover or mixed-scaling regions,
- **forbids incorrect regime claims** on any admissible window,
- and **correctly classifies the pure OU baseline** while avoiding over-interpretation
  when no clean late-time scaling regime is realized within the declared horizon.
This establishes a **mixture/crossover boundary**: failure indicates a false
regime claim, not the emergence of a new dynamical regime.

### **S-0008 — Drift-vs-diffusion confound (κ₂-slope non-identifiability)**
Demonstrates that κ₂(t)=Var(Δϕ(t)) slope α≈1 in a declared window is **not sufficient**
to distinguish true diffusive behavior from a confound in which each trajectory carries
a small **linear drift in ω(t)** (βᵢ·t) with random βᵢ across the ensemble.
Both constructions can yield α≈1 and pass slope admissibility, so identifying drift
requires additional diagnostics beyond κ₂ scaling (C1/C2 boundary).

### **S-0009 — Higher-order odd-cumulant boundary (κ₅ phase-structure diagnostic)**
Demonstrates that controlled **κ₅ ≠ 0** at the innovation level produces a **sign-reversible
phase bias** in the complex coherence ⟨exp(iΔϕ(t))⟩ while the **magnitude envelope**
remains consistent with an even-cumulant truncation (κ₂, κ₄) over the declared window,
establishing a categorical phase-structure boundary beyond κ₄.

Using a fixed late-time audit window, this toy compares:

- a **diffusion-limited baseline** (OU-like κ₂ scaling, α≈1), and  
- the same baseline with an added **per-trajectory deterministic drift term**
  ωᵢ,drift(t)=βᵢ·t, representing a **C5 observational/measurement-induced bias** rather
  than a true dynamical regime.

The classifier:

- correctly identifies the diffusion baseline as **OK**,  
- refuses the drift-contaminated case by returning **BOUNDARY**,  
- and makes **no forbidden claims on any admissible window.**

This establishes a clean **C2 vs C5 separation boundary**: failure would indicate that
κ₂-based coherence scaling cannot distinguish genuine phase diffusion from slow
instrument-induced drift.

### **S-0010 — κ₂ transport vs interface discrimination (C3 ⟷ C4 identifiability)**

Demonstrates that late-time κ₂(t)=Var(Δϕ(t)) scaling is **invariant under geometric transport (C3)**  
but departs detectably under **interface-driven phase injection or exchange (C4)**.

Within a fixed admissible scaling window:

- OU baseline and transport-filtered cases (smoothing, dispersion) remain κ₂-consistent.
- Interface-driven cases (shot-like injection, gain/loss envelopes) alter κ₂ scaling while remaining
  admissible, producing a certified **C3 vs C4 separation boundary**.

Failure indicates loss of constraint-class separability, not the emergence of a new regime.

### **S-0011 — Kurtosis-based separation of transport vs interface effects (C3 ⟷ C4 higher-moment boundary)**

Demonstrates that while κ₂ scaling remains invariant under both C3 and C4 constructions,
**higher-order moment structure (excess kurtosis)** cleanly separates them.

Within the same declared scaling window:

- OU baseline and transport-only (C3) cases exhibit **near-Gaussian increment statistics** (low kurtosis).
- Interface-injection (C4) cases exhibit **strong heavy-tail statistics** (high kurtosis), while preserving
  κ₂ scaling class and admissibility.

This establishes a **second-order C3/C4 discrimination boundary**:  
κ₂ classifies the regime; kurtosis certifies whether deviation arises from transport or interface coupling.

### **S-0012 — Sixth cumulant (κ₆) even-moment structural boundary**

Extends the Gaussian sufficiency program beyond κ₄ by demonstrating that a
controlled non-zero **sixth cumulant (κ₆ ≠ 0)** at the innovation level produces
a measurable deformation in the coherence envelope that is **not captured by
κ₂ + κ₄ closure alone**, while preserving admissibility and deterministic scaling.

Within the declared window:

- The GAUSS baseline confirms κ₂ + κ₄ magnitude closure remains valid.
- Both κ₆+ and κ₆− cases remain admissible and preserve κ₂ scaling class.
- The resulting envelope deformation is **sign-invariant** (even-order) and
  therefore diagnostic of higher-order even-moment structure rather than
  transport or interface effects.

This establishes the next categorical **even-cumulant structural boundary**
in the S-series progression.

### **S-0013 — Magnitude sufficiency breakdown under strong κ₆ forcing**

Builds directly on S-0012 to certify that when κ₆ is forced beyond a deterministic
strength threshold (`k6_heavy_b = 25.0`), the even-cumulant magnitude model
(κ₂ + κ₄) **fails in a controlled, auditable manner**, while the GAUSS and mild κ₆
cases continue to pass.

Within the same declared window:

- **GAUSS** and **κ₆_MILD** cases remain fully consistent with κ₂ + κ₄ magnitude closure (PASS).
- The **κ₆_HEAVY** construction produces a clear, reproducible **magnitude-closure failure**
  exceeding the declared tolerances (FAIL), demonstrating that κ₆ can couple into
  the envelope magnitude even when κ₄ is held fixed.

This establishes a certified **κ₆ magnitude-sufficiency boundary** and marks the
first point at which higher-order even cumulants falsify the truncated envelope
model itself rather than only deforming the phase structure.

### **S-0014 — κ₂+κ₄ magnitude closure under C5 measurement kernels with higher-cumulant contamination (“no false OK”)**

Certifies that **C5 measurement kernels can partially mask higher-cumulant (κ₆) structure**, but the framework must **forbid false PASS outcomes** when identifiability is lost.

This toy compares **truth envelopes** (computed from the unobserved ω) against **κ₂+κ₄ magnitude predictions** computed from *kernel-corrupted* observations ω̂ produced by a declared C5 measurement model (moving-average blur + additive noise).

Within a fixed audit window:

- **GAUSS_MILD** (ma=1, SNR=80 dB): κ₂+κ₄ magnitude closure remains valid (PASS).
- **K6_MILD** (κ₆-heavy innovations, ma=1, SNR=80 dB): κ₂+κ₄ closure **must fail** strongly (FAIL), proving higher even cumulants can break magnitude closure even under good measurement.
- **K6_STRONG** (same κ₆-heavy truth, but ma=25, SNR=10 dB): kernel exceeds identifiability gates, so the toy **refuses certification** and returns **BOUNDARY**—explicitly **forbidding “false OK”** under strong C5 masking.

This establishes a **C5 masking boundary**: when measurement kernels violate declared identifiability limits, the system must return **BOUNDARY** rather than certify κ₂+κ₄ magnitude closure, even if the corrupted inference appears superficially consistent.

### **S-0015 — κ₂ scaling misclassification under cross-trajectory coupling (“C2 vs C4 integrity gate”)**

Certifies that **cross-trajectory coupling (C4-style shared/interface structure) can
produce misleading κ₂(t) scaling that superficially resembles long-memory (C2-style)
behavior**, and therefore must be **detected and refused**, not misclassified.

This toy introduces a deterministic **variance-of-mean coupling detector**:

r_mean = Varₜ( meanᵢ ωᵢ(t) ) / meanᵢ Varₜ( ωᵢ(t) )

Within a fixed admissible audit window:

- **C1_OU_BASE**: independent OU trajectories.  
  κ₂ scaling classifies correctly as **OK_OU**.
- **C2_LM_TRUE**: uncoupled long-memory Gaussian (fractional innovations).  
  κ₂ scaling classifies correctly as **OK_LM**.
- **C4_COUPLED**: trajectories share a low-amplitude common component.  
  The detector flags coupling, and the classifier is forced to **BOUNDARY**  
  — explicitly **forbidding false long-memory certification.**

This establishes the first **C2 ⟷ C4 admissibility boundary** for κ₂-based
classification: long-memory and interface-coupled structure are distinguishable
at the cumulant level, and any detected coupling invalidates κ₂-only claims.

A PASS certifies that:
- **C2-style memory and C4-style coupling remain separable**, and
- the framework enforces the required **“no wrong claims” refusal logic.**

### **S-0016 — κ₂ scaling admissibility under non-stationary temporal curvature (“C1 curvature guard”)**

Certifies that **deterministic or slowly varying temporal curvature in ω(t) can
mimic long-memory-like κ₂(t) scaling within a finite audit window**, and therefore
κ₂-slope classification is admissible **only when the observed phase rate remains
locally stationary.**

This toy adds a per-trajectory quadratic temporal term and enforces a
window-restricted **curvature admissibility detector** by fitting the ensemble-mean
phase rate ω̄(t) to:

  ω̄(t) ≈ a + b·t + c·t²

and defining a dimensionless curvature significance metric:

  curv_z = |c| / SE(c)

If `curv_z` exceeds the declared `curvature_bound`, the case is labeled
**BOUNDARY**, never **OK_LM**, regardless of κ₂-slope agreement.

Within the fixed late-time audit window:

- **C1_OU_BASE**: independent OU trajectories with no curvature.  
  κ₂ scaling classifies correctly as **OK_OU**.
- **C2_LM_TRUE**: uncoupled long-memory Gaussian baseline.  
  κ₂ scaling classifies correctly as **OK_LM**.
- **C1_CURVED**: OU noise plus deterministic quadratic curvature.  
  The curvature detector triggers, and the classifier is forced to **BOUNDARY**  
  — explicitly **forbidding false long-memory certification under temporal drift.**

This establishes the first **C1 admissibility boundary for κ₂-based
classification**: apparent long-memory scaling is invalid unless temporal
curvature is ruled out.

A PASS certifies that:
- **C1-style curvature and C2-style memory remain separable**, and  
- the framework enforces the required **“no wrong claims” refusal logic** for
  κ₂-slope attribution.

### **S-0017 — Cross-window κ₂-slope regime consistency (“temporal continuation admissibility”)**

Certifies that **κ₂-slope classification is admissible only when the inferred scaling
exponent α remains stable across adjacent late-time continuation windows.**

While S-0015 guards against cross-trajectory coupling and S-0016 guards against
temporal curvature, both are *window-local.*  
S-0017 introduces the first **cross-window persistence test**, ensuring that a process
identified as OU-like or long-memory-like in one admissible window must remain
consistent in the next.

This toy evaluates two fixed late-time windows:

- **Window A:** [20, 40]  
- **Window B:** [40, 80]

and enforces the continuation rule:

  |α₍A₎ − α₍B₎| < declared consistency tolerance

Within this framework:

- **C1_OU_BASE:** short-memory OU innovations classify consistently → **OK_OU**
- **C2_LM_TRUE:** uncoupled long-memory Gaussian innovations classify consistently → **OK_LM**
- **C1_CURVED:** deterministic temporal curvature causes cross-window drift → **BOUNDARY**

A PASS certifies that **temporal stationarity is a required Σ₂ admissibility condition
for κ₂-based regime classification.**  
If scaling does not persist under continuation, the system must refuse
certification rather than produce a false “OK_LM.”

This establishes the first **cross-window stationarity boundary** for phase-evolution
diagnostics in Dual-Frame Theory.

### **S-0018 — Non-Gaussian innovation masquerade under correct κ₂ scaling (“distributional refusal gate”)**

Certifies that **κ₂(t) scaling alone is not sufficient to certify regime admissibility**, because a process can
exhibit the correct κ₂-slope α (OU-like or long-memory-like) while remaining strongly **non-Gaussian in ω**.

This toy adds a fixed, window-restricted **excess-kurtosis guard** on ω:

g₂ = μ₄/μ₂² − 3

Within a declared admissible late-time window:

- **C1_OU_BASE**: Gaussian OU innovations.  
  κ₂ scaling certifies **OK_OU**, and kurtosis remains near zero.
- **C2_LM_TRUE**: uncoupled long-memory Gaussian innovations (fixed H).  
  κ₂ scaling certifies **OK_LM**, and kurtosis remains near zero.
- **NON_GAUSS**: OU-like κ₂ scaling driven by symmetric heavy-tail innovations.  
  κ₂ slope may still fall in the OU band, but the kurtosis guard detects non-Gaussianity and forces **BOUNDARY**  
  — explicitly **forbidding false OK_OU/OK_LM certification** under distributional masquerade.

This establishes a **C2 distributional integrity boundary** for κ₂-based classification:  
regime identification requires not only scaling consistency, but also declared admissibility of the innovation
distribution under observation.

### **S-0019 — Variance-drift masquerade under admissible κ₂ scaling (“Σ₂ variance guard”)**

Certifies that **κ₂(t) scaling alone is not sufficient to certify temporal admissibility**, because a process can
exhibit OU-like or long-memory-like κ₂-slope behavior while its **variance budget drifts within the audit window**.

This toy introduces a fixed, window-restricted **variance-drift refusal gate** on ω:

v̂(t) = Varᵢ(ωᵢ(t))  
drift_z = |log(v̂(t_max)/v̂(t_min))| · √n_w

Within a declared admissible late-time window:

- **C1_OU_BASE**: stationary OU innovations.  
  κ₂ scaling certifies **OK_OU**, and the variance guard remains quiet.
- **C2_LM_TRUE**: stationary long-memory Gaussian innovations (fixed H).  
  κ₂ scaling certifies **OK_LM**, and the variance guard remains quiet.
- **C5_VAR_DRIFT**: OU-like scaling under a deterministic variance envelope.  
  The κ₂-slope alone falls within the long-memory band, but the variance guard detects drift and forces **BOUNDARY**  
  — explicitly **forbidding false OK_OU/OK_LM certification** under non-stationary variance masquerade.

This establishes a **Σ₂ temporal-stationarity integrity boundary** for κ₂-based classification:  
regime identification requires both correct κ₂ scaling and **locally stationary variance of ω(t)** within the audit window.

### **S-0020 — κ₂ scaling admissibility as a direct projection of T-frame winding-sector structure (“derivational admissibility gate”)**

Certifies that the observed late-time **κ₂(t) power-law scaling exponent α is not merely statistical, but is a direct
projection of admissible winding-sector structure in the Temporal (T) frame.**

In Dual-Frame Theory, admissible phase evolution must arise from **scalar progression constrained by winding-sector topology**.
Therefore, any κ₂(t) scaling that falls outside the admissible winding-sector envelope must be treated as **BOUNDARY**, not OK.

This toy makes that connection explicit:

- It computes κ₂(t) = Var[Δϕ(t)] from simulated ω(t) trajectories.
- It maps the inferred α onto the corresponding **T-frame winding-sector signature**.
- It verifies that:
  - **Diffusive OU processes** project to the expected winding-sector α ≈ 1.
  - **True long-memory Gaussian processes** project to the expected α ≈ 2H.
  - Any process that reproduces κ₂ scaling while violating the implied **winding-sector curvature or continuity constraints**
    is refused and labeled **BOUNDARY**.

Within a declared admissible audit window:

- **C1_OU_BASE**: short-memory OU progression.  
  κ₂ scaling is admissible and corresponds to an allowed winding sector → **OK_OU**.
- **C2_LM_TRUE**: uncoupled long-memory Gaussian progression (fixed H).  
  κ₂ scaling is admissible and corresponds to a distinct allowed winding sector → **OK_LM**.
- **NON_ADMISSIBLE**: synthetic progression constructed to match κ₂ scaling while breaking winding-sector admissibility.  
  The derivational admissibility gate detects the inconsistency and forces **BOUNDARY**.

This establishes the first **derivational closure link** between measurable κ₂(t) scaling and the underlying
**topological admissibility of motion in the T-frame** — making κ₂ not just a classifier, but a **geometric diagnostic.**

### **S-0021 — κ₂-Slope vs Coherence-Aperture Consistency (“DFT admissibility gate”)**

Certifies that the observed late-time **κ₂(t) scaling exponent α is admissible only when the implied phase-diffusion
envelope remains within the DFT coherence-aperture limit L.**

In Dual-Frame Theory, κ₂(t) describes the spread of accumulated phase:

  κ₂(t) = Var[Δϕ(t)] with Δϕ(t)=∫₀ᵗ ω(s)ds

The square-root of κ₂(t) defines the physical phase-spread:

  σϕ(t) = √κ₂(t)

DFT’s **coherence aperture L** bounds how much phase spread remains physically admissible.
Therefore, κ₂-slope regime classification is valid only when:

  η(t) = σϕ(t) / L ≤ 1 for all t in the declared audit window.

Any case with η > 1 represents an **inadmissible phase-diffusion regime** and must be refused
and labeled **BOUNDARY**, regardless of κ₂-slope agreement.

This toy makes that admissibility condition explicit:

- It computes κ₂(t) from simulated ω(t) trajectories.
- It evaluates the normalized phase-spread η(t)=√κ₂(t)/L.
- It verifies that:
  - **Diffusive OU baselines** remain within aperture → **OK_OU**.
  - **True long-memory Gaussian baselines** remain within aperture → **OK_LM**.
  - A constructed case that preserves κ₂-slope but exceeds the aperture budget
    is detected and labeled **BOUNDARY**.

Within a declared admissible audit window:

- **C1_OU_BASE**: OU-like phase progression; η ≤ 1 → **OK_OU**.
- **C2_LM_TRUE**: long-memory Gaussian progression; η ≤ 1 → **OK_LM**.
- **C5_APERTURE_EXCESS**: identical κ₂-slope class to LM_TRUE, but η > 1 → **BOUNDARY**.

This establishes the first direct **link between κ₂ scaling and the physical coherence aperture L**, making κ₂ not only
a classifier, but a structural admissibility diagnostic within DFT.

### **S-0022 — Estimator-manufactured κ₂ scaling guard (“C5 integrity gate”)**

Certifies that κ₂(t)=Var(Δϕ(t)) slope-based regime classification (OU vs long-memory) is **not admissible under certain
inference/estimator pathologies**, even when the fitted κ₂ slope falls inside an OU/LM band.

This toy enforces a strict **C5 refusal rule**: any detected estimator-integrity violation must be labeled **BOUNDARY**,
never `OK_OU` / `OK_LM`.

It makes this concrete using fixed, declared C5 confounds:

- **Overlap reuse:** stride-1 moving-average smoothing reuses samples; if reuse exceeds a declared limit, κ₂ claims are refused.
- **Resample dt mismatch:** downsampling changes the true time step; integrating with the wrong dt can manufacture scaling; such cases are refused.
- **Differencing:** replacing ω with Δω changes the modeled observable; κ₂-slope regime attribution is refused by construction.

Within a declared admissible audit window:

- **C1_OU_BASE:** short-memory OU baseline → `OK_OU`.
- **C2_LM_TRUE:** uncoupled long-memory Gaussian baseline (fixed H) → `OK_LM`.
- **C5 confounds:** may produce in-band slopes, but must be refused → `BOUNDARY`.

This establishes a **C5 estimator-integrity boundary**: κ₂-slope regime certification requires not only admissible scaling,
but also an inference pipeline that does not manufacture or invalidate that scaling.

### **S-0023 — Transport-manufactured κ₂ scaling guard (“C3 integrity gate”)**

Certifies that late-time **κ₂(t) power-law scaling** can be rendered **non-admissible for C2 memory attribution**
when a declared **C3 geometric transport binding** (dispersion / mode-mixing style) imprints a detectable
transport signature onto the observed phase-rate stream.

This toy makes the integrity boundary explicit:

- It computes κ₂(t)=Var[Δϕ(t)] from ω(t) and estimates α by audited log-slope over a fixed window.
- It applies a declared, fixed **C3 dispersive LTI kernel** to an OU baseline to produce an observed ω̂(t).
- It enforces a **transport signature detector** (window-restricted autocovariance oscillation + negative-mass ratio).
- It verifies that:
  - **OU baseline** remains admissible and classifies as **OK_OU**.
  - **True long-memory baseline** remains admissible and classifies as **OK_LM**.
  - **C3 dispersion-filtered OU** is **refused** and labeled **BOUNDARY** when transport is detected, even if κ₂ scaling appears well fit.

This establishes the C3 integrity boundary: κ₂-slope is a valid C2 diagnostic only when transport signatures are absent,
preventing “C3-manufactured scaling” from being misread as long-memory.

### **S-0024 — Cross-observable κ₂-slope regime consistency (“Σ₂ multi-constraint admissibility gate”)**

Certifies that **κ₂-based regime attribution is admissible only when all
independent Σ₂ structural diagnostics agree.**

While earlier S-series items established individual refusal boundaries
(coupling, curvature, variance drift, aperture, and transport),
S-0024 unifies them into a single admissibility gate.

This toy verifies that:

- **OU-like diffusion and true long-memory baselines remain admissible** and are correctly tagged,
- **cross-trajectory coupling (C4)** is detected and refused,
- **temporal curvature (C1)** is detected and refused,
- **variance drift (C5)** is detected and refused,
- and **coherence-aperture violations** are detected and refused,

ensuring that κ₂-slope alone can never yield a false OK classification.

This establishes the first **multi-observable Σ₂ admissibility contract**
for phase-evolution inference in DFT.

### **S-0025 — Cross-window persistence of κ₂-slope regime admissibility (“Σ₂ continuation gate”)**

Certifies that κ₂-based regime attribution (OU-like vs long-memory-like) is admissible **only if it persists across
adjacent late-time continuation windows under the full Σ₂ guard bundle** (coupling, temporal curvature, variance drift,
and coherence-aperture budget).

This toy makes the continuation requirement explicit:

- It computes κ₂(t)=Var[Δϕ(t)] from ω(t) trajectories and fits log κ₂ vs log t in two fixed windows.
- It enforces that regime tags are allowed only when:
  - both window fits are admissible,
  - the inferred scaling exponent α remains consistent across windows,
  - and no Σ₂ guard fires in either window.
- It verifies that:
  - **C1_OU_BASE** persists as OU-like across windows → `OK_OU`.
  - **C2_LM_TRUE** persists as long-memory-like across windows → `OK_LM`.
  - Constructed violations (cross-trajectory coupling, temporal curvature, aperture excess) are refused as **BOUNDARY**,
    forbidding “false OK” under continuation.

This establishes a continuation-level integrity boundary: κ₂-slope classification is not merely window-local; it is admissible
only when stable under late-time continuation and consistent with the declared Σ₂ admissibility constraints.

### **S-0026 — Real-stream admissibility harness for κ₂-slope + Σ₂ guard bundle (“C5 deployment gate”)**

Certifies a *deployment harness* that applies the already-certified κ₂-slope classifier and Σ₂ refusal logic
to a **measurement-like** phase-rate stream ω̂(t), without introducing new regime semantics.

This toy enforces the core operational requirement for field use:

- **Baselines still certify:** independent OU and true long-memory Gaussian (fixed H) cases return `OK_OU` and `OK_LM`.
- **No false OK under C5:** when observation artifacts violate declared harness limits (irregular sampling, gaps, dt mismatch),
  the system must return **`BOUNDARY`** (never `OK_*`), even if κ₂-slope fits are numerically excellent.

Mechanically:

- ω̂(t) is snapped to a declared uniform analysis grid.
- A pseudo-ensemble is constructed by fixed block-resampling (epistemic only) to form κ₂(t).
- κ₂-slope is fit in two fixed late-time windows and must remain consistent under continuation.
- Any C5/harness violation forces refusal.

This establishes a minimal, auditable I/O and refusal contract for applying the S-series ladder to real streams,
without adding new assumptions about the ontology of ω.

### **S-0027 — Finite-N / finite-horizon robustness gate (no-scan stability menu)**

Certifies that the established κ₂-slope + guard ladder yields stable tags under a fixed robustness menu:
- canonical configuration,
- finite-N stress (reduced nT, same horizon),
- finite-horizon stress (reduced horizon, canonical nT),

with a fixed replicate bundle to reduce finite-N slope jitter without scanning.

PASS requires:
- OU_BASE stays **OK_OU** across all configs,
- LM_TRUE stays **OK_LM** across all configs,
- all constructed violation cases remain **BOUNDARY** across all configs.

This item establishes a practical minimum-robustness contract for deploying the classifier/guard ladder on finite data.

---

## Why this matters for DFT

Together, these results establish that:

- κ₂ closure is **not** a mathematical tautology.
- Its validity depends on **microscopic adjacency and mixing structure**.
- Its breakdown is **detectable, quantifiable, and categorical**.
- Higher-order cumulants (κ₃, κ₄) correspond to **distinct physical failure modes**
  (phase bias vs magnitude collapse).

This provides the **validation harness** required for the next step of the DFT
program: deriving phase-noise generators and envelopes as **theorems of
microscopic motion**, rather than calibrated ansätze.

---

## Reproducibility expectations (minimal)

Any listed success must provide:

- a concise, explicit statement of assumptions,
- a minimal runnable path (script-level reproducibility),
- and a checkable output artifact (figure and/or audit CSV) with stable naming,
  including provenance headers (git commit, run time, source hash).

No success is recognized without a **defined failure interpretation**.

---

## Falsification / breakpoints (required)

Every construction here must state what failure implies.

Examples:

- If κ₃ remains non-negligible under declared mixing bounds,
  the assumed adjacency decay is incorrect.
- If envelope scaling deviates from predicted κ₂(t) behavior,
  the ω-projection model is insufficient.
- If κ₂ fails to exist or diverges,
  the assumed stationarity or scalar-uniformity conditions are violated.
- If κ₄ drives magnitude collapse while GAUSS fails validity gates,
  the result is numerical/noise-floor limited and must be discarded.

These breakpoints are **diagnostic**, not post-hoc checks.

See `success_register.md` for per-item claims, assumptions, reproduction paths,
and explicit failure logic.

---

## Scope boundary (explicit)

This tool certifies **when envelope descriptions are valid and when they fail**.
It does **not** claim:

- that ω(t) is fundamentally stochastic,
- that OU or fGn processes are ontologically primary,
- that κ₂ closure holds universally,
- or that failure implies physical impossibility rather than structural mismatch.

All conclusions are conditional on stated assumptions and audited regimes.

No extrapolation beyond those regimes is implied.
