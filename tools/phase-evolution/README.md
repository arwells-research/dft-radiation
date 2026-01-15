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