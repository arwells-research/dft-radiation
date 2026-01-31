# Phase-Evolution Theorems — Admissibility and Refusal (Audit-Level)

**Repository:** dft-radiation  
**Scope:** tools/phase-evolution toys (S-0001…S-0019)  
**Status:** v0.1 (audit theorems; normative, contract-facing)  
**Last updated:** 2026-01-18  

This document collects *audit-level theorems* for the **phase-evolution** toy suite.
These are not “physics derivations.” They are **normative soundness statements** about
what is (and is not) certifiable from windowed κ₂-scaling alone, under the declared
toy constructions.

A “theorem” here means:

- the statement is **stable under the current toy contracts**,
- it is supported by **constructive counterexamples** already present in the suite,
- and it is phrased as a **mechanical implication** (PASS/FAIL/BOUNDARY logic),
  not as ontology.

The theorems are intentionally *relative to declared guards* and *fixed audit windows*.

---

## Definitions (contract-level)

Let ωᵢ(t) be an ensemble of trajectories (i=1..nT), with phase accumulation

- Δϕᵢ(t) = ∫₀ᵗ ωᵢ(s) ds  (implemented as discrete cumulative sum with fixed Δt)

Define:

- κ₂(t) := Varᵢ[Δϕᵢ(t)]
- α := slope of log κ₂(t) vs log t over a **declared fixed late-time window** W

Let the classifier use fixed bands:

- OU band (short-memory): α ∈ [α_OU,min, α_OU,max]  → candidate OK_OU
- LM band (long-memory): α ∈ [α_LM,min, α_LM,max]  → candidate OK_LM

Let “admissible” mean the per-case fit meets declared gates (e.g. min_r2, min_points,
min_k2_end), and that any additional declared guards for that toy do not trigger.

Let “refusal guard” mean any declared window-restricted detector that can force
a non-OK outcome even if α lies in a certification band.

We use the house tags:

- OK_OU, OK_LM: certified classifications in their bands under guards
- BOUNDARY: refusal (no certification claim permitted)
- INCONCLUSIVE: no claim due to inadmissible fit / insufficient window / etc.

---

## Theorem T1 — κ₂-slope is not sufficient for regime attribution (existence of confounds)

**Statement.**  
There exist processes that are *not* genuine long-memory adjacency (C2-like in the
intended sense), yet yield α in the LM band within a fixed late-time window W.
Therefore, κ₂-slope alone is insufficient for C2 attribution.

**Constructive support (suite exhibits).**
- **S-0015:** cross-trajectory coupling (C4-style shared-mode injection) can
  produce LM-band α despite absent true long-memory adjacency.
- **S-0016:** non-stationary temporal curvature can produce LM-like α over W.
- **S-0017:** window-to-window inconsistency can yield a misleading α in one window.
- **S-0018:** strong non-Gaussian innovations can preserve OU/LM-band α.
- **S-0019:** deterministic variance drift can preserve admissible κ₂ scaling.

**Consequence.**  
Any certification rule of the form “α ∈ LM band ⇒ OK_LM” admits false positives.

---

## Theorem T2 — Any α-only certification rule is unsound (explicit false-OK construction)

**Statement.**  
Any classifier that certifies OK_OU or OK_LM based solely on α ∈ band (even with
admissibility gates on the fit quality of κ₂) is unsound: it admits at least one
constructive false-OK counterexample in the S-0015…S-0019 suite.

**Sketch (constructive).**
- Choose any α-only rule with fixed bands.
- Select a confound case whose α lies in the relevant band:
  - coupling case (S-0015), curvature case (S-0016), non-Gaussian case (S-0018),
    variance drift case (S-0019), or cross-window inconsistent case (S-0017).
- Because α lies in band and κ₂-fit admissibility can be satisfied, the α-only rule
  outputs OK_*.
- But the case violates the intended admissibility of the regime (as made explicit by
  the toy’s refusal logic), so OK_* is a false certification.

**Consequence.**  
To preserve “no wrong claims,” certification must incorporate at least one refusal guard
beyond κ₂-slope and κ₂-fit admissibility.

---

## Theorem T3 — Refusal-gated certification is sound relative to declared guards (“no false OK”)

**Statement.**  
Consider the certification scheme:

1. Require κ₂-fit admissibility in the declared window W.
2. Compute α and determine candidate band (OU / LM / neither).
3. Evaluate a fixed set of declared refusal guards G (window-restricted).
4. If any guard triggers → output BOUNDARY.
5. Else output OK_OU / OK_LM if α in corresponding band; otherwise BOUNDARY.

Then the scheme is sound **relative to the guard set G**: it forbids certification
on cases that the guards are designed to exclude, while still certifying the
uncoupled OU and uncoupled LM baselines when guards do not trigger.

**Constructive support (suite exhibits).**
- **S-0015:** variance-of-mean coupling ratio forces BOUNDARY under coupling;
  uncoupled baselines certify OK_OU and OK_LM.
- **S-0016:** curvature strength (quadratic fit on ω̄(t) in-window) forces BOUNDARY
  for curved cases; baselines certify OK_*.
- **S-0017:** cross-window α-consistency forces BOUNDARY for curved / regime-shifting
  cases; baselines remain stable.
- **S-0018:** excess-kurtosis guard forces BOUNDARY under heavy-tail innovations.
- **S-0019:** variance-drift z guard forces BOUNDARY under deterministic variance drift.

**Consequence.**  
Within the declared windows and guard definitions, the suite enforces the normative policy:
**“no false OK_* certification under known confounds.”**

**Interpretation boundary.**  
This theorem is *not* a claim that the guards are complete in nature; it is a claim that
the certification logic is internally sound with respect to the declared confound classes.

---

## Theorem T4 — Guarded certification is minimally constructive: each guard corresponds to a falsifying mode

**Statement.**  
Each refusal guard in S-0015…S-0019 corresponds to a distinct falsifying mode for α-based
classification. Removing the guard reintroduces a constructive false-OK possibility.

**Exhibits (guard → falsifying mode).**
- S-0015 coupling ratio → cross-trajectory shared-mode coupling (C4 confusion)
- S-0016 curvature metric → non-stationary curvature in ω̄(t) (C1 violation)
- S-0017 cross-window consistency → continuation instability / regime drift (C1 violation)
- S-0018 kurtosis guard → distributional non-Gaussian masquerade (innovation invalidity)
- S-0019 variance-drift z → variance nonstationarity masquerade (Σ₂-style budget drift)

**Consequence.**  
Guards are not decorative: each is tied to an explicit counterexample class that can
pass κ₂-slope bands.

**Note.**  
This theorem does **not** claim that the set of guards is complete. It claims that each
present guard is justified by a constructive falsifying mode already realized in-suite.

---

## Theorem T5 — “OK_*” is a conditional statement: OK implies both band membership and guard non-activation

**Statement.**  
In this framework, OK_OU and OK_LM are not pure band assertions. They are *conjunctions*:

- OK_OU ⇔ (α in OU band) ∧ (all refusal guards pass) ∧ (κ₂-fit admissible)
- OK_LM ⇔ (α in LM band) ∧ (all refusal guards pass) ∧ (κ₂-fit admissible)

**Consequence.**
- “OK_LM” is shorthand for “LM-band κ₂ scaling is admissible under declared confound guards.”
- A case can be LM-band by slope and still be BOUNDARY without contradiction.
- Refusal is a first-class output, not a failure: it is required for integrity.

---

## Operational corollary — What phase-evolution toys certify (and what they do not)

**Certified (operational).**
- There exist fixed-window κ₂-slope regimes (OU vs LM) that are separable on baselines.
- There exist confounds that can mimic those regimes in-window.
- Declared refusal guards can prevent false OK certification for the realized confounds.

**Not certified (out of scope).**
- Ontological claims about “true memory,” mechanism, or universal completeness of guards.
- Claims outside the declared windowing + admissibility contracts.
- Claims about arbitrary measurement pipelines beyond the declared C5 kernels (S-0014).

---

## Versioning note

This theorems document is **downstream** of the toy contracts. If the semantics of any toy
materially change (guard definition, admissibility criteria, output tags), then this document
must be reviewed and versioned accordingly (no silent mutation).

---

**End.**