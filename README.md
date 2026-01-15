# dft-radiation

**Dual-Frame Theory (DFT) — Radiation Phenomenology and Constraint Coverage**

This repository provides a **constraint-complete phenomenological reference** for radiation
within **Dual-Frame Theory (DFT)**.

Its purpose is to construct a **deliberate, non-ad-hoc bridge** between DFT’s foundational
postulates (scalar progression, Temporal/Spatial frame duality) and the observable properties
of radiation, organized by *what constrains phase evolution* rather than by historical topic
or experimental convenience.

This repository is intended to be **read structurally**, not sequentially.

Throughout this repository, “radiation” refers to propagating phase-carrying fields whose observable properties are exhausted by constraints on phase evolution.

---

## Scope and intent

### What this repository is

- A **phenomenological coverage module** for radiation in DFT  
- A **topological map** of radiation phenomena organized by constraint class  
- A reference layer that connects:
  - foundational definitions,
  - microscopic phase evolution tools,
  - and experimentally observed radiation properties

It demonstrates that radiation phenomenology can be covered **completely and finitely**
using a small set of irreducible phase constraints.

### What this repository is not

- Not an optics or EM textbook  
- Not a catalog of simulations  
- Not a single theory paper  
- Not an engineering signal-processing manual  

Individual derivations, numerical gates, and executable tools are present only insofar as
they support the constraint structure.

---

## Organizing principle

In DFT, all observable properties of radiation arise from **constraints on phase evolution**.

Rather than grouping material by topic (e.g. “optics,” “quantum,” “radar”), this repository
is organized around **five irreducible constraint classes**, each corresponding to a distinct
way phase can be constrained, transported, or inferred.

If a phenomenon does not require introduction of a new constraint class, it is treated as a binding within the existing structure, not as a structural extension.

---

## Repository structure (how to read this)

### 1. Foundations

```
foundations/
```

Defines the primitives and classification rules used everywhere else.

Key document:
- `constraint_classes.md` — the authoritative definition of the five constraint classes (C1–C5),
  including **operational classification tests** and admissibility criteria

This layer is intentionally small and stable.

---

### 2. Coverage (the finite map)

```
coverage/
```

This is the **core of the repository**.

Each subdirectory corresponds to one pre-declared constraint class:

- C1 — Temporal Constraint (Phase Rate)
- C2 — Statistical Constraint (Phase Correlation)
- C3 — Geometric Transport (Phase Transport)
- C4 — Interface Constraint (Phase–Source Interaction)
- C5 — Epistemic Constraint (Phase Observation)

Each coverage entry documents:
- what aspect of phase is constrained,
- which observables belong there,
- when coverage is considered closed,
- and the fundamental physical boundary of that class.

This layer is **finite by design**. Empty or partial entries indicate planned scope, not gaps.

---

### 3. Phenomena (bindings)

```
phenomena/
```

Phenomena such as radiation, HOM interference, antibunching, Bell correlations, RADAR, etc.,
are treated as **bindings** to the coverage layer.

A phenomenon:
- does not introduce new constraint classes,
- instead instantiates one or more existing ones,
- and explicitly declares its **mapped constraints**.

This enforces the non-ad-hoc structure during normal use and browsing.

---

### 4. Theorems and tools

```
theorems/
tools/
```

These directories provide the **constructive bridge** between DFT primitives and coverage:

- `theorems/` contains the narrative and formal claims (A0–A4 ladder)
- `tools/` contains executable mechanisms and S-000x toy models that validate those claims

They are included here for completeness, but may later be factored into a standalone tools
repository without changing the phenomenological structure.

---

## Constraint classes at a glance

| Class | Name | What is constrained |
|-----|------|---------------------|
| C1 | Temporal | Phase rate ω(t), frequency, chirp |
| C2 | Statistical | Phase correlations and joint distributions |
| C3 | Geometric Transport | Spatial and modal phase transport |
| C4 | Interface | Source ↔ radiation phase handshake |
| C5 | Epistemic | Observation, inference, information limits |

Every radiation phenomenon addressed in this repository maps to one or more of these classes.

---

## Completeness claim (radiation only)

For radiation, the constraint classes C1–C5 form a **complete and non-redundant set**.

Any proposed “property of radiation” must:
- instantiate one or more of these constraints, or
- be a derived or compound observable built from them.

No additional radiation-specific constraint classes are introduced in this framework.

This is a **coverage claim**, not a claim of final derivation for all entries.

---

## Coverage maturity

Coverage is declared **complete by construction** at the level of constraint classes.
Individual coverage entries may be at different stages of formal development.

| Constraint class | Formalized | Bounded | Placeholder |
|-----------------|------------|---------|-------------|
| C1 — Temporal | ✔ | ☐ | ☐ |
| C2 — Statistical | ✔ | ☐ | ☐ |
| C3 — Geometric Transport | ☐ | ✔ | ☐ |
| C4 — Interface | ☐ | ✔ | ☐ |
| C5 — Epistemic | ✔ | ☐ | ☐ |

This table tracks documentation and derivational maturity only. It does not imply provisional status of any constraint class.

- **Formalized**: derivations and tools are in place  
- **Bounded**: limits and admissibility are defined, full derivation pending  
- **Placeholder**: scope reserved, treatment pending  

This status reflects *documentation maturity*, not theoretical legitimacy.

---

## Auditability and stress-testing

Completeness is made **inspectable and testable** via two public artifacts:

- **Binding Atlas (`BINDING_ATLAS.md`)**  
  Maps constraint classes and their combinations to known radiation phenomena,
  including a reverse index from phenomena to constraint signatures.

- **Challenge Log (`CHALLENGE_LOG.md`)**  
  Documents classification edge cases, boundary clarifications, and the resulting
  updates to the framework.

These artifacts are intended to invite informed scrutiny rather than deflect it.

---

## How to use this repository

- **To classify a phenomenon:**  
  Identify which aspect of phase evolution is constrained (rate, correlation, geometry,
  interface, or observation), then navigate to the corresponding constraint class in
  `coverage/`. Operational classification tests are defined in
  `foundations/constraint_classes.md`.

- **New readers:** start with `foundations/constraint_classes.md`, then browse `coverage/`
- **Experimentalists:** locate your observable in `coverage/`, then see which phenomena bind to it
- **Theorists:** follow links from coverage entries to `theorems/` and `tools/`
- **Reviewers:** inspect the finite scope, boundaries, and Challenge Log history

---

## Revision and admissibility policy

The constraint set C1–C5 is asserted complete for radiation phenomenology **as of this release**.

Discovery of a genuine radiation phenomenon that cannot be expressed as a binding of C1–C5
without introducing a new fundamental phase-constraint dimension or a non-phase degree of
freedom essential to radiation itself would constitute a major theoretical development and
would trigger a **documented revision** of this framework, not an unannounced extension.

---

## Status

This repository is under active construction.

Coverage slots are pre-declared intentionally; some entries are placeholders awaiting formal
derivation or diagnostic treatment. This reflects planned scope and staged development, not
post-hoc expansion.

Coverage placeholders are filled as derivations and diagnostics stabilize; no timeline commitments are implied.

---

## License and citation

See `LICENSE` and `CITATION.cff`.

All material is released under CC BY 4.0 unless otherwise noted.

---

**dft-radiation**  
*A constraint-complete map of radiation phenomenology in Dual-Frame Theory*