# Tools — Radiation Phenomenology

This directory contains **phenomenological tools** for the `dft-radiation` repository.

These tools provide **constructive bridges** between the C1–C5 radiation constraint
framework and **observable phase behavior** (spectra, coherence, correlations,
and inference effects). They operationalize the framework without introducing
new primitives or mechanisms.

The tools here are **not part of the DFT core axioms** and are **not engineering utilities**.
They exist to:

- demonstrate **closure conditions** for constraint classes,
- generate **phase-based diagnostics** (envelopes, correlators, spectra),
- expose **boundary behavior** and refusal regimes,
- support classifications recorded in the **Binding Atlas** and **Challenge Log**.

These tools make the radiation program **constructive**, not merely classificatory.

---

## What qualifies as a tool

A directory appears here **only if** it satisfies all of the following:

- Makes a **checkable phenomenological claim** tied explicitly to one or more
  radiation constraint classes (C1–C5)
- Operates under **explicit assumptions** (stationarity, mixing, locality, etc.)
- Defines **failure implications** (what breaks if the construction fails)
- Demonstrates **non-emptiness** (the construction actually produces structure)
- Avoids free parameter fitting, tuning to data, or post-hoc justification
- Does **not** introduce new radiation primitives or constraint classes

Anything exploratory, provisional, or unconstrained does **not** belong here.

---

## Relationship to the framework

The role of tools in `dft-radiation` is deliberately narrow:

- They do **not** prove the constraint classes
- They do **not** derive radiation from first principles
- They do **not** replace domain-specific theory (optics, QED, etc.)

Instead, they answer questions of the form:

- *Given C1–C5, what phase behavior is admissible?*
- *What envelope forms, correlators, or limits necessarily follow?*
- *Where do boundaries (uncertainty, diffraction, information limits) appear constructively?*

In this sense, tools form a **binding layer** between:
- abstract constraint definitions, and
- concrete radiation phenomenology.

---

## Active tools

### `phase-evolution/`

**A minimal constructive program explaining why phase decoherence envelopes exist
and why their form is constrained.**

This tool provides a bridge from DFT’s motion-first perspective to
microscopic **phase evolution models** used across radiation phenomena
(coherence decay, antibunching, interference, correlation tests).

Scope:

- Treats the phase-advance rate **ω(t)** as a projection of scalar progression
- Models phase as an **accumulated constraint**, not a state variable
- Derives coherence envelopes via **cumulant structure**
- Classifies envelope families by **correlation decay and mixing assumptions**
- Enforces **explicit falsification breakpoints** tied to constraint closure

This directory is intentionally **theory-forward and code-light**.
Executable content, where present, exists solely to demonstrate
registered successes or boundary failures and is subordinate to
the stated assumptions and interpretations.

---

## Status and expectations

Tools in this directory are **stable but not exhaustive**.
New tools may be added only when they meet the same verification
and non-ad-hoc standards.

The presence of a tool indicates **bounded success**, not completeness.
All tools are subject to future stress-testing via the Challenge Log.

Verified outcomes from this program are recorded in `success_register.md`.
Cases that fail, remain exploratory, or do not satisfy explicit success criteria
are retained locally and do not propagate upward into theorems, coverage claims,
or the Binding Atlas.

---

**Tools — dft-radiation**  
*Constructive bridges from phase constraints to observable radiation behavior*