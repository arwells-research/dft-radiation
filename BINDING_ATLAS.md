# Binding Atlas — Radiation Phenomenology

**Version:** 0.1.0  
**Last updated:** 2026-01-14  
**Phenomena catalogued:** 28  
**Constraint combinations observed:** 17  

This document is the **Binding Atlas** for the `dft-radiation` repository.

Its purpose is to make the **constraint-completeness claim inspectable** by explicitly mapping
well-known radiation phenomena to their **constraint signatures** within the C1–C5 framework.

The atlas serves three functions simultaneously:

1. **Navigation** — allowing readers to locate where a phenomenon lives in the constraint map  
2. **Auditability** — making it explicit which constraints are exercised by which phenomena  
3. **Stress-testing** — exposing gaps, ambiguities, or over-concentration in constraint usage  

This is a *classification artifact*, not a derivation or tutorial.

---

## How to read this document

Each phenomenon is represented by a **constraint signature**: the set of constraint classes
(C1–C5) it instantiates.

- Constraint signatures do **not** imply mechanism or interpretation  
- They indicate *which aspects of phase evolution are constrained*  
- A phenomenon may bind to multiple classes simultaneously  

Operational classification tests are defined in  
`foundations/constraint_classes.md`.

---

## Section I — Phenomenon → Constraint Signature (Reverse Index)

This section answers:  
**“I work on X — where does it live in this framework?”**

| Phenomenon | Constraint signature | Primary | Notes |
|-----------|---------------------|---------|-------|
| Monochromatic radiation | C1 | C1 | Fixed phase rate |
| Chirped pulse | C1 | C1 | Time-dependent ω(t) |
| Spectral linewidth | C1 | C1 | Temporal phase spread |
| Doppler redshift | C1 | C1 | Kinematic phase-rate change |
| First-order coherence (G¹) | C2 | C2 | Field self-correlation |
| Second-order coherence (G², g²) | C2 | C2 | Coincidence statistics |
| Antibunching | C2 | C2 | Non-classical statistics |
| Bell test (photonic) | C2 + C5 | C2 | Correlation + inference |
| HOM interference | C2 + C5 | C2 | Two-photon correlation |
| Polarization | C3 | C3 | Vector phase geometry |
| Orbital angular momentum (OAM) | C3 | C3 | Helical phase structure |
| Birefringence | C3 | C3 | Mode-dependent delay |
| Dispersion in matter | C3 | C3 | Frequency-dependent transport |
| Gravitational lensing | C3 | C3 | Path-dependent curvature |
| Spontaneous emission | C4 | C4 | Scalar → phase interface |
| Stimulated emission | C4 | C4 | Interface with feedback |
| Photoelectric effect | C4 | C4 | Phase → scalar interface |
| Parametric down-conversion | C4 + C2 | C4 | Nonlinear interface + entanglement |
| Superradiance | C2 + C4 | C2 | Collective emission |
| Cavity QED | C3 + C4 + C5 | C3 | Mode structure dominates |
| RADAR ranging | C1 + C3 + C5 | C1 | Phase rate + transport + inference |
| LIDAR | C1 + C3 + C5 | C1 | Same signature as RADAR |
| Interferometry (Michelson) | C3 + C5 | C3 | Path difference inference |
| Quantum eraser | C2 + C5 | C2 | Correlation manipulation † |
| Raman scattering | C3 + C4 | C4 | Inelastic interface |
| Synchrotron radiation | C1 + C3 + C4 | C4 | Accelerated emission |
| Thermal radiation | C1 + C2 + C4 | C4 | Statistical emission |
| Coherent backscattering | C2 + C3 | C3 | Correlated transport |
| Optical solitons | C1 + C3 + C4 | C3 | Nonlinear phase balance † |

† See `CHALLENGE_LOG.md` for boundary-clarification discussion.

---

## Section II — Constraint → Phenomena (Forward Index)

This section answers:  
**“What kinds of phenomena exercise a given constraint?”**

### C1 — Temporal Constraint (Phase Rate)
- monochromatic radiation (laser stabilization)
- spectral linewidth and bandwidth (Fourier-limited pulses)
- chirp (pulse compression)
- Doppler and gravitational redshift

### C2 — Statistical Constraint (Phase Correlation)
- classical and quantum coherence (G¹, G²)
- bunching / antibunching
- Bell correlations
- HOM interference
- superradiance

### C3 — Geometric Transport (Phase Transport)
- polarization
- orbital angular momentum
- wavefront shaping
- dispersion and birefringence
- gravitational lensing
- coherent backscattering

### C4 — Interface Constraint (Phase–Source Interaction)
- spontaneous and stimulated emission
- absorption and gain
- photoelectric transitions
- parametric frequency conversion
- thermal radiation
- synchrotron radiation

### C5 — Epistemic Constraint (Phase Observation)
- detector response
- windowing and filtering
- SNR-limited inference
- coincidence detection
- estimator bias and resolution

---

## Section III — Constraint-Combination Catalog

This section catalogs **observed combinations** of constraint classes and representative
phenomena.

The purpose is to make **coverage density visible** and highlight potential gaps.

| Constraint combination | Representative phenomena |
|-----------------------|--------------------------|
| C1 | monochromatic radiation |
| C2 | antibunching |
| C3 | polarization |
| C4 | spontaneous emission |
| C5 | SNR-limited spectroscopy |
| C1 + C2 | linewidth statistics |
| C1 + C3 | dispersive pulse propagation |
| C1 + C5 | spectral estimation |
| C2 + C5 | Bell tests, HOM |
| C2 + C4 | superradiance |
| C2 + C3 | coherent backscattering |
| C3 + C5 | interferometry |
| C3 + C4 | Raman scattering |
| C1 + C3 + C4 | synchrotron radiation |
| C1 + C3 + C5 | RADAR / LIDAR |
| C1 + C2 + C4 | thermal radiation |
| C3 + C4 + C5 | cavity QED |

### Interpretive note

C1–C3 frequently appear alone, reflecting **intrinsic properties of radiation in flight**.
C4 and C5 rarely appear in isolation, as radiation must be both **created** (C4) and
**observed** (C5) to be studied. Their presence primarily as boundary conditions is
physically expected.

### Unpopulated combinations

Some combinations (e.g. C1 + C4, C4 alone, C5 alone) remain unpopulated. This may indicate:

- physical inadmissibility,
- phenomena not yet classified,
- or redundancy that could motivate future refinement.

Distinguishing these cases is an explicit part of framework validation.

---

## Scope and limitations

- This atlas does **not** assert mechanistic explanations  
- It does **not** imply equivalence between phenomena sharing signatures  
- It does **not** preclude multiple valid bindings for a single phenomenon  

Its sole role is to make the **constraint structure explicit and inspectable**.

---

## Relationship to Challenge Log

Any phenomenon that:
- is difficult to classify,
- requires boundary clarification,
- or initially appears to require a new constraint class

**must** be logged in `CHALLENGE_LOG.md`.

The Binding Atlas reflects *resolved classifications*; the Challenge Log documents
the process of resolution.

---

## Status

This document is under active population.

Additions to this atlas are treated as **tests of completeness**, not expansions of scope.
Proposed phenomena that resist classification are considered potential falsification
events and must be documented.

---

**Binding Atlas**  
*dft-radiation — making constraint completeness inspectable*