# Constraint Classes for Radiation Phenomenology

This document defines the **constraint classes** used to organize radiation phenomena within
**Dual-Frame Theory (DFT)**.

It assumes familiarity with DFT primitives such as **scalar progression** and
**Temporal/Spatial frame duality**. For foundational context, see the DFT core documents.
The focus here is phenomenological classification, not axiomatic derivation.

The purpose of these classes is **not** to catalog historical physics topics, but to enumerate
the *irreducible ways* in which the phase variable of radiation can be constrained, transported,
or inferred. Together, they form a **complete and non-ad-hoc phenomenological map** linking
DFT’s foundational postulates to observable radiation phenomena.

Each constraint class is defined by:
- what aspect of phase evolution it constrains,
- a **coverage closure criterion** (when investigation is complete),
- and a **physical boundary** marking the limit of applicability.

No additional radiation-specific constraint classes are permitted without introducing a new
primitive variable or a new projection dimension beyond the DFT framework.

---

## Overview

In DFT, radiation is understood as **phase evolution arising from scalar progression**
projected into spacetime.

All observable properties of radiation arise from constraints on one or more of the following
five aspects of phase evolution:

1. **Temporal rate** of phase advance  
2. **Statistical relationships** between phase values  
3. **Geometric distribution and transport** of phase across space and modes  
4. **Interface dynamics** between non-radiative states and radiative phase  
5. **Epistemic projection** of phase into measured data  

These correspond to the five constraint classes **C1–C5** defined below.

---

## C1 — Temporal Constraint (Phase Rate)

### Definition  
Governs how the phase value evolves along a linear time axis.

This class constrains the **instantaneous and time-dependent rate of phase advance**,
ω(t) = dφ/dt.

### Covered Phenomena  
Examples include:
- mean frequency
- frequency drift and stability
- spectral linewidth and bandwidth
- chirp (time-dependent frequency acceleration)
- redshift (kinematic, gravitational, cosmological)

All such phenomena are manifestations of how rapidly, and how smoothly, phase advances in time.

### Coverage Closure  
The temporal investigation is **closed** if the phase rate ω(t) is fully specified,
including its time dependence, stability, and admissible fluctuations.

Any additional temporal descriptor must reduce to a reparameterization of ω(t).

### Boundary  
The fundamental boundary of C1 is the **time–energy uncertainty limit**:

ΔE · Δt ≥ ħ / 2

Beyond this limit, further temporal refinement of phase evolution is physically inadmissible.

---

## C2 — Statistical Constraint (Phase Correlation)

### Definition  
Governs the relationship between two or more phase values at distinct times, events,
or subsystems.

This class constrains the **joint statistical structure** of phase: how φ₁ and φ₂
(or higher-order sets) co-vary.

### Covered Phenomena  
Examples include:
- first-order (classical) coherence functions (G¹)
- second-order and higher correlations (G², g²(τ))
- bunching and antibunching
- Bell correlations and nonlocal phase statistics
- radiation-mediated source–source correlations
  (e.g. superradiance, collective emission)

C2 encompasses both **field self-correlations** and **correlations induced between distinct
emitters via radiation**. The distinction lies in *what is being correlated*, not in the
constraint class.

### Coverage Closure  
The statistical investigation is **closed** if the joint probability distribution of
phase values is fully specified for the observables of interest.

Higher-order correlations do not constitute new constraint types; they are refinements within C2.

### Boundary  
The fundamental boundary of C2 is reached at **Bell-inequality violation**, marking the
maximum physically admissible nonlocal correlation.

---

## C3 — Geometric Transport (Phase Transport)

### Definition  
Governs the distribution and transport of phase across spatial coordinates and modal structure.

This class constrains **how phase propagates through space**, including its organization across
transverse dimensions, polarization states, spatial modes, and propagation paths.

### Covered Phenomena  
Examples include:
- polarization (vector phase structure)
- orbital angular momentum (helical phase)
- spatial coherence and wavefront shape
- dispersion and birefringence (phase accumulated during propagation in matter)
- gravitational lensing and path-dependent phase curvature

All such effects arise from the geometry of phase transport, not from its temporal rate
or statistical structure.

### Coverage Closure  
The transport investigation is **closed** if the spatial phase distribution and mode structure
are fully specified along the propagation path.

### Boundary  
The fundamental boundary of C3 is the **diffraction limit**, arising from the finite wavelength
of radiative phase transport and setting the minimum resolvable spatial phase structure.

---

## C4 — Interface Constraint (Phase–Source Interaction)

### Definition  
Governs the transition between non-radiative scalar states and radiative phase evolution.

This class constrains the **interface (“handshake”)** by which scalar progression is admitted
into, or removed from, radiative phase channels.

### Covered Phenomena  
Examples include:
- emission delay and timing jitter
- quantum yield (radiative vs non-radiative branching)
- radiation reaction and back-action
- absorption, gain, and photoelectric transitions
- spontaneous vs stimulated emission
- nonlinear emission processes (e.g. parametric down-conversion)

Absorption, gain, and scattering processes that exchange radiative and non-radiative states
belong to C4, even when embedded within extended propagation scenarios.

Mechanism-specific emission models (e.g. transition rates, selection rules) are **not**
treated as independent constraint classes. They emerge from the structure and strength of
the C4 interface coupling, combined with C1 spectral constraints and source state preparation.
Such models are treated as **bindings** that instantiate C4 together with other constraints.

### Coverage Closure  
The interface investigation is **closed** if the coupling strength, branching ratios, and
temporal characteristics of the radiative channel are specified.

### Boundary  
The fundamental boundary of C4 is reached at **vacuum fluctuations / zero-point coupling**,
where the notion of an isolated source ceases to be meaningful.

---

## C5 — Epistemic Constraint (Phase Observation)

### Definition  
Governs the transformation of physical phase information into an inferred data set.

This class constrains **how phase is observed**, including all distortions introduced by
measurement, detection, and inference.

### Covered Phenomena  
Examples include:
- temporal and spectral windowing
- detector response and bandwidth
- signal-to-noise ratio (SNR)
- estimator bias and resolution limits
- inferred vs intrinsic linewidth

These effects do not alter the underlying phase evolution, but determine what can be known
about it.

### Coverage Closure  
The observational investigation is **closed** if all measurement-induced distortions are
explicitly accounted for in the inference pipeline.

### Boundary  
The fundamental boundary of C5 is set by **information capacity**, expressed via
**Shannon entropy limits on distinguishable states**.

---

## Operational Classification Tests

The following tests provide a **mechanical procedure** for classifying phenomena within
this framework:

### Temporal vs Statistical (C1 vs C2)
- *Could the property be measured from a single, perfect field realization?*  
  Yes → C1  
  No; requires ensembles, coincidences, or joint statistics → C2  

**Note:** Properties derivable from Fourier analysis of a single realization
(e.g. spectrum, bandwidth) belong to C1. Properties requiring phase comparison across
realizations or subsystems (e.g. coherence time, g²) belong to C2.

### Transport vs Interface (C3 vs C4)
- *Is phase preserved as radiative phase, or exchanged with non-radiative degrees of freedom?*  
  Preserved → C3  
  Exchanged → C4

### Transport vs Observation (C3 vs C5)
- *If the detector were removed, would the effect still exist?*  
  Yes → not C5

### Interface vs Observation (C4 vs C5)
- *Could a better detector eliminate the effect?*  
  Yes → C5  
  No; it is a physical coupling → C4

These tests are **normative for classification of radiation phenomena within this framework**
and must be applied before introducing new structure.

---

## Admissibility and Revision Clause

The constraint set **C1–C5** is asserted complete for **radiation phenomenology**
**as of 2026-01-18**.

As of this revision, the **κ₂-based phase-evolution admissibility program has been
mechanized and closed via the S-0001 through S-0019 validation series.**

These results establish the empirical and diagnostic refusal boundaries for:

- C1 temporal rate admissibility
- C2 statistical (memory vs Gaussianity vs coupling) admissibility
- C4 interface coupling identifiability
- C5 epistemic (kernel masking vs estimator limits) admissibility

Discovery of a genuine radiation phenomenon that cannot be expressed as a binding of
C1–C5 without:

(a) introducing a new **fundamental phase-constraint dimension**, or  
(b) requiring a **non-phase degree of freedom essential to radiation itself**

would constitute a major theoretical development and would trigger a **documented revision**
of this framework, not an unannounced extension.

Such a discovery would represent the introduction of a true **C6** and must be treated as
a versioned event.

---

## Completeness Statement

The five constraint classes C1–C5 form a **complete and non-redundant set** for
phase-based radiation phenomenology within DFT.

They correspond to the fundamental aspects of phase evolution:
- its temporal rate,
- its statistical structure,
- its geometric transport,
- its boundary conditions of creation and destruction,
- and its observability.

Any proposed “property of radiation” must be expressible as an instantiation or combination
of these constraints. Phenomena that do not introduce a new constraint class are treated as
**bindings**, not structural extensions.

This guarantees that the phenomenological program remains finite, principled, falsifiable,
and auditable.