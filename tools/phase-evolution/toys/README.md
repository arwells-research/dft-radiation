# Toy models (Phase Evolution)

Toy scripts under this directory are **self-auditing** and are the only executable artifacts
permitted to produce local outputs (which must remain untracked under `toys/outputs/`).

Each toy prints PASS/FAIL banners and returns a nonzero exit code on failure.

## Registered / canonical toy scripts

- `s0001_k2_scaling_demo.py`
  κ₂ scaling classification (OU vs long-memory). Audits log κ₂ vs log t slope in declared windows.

- `s0002_k2_prediction_demo.py`
  κ₂ predictive closure test (autocovariance → κ₂ → envelope). fGn passes; OU fails without structural truncation.

- `s0003_k3_gaussian_boundary_demo.py`
  κ₃ boundary: phase-bias signature with κ₂ magnitude closure retained.

- `s0004_k4_even_cumulant_boundary_demo.py`
  κ₄ boundary: even-cumulant magnitude collapse falsifies κ₂-only closure.

- `s0004_sensitivity_sigma_scale.py`
  Sensitivity sweep that brackets κ₄ boundary onset by ρ(t*) criterion.

- `s0005_c1_c2_c5_windowing_demo.py`
  Cross-constraint closure (C1 ⟷ C2 ⟷ C5): measurement kernel on ω (window + detector LPF + SNR)
  can degrade estimates but cannot fake OU ↔ long-memory regime transition when the κ₂-slope
  discriminant is applied only in an admissible scaling window.

- `s0006_radiation_omega_generators.py`
  Radiation ω generator families (C1 ⟷ C2 ⟷ C3 ⟷ C4): κ₂-slope classification is stable across a
  declared menu of ω(t) generators; C3 filtering and C4 shot injection do not change κ₂ scaling class
  on any admissible case.

- `s0007_mixture_crossover_audit.py`
  Mixture + crossover restraint (OU ⊕ fGn; early/late windows): verifies that the classifier makes
  no wrong claims under declared mixtures and returns BOUNDARY where scaling is ambiguous.

- `s0008_drift_vs_diffusion_demo.py`
  Drift-vs-diffusion confound: certifies that κ₂-slope ≈ 1 is insufficient to distinguish diffusion
  from per-trajectory linear drift; requires additional diagnostics beyond κ₂ scaling.

- `s0009_k5_higher_cumulant_boundary_demo.py`
  κ₅ odd-cumulant boundary: κ₅ introduces a sign-reversible phase bias in ⟨exp(iΔϕ(t))⟩ while magnitude
  remains consistent with even-cumulant truncation over a fixed window.

- `s0010_c3_vs_c4_k2_transport_interface_demo.py`
  C3 ⟷ C4 identifiability boundary: κ₂-slope remains invariant under declared C3 (linear transport filters)
  and declared stationary C4 injections; κ₂-slope alone cannot separate C3 from C4.

- `s0011_c3_c4_kurtosis_separation_demo.py`
  C3 ⟷ C4 separability via ω-tail structure: while κ₂-slope remains invariant (S-0010),
  excess kurtosis of ω (g₂) stays ~0 for OU/C3 but becomes strongly positive under declared C4 shot injection.

- `s0012_k6_even_cumulant_boundary_demo.py`
  κ₆ structural boundary: with κ₂ and κ₄ held fixed, controlled sixth-cumulant forcing
  produces an even-order deformation in the coherence envelope that is not captured by
  κ₂+κ₄ closure alone. GAUSS, K6+, and K6− remain admissible and preserve κ₂-slope class.

- `s0013_k6_magnitude_sufficiency_boundary_demo.py`
  κ₆ magnitude sufficiency boundary: under strong κ₆ forcing (heavy-tail innovations with
  κ₄≈0), the κ₂+κ₄ magnitude-closure model fails reproducibly in the declared audit window,
  while GAUSS and mild κ₆ cases pass. This certifies the first point at which higher-order
  even cumulants falsify magnitude closure rather than only deforming phase structure.

- `s0014_c5_kernels_hide_higher_cumulants_demo.py`
  C5 masking boundary: certifies that measurement kernels (finite averaging + noise)
  can hide higher-cumulant structure from observation, but must never produce a false
  PASS. When identifiability is lost (strong kernel), the result is forced to BOUNDARY;
  when identifiability is retained (mild kernel), κ₂+κ₄ magnitude closure must fail for
  heavy κ₆ contamination and pass for the GAUSS baseline.

- `s0015_c2_vs_c4_cross_trajectory_confusion_demo.py`
  C2 ⟷ C4 classification integrity: certifies that **cross-trajectory coupling** (C4-style
  shared/interface structure) can produce misleading κ₂(t) scaling, so the classifier must
  enforce a **refusal gate**. Uses a window-restricted **variance-of-mean ratio**
  r_mean = Var_t( mean_i ωᵢ(t) ) / mean_i Var_t( ωᵢ(t) ) to detect coupling and force
  **BOUNDARY** (never OK_*) when coupling is present, while still correctly certifying
  **OK_OU** and **OK_LM** on uncoupled baselines in the declared audit window.

- `s0016_temporal_curvature_boundary_demo.py`
  C1 curvature admissibility guard: certifies that **non-stationary temporal curvature** in ω(t)
  can mimic long-memory-like κ₂(t) scaling within a finite audit window, so κ₂-slope attribution
  must enforce a **refusal gate**. Fits the window-restricted ensemble mean ω̄(t) to a quadratic
  ω̄(t)≈a+b·t+c·t² and uses a dimensionless **quadratic significance detector**
  `curv_z = |c| / SE(c)` to flag curvature and force **BOUNDARY** (never OK_*) when curvature is
  detected, while still correctly certifying **OK_OU** and **OK_LM** on locally stationary baselines.

- `s0017_cross_window_regime_consistency_demo.py`
  Cross-window κ₂-slope regime consistency (C1 guard): certifies that κ₂-slope
  classification is admissible only when the inferred scaling exponent α remains
  stable across adjacent late-time continuation windows. Independent OU and true
  long-memory baselines pass (OK_OU and OK_LM), while temporally curved processes
  are refused and labeled **BOUNDARY** rather than misclassified.

- `s0018_non_gaussian_masquerade_boundary_demo.py`
  Non-Gaussian masquerade boundary: certifies that **κ₂-slope scaling can match OU/LM bands** while ω remains
  strongly non-Gaussian. Enforces a window-restricted **excess-kurtosis guard** on ω (g₂) and forces **BOUNDARY**
  whenever non-Gaussianity is detected, forbidding false OK_OU/OK_LM certification.

- `s0019_variance_drift_masquerade_boundary_demo.py`
  Variance-drift masquerade boundary (Σ₂ guard): certifies that **κ₂-slope scaling
  is admissible only when the variance budget of ω(t) is locally stationary in the
  declared audit window.** Independent OU and true long-memory Gaussian baselines
  pass (OK_OU and OK_LM). A deterministic variance-drift process that matches the
  long-memory κ₂ band is detected and refused, and is labeled **BOUNDARY** rather
  than misclassified — enforcing “no false OK” under variance drift.

- `s0020_winding_sector_admissibility_demo.py`
  Winding-sector admissibility gate (derivational): certifies that **κ₂(t) scaling is
  not sufficient for regime attribution unless the implied T-frame phase progression
  remains admissible under winding-sector continuation.** Independent OU and true
  long-memory Gaussian baselines pass (OK_OU and OK_LM). A constructed case that
  matches κ₂-slope locally but violates winding-sector continuity is refused and
  labeled **BOUNDARY**, enforcing “no false OK_LM” under sector violation.

- `s0021_k2_vs_aperture_admissibility_demo.py`
  Coherence-aperture admissibility gate (derivational): certifies that **κ₂(t) scaling is
  not sufficient for regime attribution unless the implied phase-spread budget remains
  within the declared coherence aperture L over the audit window.** Independent OU and
  true long-memory Gaussian baselines pass (OK_OU and OK_LM) with `eta_max ≤ 1`, where
  `eta(t)=sqrt(κ₂(t))/L`. A constructed aperture-excess case preserves κ₂-slope class
  but forces `eta_max > 1` and is refused and labeled **BOUNDARY**, enforcing “no false OK_*”
  under coherence-aperture violation.

- `s0022_estimator_manufactured_scaling_guard.py`
  Estimator-manufactured scaling refusal gate (C5 integrity): certifies that **κ₂(t) slope-based regime
  attribution is not admissible under declared inference/estimator pathologies**, even when fitted α
  falls in OU/LM bands. Independent OU and true long-memory Gaussian baselines pass (OK_OU and OK_LM).
  Fixed C5 confounds (overlap reuse, resample dt mismatch, and differencing) are forced to **BOUNDARY**
  (never OK_*), enforcing “no false OK” under estimator-manufactured scaling.

- `s0023_transport_masquerade_guard.py`
  Transport-manufactured κ₂ scaling guard (C3 integrity): certifies that **κ₂-slope regime attribution is
  not admissible when a declared C3 dispersive transport binding imprints a detectable transport signature**
  in the audit window. Independent OU and true long-memory Gaussian baselines pass (OK_OU and OK_LM).
  A dispersion-filtered OU case is forced to **BOUNDARY** when transport is detected, enforcing “no false OK_*”
  under C3-induced scaling masquerade.

- `s0024_cross_observable_consistency_guard.py`
  Cross-observable admissibility gate: certifies that **κ₂-slope regime attribution
  must remain consistent with all independent Σ₂ guards (coupling, curvature,
  variance drift, and aperture).** Independent OU and true long-memory Gaussian
  baselines pass (OK_OU and OK_LM). Any case that preserves κ₂-slope locally but
  violates a guard is refused and labeled **BOUNDARY**, enforcing “no false OK”
  under multi-constraint inconsistency.

- `s0025_cross_window_persistence_guard.py`
  Cross-window persistence gate (Σ₂ continuation): certifies that κ₂-slope regime attribution (OK_OU/OK_LM) is admissible
  only when scaling persists across **adjacent late-time continuation windows** and the declared Σ₂ guard bundle remains
  quiet in **both** windows. Independent OU and true long-memory Gaussian baselines persist and pass. Constructed cases
  that violate coupling, curvature, or aperture constraints are refused and labeled **BOUNDARY**, enforcing “no false OK”
  under continuation.

- `s0026_real_stream_harness_guard.py`
  Real-stream admissibility harness (C5 deployment gate): certifies that a measurement-like ω̂(t)
  stream can be evaluated by the same κ₂-slope + Σ₂ refusal ladder without producing false `OK_*`
  outcomes. Synthetic OU and true long-memory Gaussian baselines still certify `OK_OU` and `OK_LM`,
  while irregular sampling / gaps / dt mismatch trigger a forced **BOUNDARY** (never `OK_*`),
  preserving the “no false OK under C5” contract.

- `s0027_finite_N_robustness_gate.py`
  Finite-N / finite-horizon robustness gate (no-scan): evaluates the full κ₂+guards
  ladder under a fixed config menu (canonical, finite-N stress, finite-horizon stress)
  with a fixed replicate bundle. PASS requires OU_BASE stays **OK_OU**, LM_TRUE stays
  **OK_LM**, and all constructed violations remain **BOUNDARY** across all configs.

## Run examples

From `tools/phase-evolution/`:

    python toys/s0001_k2_scaling_demo.py
    python toys/s0005_c1_c2_c5_windowing_demo.py
    python toys/s0005_c1_c2_c5_windowing_demo.py --write_outputs
    python toys/s0006_radiation_omega_generators.py
    python toys/s0006_radiation_omega_generators.py --write_outputs
    python toys/s0007_mixture_crossover_audit.py
    python toys/s0008_drift_vs_diffusion_demo.py
    python toys/s0008_drift_vs_diffusion_demo.py --write_outputs
    python toys/s0009_k5_higher_cumulant_boundary_demo.py
    python toys/s0009_k5_higher_cumulant_boundary_demo.py --write_outputs
    python toys/s0010_c3_vs_c4_k2_transport_interface_demo.py
    python toys/s0010_c3_vs_c4_k2_transport_interface_demo.py --write_outputs
    python toys/s0011_c3_c4_kurtosis_separation_demo.py
    python toys/s0011_c3_c4_kurtosis_separation_demo.py --write_outputs
    python toys/s0012_k6_even_cumulant_boundary_demo.py
    python toys/s0012_k6_even_cumulant_boundary_demo.py --write_outputs
    python toys/s0013_k6_magnitude_sufficiency_boundary_demo.py
    python toys/s0013_k6_magnitude_sufficiency_boundary_demo.py --write_outputs
    python toys/s0014_c5_kernels_hide_higher_cumulants_demo.py
    python toys/s0014_c5_kernels_hide_higher_cumulants_demo.py --write_outputs
    python toys/s0015_c2_vs_c4_cross_trajectory_confusion_demo.py
    python toys/s0015_c2_vs_c4_cross_trajectory_confusion_demo.py --write_outputs
    python toys/s0016_temporal_curvature_boundary_demo.py
    python toys/s0016_temporal_curvature_boundary_demo.py --write_outputs
    python toys/s0017_cross_window_regime_consistency_demo.py
    python toys/s0017_cross_window_regime_consistency_demo.py --write_outputs
    python toys/s0018_non_gaussian_masquerade_boundary_demo.py
    python toys/s0018_non_gaussian_masquerade_boundary_demo.py --write_outputs
    python toys/s0019_variance_drift_masquerade_boundary_demo.py
    python toys/s0019_variance_drift_masquerade_boundary_demo.py --write_outputs
    python toys/s0020_winding_sector_admissibility_demo.py
    python toys/s0020_winding_sector_admissibility_demo.py --write_outputs
    python toys/s0021_k2_vs_aperture_admissibility_demo.py
    python toys/s0021_k2_vs_aperture_admissibility_demo.py --write_outputs
    python toys/s0022_estimator_manufactured_scaling_guard.py
    python toys/s0022_estimator_manufactured_scaling_guard.py --write_outputs    
    python toys/s0023_transport_masquerade_guard.py
    python toys/s0023_transport_masquerade_guard.py --write_outputs 
    python toys/s0024_cross_observable_consistency_guard.py
    python toys/s0024_cross_observable_consistency_guard.py --write_outputs 
    python toys/s0025_cross_window_persistence_guard.py
    python toys/s0025_cross_window_persistence_guard.py --write_outputs 
    python toys/s0026_real_stream_harness_guard.py
    python toys/s0026_real_stream_harness_guard.py --write_outputs 
    python toys/s0027_finite_N_robustness_gate.py
    python toys/s0027_finite_N_robustness_gate.py --write_outputs
