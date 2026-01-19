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

