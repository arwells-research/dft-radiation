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

## Run examples

From `tools/phase-evolution/`:

    python toys/s0001_k2_scaling_demo.py
    python toys/s0005_c1_c2_c5_windowing_demo.py
    python toys/s0005_c1_c2_c5_windowing_demo.py --write_outputs
    python toys/s0006_radiation_omega_generators.py
    python toys/s0006_radiation_omega_generators.py --write_outputs
