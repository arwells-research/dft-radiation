#!/usr/bin/env python3
"""
S-0006 — Radiation ω Generator Families: audited κ2 scaling classification across binding primitives

Goal (auditable, non-ad-hoc):
  Provide a finite, declared menu of ω(t) generator primitives representing
  phenomenological “bindings” of C1/C2/C3/C4 constraints, and certify that
  the κ2(t)=Var(Δϕ(t)) scaling exponent α is:
    (a) measurable in a declared late-time window (high r² + κ2 adequacy), and
    (b) consistent with pre-declared expectations for each generator family.

This is NOT a new stochastic ontology. It is a diagnostic harness:
  ω is treated as a stationary projection-variable for this toy program.

Constraint tie-in (binding intuition):
  - C1 (temporal): base ω dynamics / spectrum class (OU-like short memory).
  - C2 (statistical): long-memory baseline (fGn) as a controlled contrast.
  - C3 (transport): linear dispersive-like filtering of ω (does not change memory class,
        but can change inferred bandwidth/smoothness).
  - C4 (interface): event-driven injections (shot-like) producing non-Gaussian increments
        while keeping long-time κ2 scaling in the same class.
  - C5 (epistemic) is deliberately excluded here; certified separately in S-0005.

Core discriminant (audit-grade):
  For an ensemble of ω trajectories:
    Δϕ(t)=∫ω(s)ds ; κ₂(t)=Var(Δϕ(t))
  Fit α in log κ₂ ~ const + α log t over a declared late-time window.

Design rules:
  - deterministic seeds
  - no post-hoc tuning: generator menu + expectations declared in code
  - explicit admissibility checks (r², κ₂ adequacy, sufficient points)
  - explicit failure interpretation (no certified scaling window / boundary)
  - outputs optionally written to toys/outputs/ (untracked per .gitignore)
  
Centering convention: We remove a single global DC estimate from ω over the 
full ensemble×time grid to prevent trivial phase drift. We do not subtract a 
per-trajectory time mean (which imposes a bridge constraint and can invalidate 
late-window κ₂ scaling fits).

Run:
  python toys/s0006_radiation_omega_generators.py
  python toys/s0006_radiation_omega_generators.py --write_outputs

Exit code:
  0 on AUDIT PASS
  2 on AUDIT FAIL  (any admissible generator violates its declared α expectation)
  3 on INCONCLUSIVE (one or more generators are not admissible in the declared window)

Notes:
  - Intentionally self-contained (no shared framework module).
  - If you later refactor common helpers, replace local functions with imports
    without changing semantics.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


# -----------------------------
# Config
# -----------------------------

@dataclass(frozen=True)
class S0006Config:
    # deterministic
    seed: int = 24680

    # grid (long run, late-time window fit)
    dt: float = 0.1
    n_steps: int = 2048

    # ensemble
    n_trajectories: int = 512

    # OU (short-memory)
    ou_theta: float = 0.12
    ou_sigma: float = 1.00

    # fGn (long-memory baseline)
    fgn_H: float = 0.70
    fgn_sigma: float = 1.00

    # C3-style transport filter (applied to ω per trajectory)
    # Simple symmetric FIR smoothing widths (samples); 1 disables.
    transport_m_list: Tuple[int, ...] = (1, 8, 32)

    # C4-style interface: event-driven injection into ω (shot in ω)
    # Poisson events with exponential amplitude distribution (mean amp),
    # and event kernel as causal exp decay in ω with time constant tau_shot (samples).
    shot_rate_per_time: float = 0.20  # events per unit time (per trajectory)
    shot_amp_mean: float = 2.0
    shot_tau_samples: int = 12
    shot_sign_symmetric: bool = True  # if True, random sign per event

    # C4 shot params for fGn variants (kept weak so fGn scaling remains dominant in the audit window)
    shot_rate_per_time_fgn: float = 0.03
    shot_amp_mean_fgn: float = 0.50

    # audit window in time units (late-time scaling window)
    audit_window_t_min: float = 70.0
    audit_window_t_max: float = 170.0

    # admissibility
    min_r2: float = 0.985
    k2_min_at_tmax: float = 0.50

    # expectation tolerances (predeclared; no tuning)
    # OU expected alpha ~ 1
    ou_alpha_target: float = 1.00
    ou_alpha_tol: float = 0.20

    # fGn expected alpha ~ 2H
    fgn_alpha_tol: float = 0.20

    # outputs (untracked)
    out_dir: str = "toys/outputs"
    out_audit_csv: str = "s0006_audit.csv"
    out_cases_csv: str = "s0006_cases.csv"


# -----------------------------
# Provenance helpers (CSV comment header convention)
# -----------------------------

def _run_utc_iso() -> str:
    dt = datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0)
    return dt.isoformat().replace("+00:00", "Z")


def _git_commit_short() -> str:
    try:
        p = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            text=True,
        )
        if p.returncode != 0:
            return "nogit"
        s = (p.stdout or "").strip()
        return s if s else "nogit"
    except Exception:
        return "nogit"


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _write_csv_with_provenance_header(path: Path, header_kv: Dict[str, str], rows: List[List[object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        for k, v in header_kv.items():
            f.write(f"# {k}={v}\n")
        w = csv.writer(f)
        for r in rows:
            w.writerow(r)


# -----------------------------
# Generators
# -----------------------------

def simulate_ou(cfg: S0006Config, rng: np.random.Generator) -> np.ndarray:
    """
    Euler–Maruyama OU per trajectory:
        dω = -θ ω dt + σ dW
    """
    nT, n = int(cfg.n_trajectories), int(cfg.n_steps)
    dt = float(cfg.dt)
    theta = float(cfg.ou_theta)
    sigma = float(cfg.ou_sigma)

    omega = np.empty((nT, n), dtype=float)
    sdt = np.sqrt(dt)
    for i in range(nT):
        x = 0.0
        for k in range(n):
            x = x + (-theta * x) * dt + sigma * sdt * float(rng.standard_normal())
            omega[i, k] = x
    return omega


def fgn_davies_harte(n_steps: int, H: float, rng: np.random.Generator) -> np.ndarray:
    """
    Davies–Harte fGn innovations (Gaussian), unit-std approx, length n_steps.
    """
    if not (0.0 < H < 1.0):
        raise ValueError("H must be in (0,1)")
    n = int(n_steps)

    k = np.arange(0, n)
    gamma = 0.5 * (np.abs(k + 1) ** (2 * H) + np.abs(k - 1) ** (2 * H) - 2 * np.abs(k) ** (2 * H))
    r = np.concatenate([gamma, gamma[-2:0:-1]])

    lam = np.fft.rfft(r).real
    lam[lam < 0] = 0.0

    m = lam.shape[0]
    z = rng.normal(size=m) + 1j * rng.normal(size=m)
    x = np.fft.irfft(np.sqrt(lam) * z, n=r.shape[0])[:n]

    std = float(np.std(x))
    if std <= 0:
        raise RuntimeError("fGn generation produced non-positive std.")
    return x / std


def simulate_fgn(cfg: S0006Config, rng: np.random.Generator) -> np.ndarray:
    nT, n = int(cfg.n_trajectories), int(cfg.n_steps)
    H = float(cfg.fgn_H)
    sigma = float(cfg.fgn_sigma)

    omega = np.empty((nT, n), dtype=float)
    for i in range(nT):
        x = fgn_davies_harte(n, H, rng)
        omega[i, :] = sigma * x
    return omega


def center_global_dc(omega):
    # subtract ONE constant across everything
    return omega - float(np.mean(omega))

def fir_smooth_per_traj(x: np.ndarray, m: int) -> np.ndarray:
    """
    Symmetric moving-average FIR smoothing (reflect padding), per trajectory.
    C3-style transport / dispersion surrogate (linear, time-invariant).
    """
    m = int(m)
    if m <= 1:
        return x.copy()

    k = np.ones(m, dtype=float) / float(m)
    pad = m // 2

    nT, n = x.shape
    out = np.empty_like(x)
    for i in range(nT):
        xp = np.pad(x[i, :], (pad, pad), mode="reflect")
        y = np.convolve(xp, k, mode="valid")
        out[i, :] = y[:n]
    return out


def add_shot_in_omega(
    omega: np.ndarray,
    cfg: S0006Config,
    rng: np.random.Generator,
    *,
    rate_per_time=None, 
    amp_mean=None
) -> np.ndarray:
    """
    C4-style interface surrogate:
      Add event-driven injections directly into ω(t).
      Events are Poisson in continuous time with rate λ per unit time.

    Each event adds a causal exponential kernel to ω:
      h[k] = A * exp(-k / tau_shot)   for k>=0
    with A drawn from Exp(mean=shot_amp_mean), optional random sign.

    This produces non-Gaussian increments while preserving long-time diffusive scaling
    in typical parameter ranges (diagnostic, not ontological).
    """
    y = omega.copy()
    nT, n = y.shape

    lam = float(cfg.shot_rate_per_time if rate_per_time is None else rate_per_time)
    amp_mean_eff = float(cfg.shot_amp_mean if amp_mean is None else amp_mean)
    dt = float(cfg.dt)
    tau = int(cfg.shot_tau_samples)
    if tau <= 0 or lam <= 0:
        return y

    # expected events per trajectory
    mean_events = lam * (n * dt)
    # precompute kernel tail length (truncate when exp(-L/tau) < 1e-6)
    L = int(max(1, np.ceil(tau * 14.0)))
    kernel = np.exp(-np.arange(L, dtype=float) / float(tau))

    for i in range(nT):
        n_events = int(rng.poisson(mean_events))
        if n_events <= 0:
            continue

        # event times uniform over [0, n-1] (discrete samples)
        t_idx = rng.integers(low=0, high=n, size=n_events)

        amps = rng.exponential(scale=amp_mean_eff, size=n_events)
        if bool(cfg.shot_sign_symmetric):
            signs = rng.choice([-1.0, 1.0], size=n_events)
            amps = amps * signs

        for j in range(n_events):
            t0 = int(t_idx[j])
            A = float(amps[j])
            # apply kernel starting at t0, truncated at n
            t1 = min(n, t0 + L)
            y[i, t0:t1] += A * kernel[: (t1 - t0)]

    return y


# -----------------------------
# κ2(t) and scaling fit
# -----------------------------

def delta_phi_from_omega(omega0: np.ndarray, dt: float) -> np.ndarray:
    return np.cumsum(omega0, axis=1) * float(dt)


def k2_direct(delta_phi: np.ndarray) -> np.ndarray:
    return np.var(delta_phi, axis=0, ddof=0)


def _mask_window(t: np.ndarray, cfg: S0006Config) -> np.ndarray:
    return (t >= float(cfg.audit_window_t_min)) & (t <= float(cfg.audit_window_t_max))


def fit_alpha_loglog(t: np.ndarray, k2: np.ndarray, cfg: S0006Config) -> Dict[str, float]:
    """
    Fit log k2 = b + alpha log t over the audit window.
    Returns alpha, r2, n_points, k2_at_tmax (adequacy), plus pass flag.
    """
    t = np.asarray(t, dtype=float)
    k2 = np.asarray(k2, dtype=float)

    eps = 1e-18
    mask = _mask_window(t, cfg) & np.isfinite(k2) & (k2 > eps)

    npts = int(np.count_nonzero(mask))
    if npts < 12:
        return {"alpha": float("nan"), "r2": float("nan"), "n_points": npts, "k2_tmax": float("nan"), "pass": False}

    x = np.log(t[mask])
    y = np.log(k2[mask])

    A = np.vstack([np.ones_like(x), x]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    yhat = A @ coef

    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else float("nan")
    alpha = float(coef[1])

    idx_tmax = int(np.where(mask)[0][-1])
    k2_tmax = float(k2[idx_tmax])

    passed = bool(
        np.isfinite(alpha)
        and np.isfinite(r2)
        and (r2 >= float(cfg.min_r2))
        and (k2_tmax >= float(cfg.k2_min_at_tmax))
    )
    return {"alpha": alpha, "r2": float(r2), "n_points": npts, "k2_tmax": k2_tmax, "pass": passed}


# -----------------------------
# Expectations + audit logic
# -----------------------------

def expected_alpha_ranges(cfg: S0006Config) -> Dict[str, Tuple[float, float]]:
    """
    Declared expectations, expressed as alpha in [lo, hi].
    Keep these broad enough to avoid accidental false negatives, but tight enough
    to be meaningful and non-ad-hoc.
    """
    ou_lo = float(cfg.ou_alpha_target - cfg.ou_alpha_tol)
    ou_hi = float(cfg.ou_alpha_target + cfg.ou_alpha_tol)

    fgn_target = float(2.0 * cfg.fgn_H)
    fgn_lo = float(fgn_target - cfg.fgn_alpha_tol)
    fgn_hi = float(fgn_target + cfg.fgn_alpha_tol)

    # Transport-filtered variants should not change the *scaling class* in α,
    # only smooth high-frequency content; keep same α expectations.
    return {
        "OU": (ou_lo, ou_hi),
        "OU_C3_FIR": (ou_lo, ou_hi),
        "OU_C4_SHOT": (ou_lo, ou_hi),
        "OU_C3_FIR_C4_SHOT": (ou_lo, ou_hi),

        "fGn": (fgn_lo, fgn_hi),
        "fGn_C3_FIR": (fgn_lo, fgn_hi),
        "fGn_C4_SHOT": (fgn_lo, fgn_hi),
        "fGn_C3_FIR_C4_SHOT": (fgn_lo, fgn_hi),
    }


def check_expectation(name: str, alpha: float, cfg: S0006Config, ranges: Dict[str, Tuple[float, float]]) -> bool:
    if name not in ranges:
        return False
    lo, hi = ranges[name]
    return bool(np.isfinite(alpha) and (alpha >= lo) and (alpha <= hi))


def main() -> int:
    p = argparse.ArgumentParser(description="S-0006: ω generator families → audited κ2 scaling classification.")
    p.add_argument("--seed", type=int, default=S0006Config.seed)
    p.add_argument("--write_outputs", action="store_true", help="Write untracked CSV outputs to toys/outputs/.")
    # keep overrides minimal; this toy’s point is stable, declared expectations
    p.add_argument("--min_r2", type=float, default=S0006Config.min_r2)
    p.add_argument("--k2_min_at_tmax", type=float, default=S0006Config.k2_min_at_tmax)
    args = p.parse_args()

    cfg = S0006Config(
        seed=int(args.seed),
        min_r2=float(args.min_r2),
        k2_min_at_tmax=float(args.k2_min_at_tmax),
    )

    rng = np.random.default_rng(cfg.seed)
    t = (np.arange(int(cfg.n_steps)) + 1) * float(cfg.dt)

    ranges = expected_alpha_ranges(cfg)

    # --- Build generator menu (finite, declared) ---
    # Base ω for OU and fGn (truth-side)
    omega_ou = center_global_dc(simulate_ou(cfg, rng=np.random.default_rng(cfg.seed + 1)))
    omega_fgn = center_global_dc(simulate_fgn(cfg, rng=np.random.default_rng(cfg.seed + 2)))

    # Derived variants (C3 FIR transport, C4 shot interface)
    # Choose one representative FIR width for “C3 present” (use mid list if available).
    m_c3 = int(cfg.transport_m_list[min(1, len(cfg.transport_m_list) - 1)])  # e.g., 8 if present

    omega_ou_c3 = center_global_dc(fir_smooth_per_traj(omega_ou, m=m_c3))
    omega_fgn_c3 = center_global_dc(fir_smooth_per_traj(omega_fgn, m=m_c3))

    omega_ou_c4 = center_global_dc(add_shot_in_omega(omega_ou,  cfg, rng=np.random.default_rng(cfg.seed + 100)))
    omega_fgn_c4 = center_global_dc(add_shot_in_omega(
        omega_fgn, cfg,
        rng=np.random.default_rng(cfg.seed + 101),
        rate_per_time=cfg.shot_rate_per_time_fgn,
        amp_mean=cfg.shot_amp_mean_fgn,
    ))
    # Combine C3 + C4 (order: interface injection then transport smoothing)
    omega_ou_c3c4 = center_global_dc(fir_smooth_per_traj(omega_ou_c4, m=m_c3))
    omega_fgn_c3c4 = center_global_dc(fir_smooth_per_traj(omega_fgn_c4, m=m_c3))

    menu: List[Tuple[str, np.ndarray]] = [
        ("OU", omega_ou),
        ("OU_C3_FIR", omega_ou_c3),
        ("OU_C4_SHOT", omega_ou_c4),
        ("fGn", omega_fgn),
        ("fGn_C3_FIR", omega_fgn_c3),
        ("fGn_C4_SHOT", omega_fgn_c4),
        # (optional) combined binding: keep name aligned with expectations by reusing the base range
        ("OU_C3_FIR_C4_SHOT", omega_ou_c3c4),
        ("fGn_C3_FIR_C4_SHOT", omega_fgn_c3c4),
    ]

    # --- Evaluate all ---
    rows: List[List[object]] = []
    any_inconclusive = False
    any_fail = False

    # Report window and expectation targets once (human-auditable)
    fgn_target = 2.0 * float(cfg.fgn_H)
    print(
        f"[S-0006] window=[{cfg.audit_window_t_min:.1f},{cfg.audit_window_t_max:.1f}] "
        f"min_r2={cfg.min_r2:.3f} k2_min@tmax={cfg.k2_min_at_tmax:.3g} "
        f"targets: OU~{cfg.ou_alpha_target:.2f}±{cfg.ou_alpha_tol:.2f}, fGn~{fgn_target:.2f}±{cfg.fgn_alpha_tol:.2f}"
    )

    for gen_name, omega in menu:
        dphi = delta_phi_from_omega(omega, cfg.dt)
        k2 = k2_direct(dphi)
        fit = fit_alpha_loglog(t, k2, cfg)

        status = "ADMISSIBLE" if bool(fit["pass"]) else "INADMISSIBLE"
        if status == "INADMISSIBLE":
            any_inconclusive = True

        exp_ok = False
        if status == "ADMISSIBLE":
            exp_ok = check_expectation(gen_name, float(fit["alpha"]), cfg, ranges)
            if not exp_ok:
                any_fail = True

        tag = "OK" if (status == "ADMISSIBLE" and exp_ok) else ("BOUNDARY" if status == "INADMISSIBLE" else "FAIL")

        lo, hi = ranges.get(gen_name, (float("nan"), float("nan")))
        print(
            f"[S-0006:{gen_name}] alpha={fit['alpha']:.3f} r2={fit['r2']:.3f} k2_tmax={fit['k2_tmax']:.3g} "
            f"exp=[{lo:.3f},{hi:.3f}] -> {tag}"
        )

        rows.append(
            [
                gen_name,
                float(fit["alpha"]),
                float(fit["r2"]),
                float(fit["k2_tmax"]),
                int(bool(fit["pass"])),
                float(lo),
                float(hi),
                tag,
            ]
        )

    # --- Gate logic ---
    # - If any generator is inadmissible, we mark INCONCLUSIVE (boundary / no certified scaling window).
    # - If all admissible but any violates expectation, AUDIT FAIL.
    # - If all admissible and all meet expectation, AUDIT PASS.
    if any_inconclusive:
        print("[S-0006] INCONCLUSIVE: one or more generator families are not admissible in the declared window.")
        exit_code = 3
    elif any_fail:
        print("[S-0006] AUDIT FAILED: one or more admissible generator families violate declared α expectations.")
        exit_code = 2
    else:
        print("[S-0006] AUDIT PASSED: all generator families admissible and consistent with declared α expectations.")
        exit_code = 0

    # --- Optional outputs ---
    if args.write_outputs:
        out_dir = Path(cfg.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        header_kv = {
            "git_commit": _git_commit_short(),
            "run_utc": _run_utc_iso(),
            "toy_file": Path(__file__).name,
            "source_sha256": _sha256_of_file(Path(__file__)),
        }

        cases_path = out_dir / cfg.out_cases_csv
        out_rows: List[List[object]] = [
            ["generator", "alpha", "r2", "k2_tmax", "admissible", "exp_lo", "exp_hi", "tag"]
        ]
        out_rows.extend(rows)
        _write_csv_with_provenance_header(cases_path, header_kv, out_rows)
        print(f"Wrote (untracked): {cases_path}")

        audit_path = out_dir / cfg.out_audit_csv
        audit_rows = [
            ["field", "value"],
            ["exit_code", exit_code],
            ["seed", int(cfg.seed)],
            ["dt", float(cfg.dt)],
            ["n_steps", int(cfg.n_steps)],
            ["n_trajectories", int(cfg.n_trajectories)],
            ["audit_window_t_min", float(cfg.audit_window_t_min)],
            ["audit_window_t_max", float(cfg.audit_window_t_max)],
            ["min_r2", float(cfg.min_r2)],
            ["k2_min_at_tmax", float(cfg.k2_min_at_tmax)],
            ["ou_theta", float(cfg.ou_theta)],
            ["ou_sigma", float(cfg.ou_sigma)],
            ["fgn_H", float(cfg.fgn_H)],
            ["fgn_sigma", float(cfg.fgn_sigma)],
            ["transport_m_used", int(m_c3)],
            ["shot_rate_per_time", float(cfg.shot_rate_per_time)],
            ["shot_amp_mean", float(cfg.shot_amp_mean)],
            ["shot_tau_samples", int(cfg.shot_tau_samples)],
            ["shot_sign_symmetric", int(bool(cfg.shot_sign_symmetric))],
        ]
        _write_csv_with_provenance_header(audit_path, header_kv, audit_rows)
        print(f"Wrote (untracked): {audit_path}")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())