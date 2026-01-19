#!/usr/bin/env python3
"""
S-0013 — κ₆ magnitude sufficiency boundary (even-cumulant beyond κ₄)

Goal:
  Certify that a controlled, non-negligible κ₆ contribution at the innovation level
  can produce a systematic magnitude-envelope deviation that κ₂+κ₄ truncation cannot
  explain (i.e., a categorical even-cumulant sufficiency boundary beyond κ₄).

Key idea:
  For i.i.d. ω innovations, Δϕ(t) = dt * sum_{k<=t/dt} ω_k.
  Then κ_n[Δϕ(t)] scales as (dt^n) * N(t) * κ_n[ω].
  The relative importance of κ₆ vs κ₂ grows as dt^4, so we use a larger dt while
  keeping κ₂ moderate by auditing over a short fixed time window.

Cases:
  - GAUSS: ω ~ N(0,1) (baseline)
  - K6_MILD: discrete symmetric 3-point with mean=0, Var=1, μ4=3 (=> κ4=0), κ6=-6 (small effect)
  - K6_HEAVY: discrete symmetric 5-point with mean=0, Var=1, μ4=3 (=> κ4=0), κ6>>0 (large effect)

Prediction model (even-cumulant magnitude truncation):
    log |Z(t)| ≈ -κ2(t)/2 + κ4(t)/24
  where Z(t) := ⟨exp(iΔϕ(t))⟩ and κn(t) are empirical cumulants of Δϕ(t) across
  the ensemble at fixed t.

Certification target:
  - GAUSS must PASS magnitude closure in the declared window.
  - K6_HEAVY must FAIL magnitude closure in the declared window (systematic deviation).
  - K6_MILD is expected to be close-to-baseline (may PASS); it is included as a control.

Exit codes:
  0 — PASS: GAUSS passes AND K6_HEAVY fails (with κ6 sufficiently large in-window).
  2 — FAIL: Any required audit condition fails.
  3 — INCONCLUSIVE: coherence floor too low in the audit window (no magnitude claim allowed).

Conventions:
  - Deterministic seeds only.
  - Fixed audit window; no scanning / optimization.
  - Outputs written only under toys/outputs/ and only when --write_outputs is passed.
  - CSV outputs include provenance header lines (# key=value).
"""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


@dataclass(frozen=True)
class S0013Config:
    seed: int = 13013

    # Choose dt large enough that κ6 term is not negligible relative to κ2,
    # while keeping κ2 moderate by auditing early times.
    dt: float = 0.20
    n_steps: int = 512
    n_trajectories: int = 2048

    # Fixed audit window (absolute times)
    t_min: float = 4.0
    t_max: float = 7.0

    # Magnitude-closure audit tolerances in the window (operate on log c)
    max_median_abs_dlogc: float = 0.02
    max_p95_abs_dlogc: float = 0.06
    max_abs_drift_slope: float = 0.03  # slope of dlogc vs t within window

    # For a "must-fail" case, require deviation exceeds this lower bound (median)
    min_median_abs_dlogc_fail: float = 0.02

    # Coherence floor guard: if |Z| too small in window, log comparisons are unstable
    coherence_floor: float = 5.0e-3

    # Window fit admissibility for reporting (not a hard gate for pass/fail here)
    min_points: int = 12

    # Heavy-tail parameter for K6_HEAVY (value b). Larger -> larger κ6.
    # Keep b^2 >= 3 to keep probabilities valid.
    k6_heavy_b: float = 25.0

    out_dir: str = "toys/outputs"
    out_compare_csv: str = "s0013_envelope_compare.csv"
    out_error_png: str = "s0013_error_diagnostic.png"
    out_audit_csv: str = "s0013_audit.csv"


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


def _audit_mask(t: np.ndarray, t_min: float, t_max: float) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    return np.isfinite(t) & (t >= float(t_min)) & (t <= float(t_max))


def _linear_slope(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    good = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(good) < 4:
        return float("nan")
    xv = x[good]
    yv = y[good]
    A = np.vstack([np.ones_like(xv), xv]).T
    coef, *_ = np.linalg.lstsq(A, yv, rcond=None)
    return float(coef[1])


def _discrete_innovations(
    *,
    n: Tuple[int, int],
    support: np.ndarray,
    probs: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    support = np.asarray(support, dtype=float)
    probs = np.asarray(probs, dtype=float)
    if support.ndim != 1 or probs.ndim != 1 or support.size != probs.size:
        raise ValueError("support/probs mismatch")
    if not np.isclose(np.sum(probs), 1.0):
        raise ValueError("probs must sum to 1")
    if np.any(probs < 0):
        raise ValueError("probs must be nonnegative")

    idx = rng.choice(np.arange(support.size), size=n, p=probs)
    return support[idx].astype(float)


def simulate_omega_gauss(cfg: S0013Config, rng: np.random.Generator) -> np.ndarray:
    return rng.standard_normal(size=(int(cfg.n_trajectories), int(cfg.n_steps))).astype(float)


def simulate_omega_k6_mild(cfg: S0013Config, rng: np.random.Generator) -> np.ndarray:
    # 3-point symmetric: {-a, 0, +a}, probs {p/2, 1-p, p/2}
    # Enforce Var=1 and μ4=3 -> p=1/3, a^2=3.
    # Then κ4=0 and κ6 = μ6 - 15 = 9 - 15 = -6.
    p = 1.0 / 3.0
    a = np.sqrt(3.0)
    support = np.array([-a, 0.0, +a], dtype=float)
    probs = np.array([0.5 * p, 1.0 - p, 0.5 * p], dtype=float)
    return _discrete_innovations(n=(int(cfg.n_trajectories), int(cfg.n_steps)), support=support, probs=probs, rng=rng)


def simulate_omega_k6_heavy(cfg: S0013Config, rng: np.random.Generator) -> np.ndarray:
    """
    Heavy even-cumulant case designed to be finite-sample effective.

    Innovations are symmetric with support {±a, ±b}:
        P(±b) = q each
        P(±a) = p each, where p = 0.5 - q
    Enforce mean=0 and Var=1 exactly by solving for a given (b, q).

    q is chosen so that, by a reference step i* in the middle of the audit window,
    we expect ~N_tail_target trajectories to have seen at least one tail event.
    This prevents the "q so small we never sample tails" failure mode.

    We do NOT force κ4=0 here; the predictor already includes κ4. S-0013 is about
    κ6 breaking κ2+κ4 magnitude sufficiency.
    """
    b = float(cfg.k6_heavy_b)

    # Reference time t* (mid-window) and corresponding step count
    t_star = 0.5 * (float(cfg.t_min) + float(cfg.t_max))
    i_star = int(max(1, round(t_star / float(cfg.dt))))

    # Target: expected number of trajectories with >=1 tail event by i_star.
    # Tune this if needed, but keep it fixed (no scanning).
    N_tail_target = 80.0

    n_tr = int(cfg.n_trajectories)

    # For small q: P(traj has tail by i*) ≈ 2 q i*
    # Expected count ≈ n_tr * 2 q i*  => q ≈ N_target / (2 i* n_tr)
    q = float(N_tail_target) / (2.0 * float(i_star) * float(n_tr))

    # Clamp to sane range (still deterministic; no tuning)
    q = float(np.clip(q, 1e-6, 0.10))

    # Enforce feasibility under Var=1: need 1 - 2 q b^2 > 0
    # If violated, reduce q (do not reduce b silently).
    q_max_feasible = 0.49 / (b * b)  # conservative margin < 0.5/b^2
    if q >= q_max_feasible:
        q = 0.9 * q_max_feasible

    p = 0.5 - q
    if p <= 0.0:
        raise ValueError("q too large; need p>0")

    rhs = 1.0 - 2.0 * q * (b * b)
    if rhs <= 0.0:
        raise ValueError("Infeasible (Var=1): reduce k6_heavy_b.")

    a2 = rhs / (2.0 * p)
    if a2 <= 0.0:
        raise ValueError("Moment solution invalid (a^2 <= 0).")
    a = float(np.sqrt(a2))

    support = np.array([+a, -a, +b, -b], dtype=float)
    probs = np.array([p, p, q, q], dtype=float)

    return _discrete_innovations(
        n=(int(cfg.n_trajectories), int(cfg.n_steps)),
        support=support,
        probs=probs,
        rng=rng,
    )

def delta_phi_from_omega(omega: np.ndarray, dt: float) -> np.ndarray:
    return np.cumsum(np.asarray(omega, dtype=float), axis=1) * float(dt)


def complex_coherence(delta_phi: np.ndarray) -> np.ndarray:
    return np.mean(np.exp(1j * np.asarray(delta_phi, dtype=float)), axis=0)


def cumulants_2_4_6(delta_phi: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Empirical cumulants of Δϕ(t) across ensemble at each t (central-moment formulas).
    x = np.asarray(delta_phi, dtype=float)
    mu = np.mean(x, axis=0)
    xc = x - mu[None, :]

    m2 = np.mean(xc**2, axis=0)
    m3 = np.mean(xc**3, axis=0)
    m4 = np.mean(xc**4, axis=0)
    m6 = np.mean(xc**6, axis=0)

    k2 = m2
    k4 = m4 - 3.0 * (m2**2)
    # κ6 = μ6 - 15 μ4 μ2 - 10 μ3^2 + 30 μ2^3
    k6 = m6 - 15.0 * m4 * m2 - 10.0 * (m3**2) + 30.0 * (m2**3)
    return k2, k4, k6


def predict_magnitude_from_even_cumulants(k2: np.ndarray, k4: np.ndarray) -> np.ndarray:
    # |Z|_pred ≈ exp(-k2/2 + k4/24)
    logc = (-0.5 * k2) + (k4 / 24.0)
    return np.exp(logc)


def audit_magnitude_case(
    *,
    name: str,
    t: np.ndarray,
    z_meas: np.ndarray,
    c_pred: np.ndarray,
    k6: np.ndarray,
    cfg: S0013Config,
) -> Dict[str, float | str | int]:
    t = np.asarray(t, dtype=float)
    z_meas = np.asarray(z_meas, dtype=complex)
    c_pred = np.asarray(c_pred, dtype=float)

    mask = _audit_mask(t, cfg.t_min, cfg.t_max)
    c_meas = np.abs(z_meas)

    cmin = float(np.min(c_meas[mask])) if np.any(mask) else float("nan")
    incoherent = bool(np.any(mask) and (cmin < float(cfg.coherence_floor)))

    eps = 1e-18
    dlogc = np.log(np.maximum(c_meas, eps)) - np.log(np.maximum(c_pred, eps))
    dlogc_w = dlogc[mask] if np.any(mask) else np.asarray([], dtype=float)

    med_abs = float(np.median(np.abs(dlogc_w))) if dlogc_w.size else float("nan")
    p95_abs = float(np.quantile(np.abs(dlogc_w), 0.95)) if dlogc_w.size else float("nan")
    drift_slope = _linear_slope(t[mask], dlogc_w) if dlogc_w.size else float("nan")

    mag_ok = bool(
        (not incoherent)
        and np.isfinite(med_abs)
        and np.isfinite(p95_abs)
        and np.isfinite(drift_slope)
        and (med_abs <= float(cfg.max_median_abs_dlogc))
        and (p95_abs <= float(cfg.max_p95_abs_dlogc))
        and (abs(drift_slope) <= float(cfg.max_abs_drift_slope))
    )

    # Report κ6 at a reference t* near the center of the window (diagnostic only)
    if np.any(mask):
        idx = int(np.where(mask)[0][len(np.where(mask)[0]) // 2])
        k6_tstar = float(k6[idx])
    else:
        k6_tstar = float("nan")

    verdict: str
    if incoherent:
        verdict = "INCONCLUSIVE"
    elif mag_ok:
        verdict = "OK"
    else:
        verdict = "FAIL"

    return {
        "case": name,
        "coherence_min": cmin,
        "median_abs_dlogc": med_abs,
        "p95_abs_dlogc": p95_abs,
        "drift_slope": drift_slope,
        "k6_tstar": k6_tstar,
        "mag_ok": int(mag_ok),
        "verdict": verdict,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="S-0013: κ6 even-cumulant magnitude sufficiency boundary demo.")
    p.add_argument("--seed", type=int, default=S0013Config.seed)
    p.add_argument("--write_outputs", action="store_true")
    p.add_argument("--k6_heavy_b", type=float, default=S0013Config.k6_heavy_b)
    args = p.parse_args()

    cfg = S0013Config(seed=int(args.seed), k6_heavy_b=float(args.k6_heavy_b))

    rng_g = np.random.default_rng(cfg.seed + 1)
    rng_m = np.random.default_rng(cfg.seed + 2)
    rng_h = np.random.default_rng(cfg.seed + 3)

    t = (np.arange(int(cfg.n_steps)) + 1) * float(cfg.dt)
    mask = _audit_mask(t, cfg.t_min, cfg.t_max)
    n_w = int(np.count_nonzero(mask))

    print(
        f"[S-0013] window=[{cfg.t_min:.1f},{cfg.t_max:.1f}] n_w={n_w} dt={cfg.dt:.3g} "
        f"tol(med,p95,drift)=({cfg.max_median_abs_dlogc:.3g},{cfg.max_p95_abs_dlogc:.3g},{cfg.max_abs_drift_slope:.3g}) "
        f"fail_med_min={cfg.min_median_abs_dlogc_fail:.3g} coh_floor={cfg.coherence_floor:.3g} k6_heavy_b={cfg.k6_heavy_b:.3g}"
    )

    if n_w < int(cfg.min_points):
        print("[S-0013] INCONCLUSIVE: audit window has too few points.")
        return 3

    cases: List[Tuple[str, np.ndarray]] = [
        ("GAUSS", simulate_omega_gauss(cfg, rng=rng_g)),
        ("K6_MILD", simulate_omega_k6_mild(cfg, rng=rng_m)),
        ("K6_HEAVY", simulate_omega_k6_heavy(cfg, rng=rng_h)),
    ]

    audit_rows: List[Dict[str, float | str | int]] = []
    compare_table: List[List[object]] = []

    any_inconclusive = False
    results_by_case: Dict[str, Dict[str, float | str | int]] = {}

    for name, omega in cases:
        dphi = delta_phi_from_omega(omega, cfg.dt)
        z_meas = complex_coherence(dphi)
        k2, k4, k6 = cumulants_2_4_6(dphi)

        # Predictor truncation: even-cumulant magnitude model excludes κ6 by design.
        c_pred = predict_magnitude_from_even_cumulants(k2, k4)

        res = audit_magnitude_case(name=name, t=t, z_meas=z_meas, c_pred=c_pred, k6=k6, cfg=cfg)
        audit_rows.append(res)
        results_by_case[name] = res

        c_meas = np.abs(z_meas)
        eps = 1e-18
        dlogc = np.log(np.maximum(c_meas, eps)) - np.log(np.maximum(c_pred, eps))

        print(
            f"[S-0013:{name}] "
            f"coh_min={float(res['coherence_min']):.3g} "
            f"med|dlogc|={float(res['median_abs_dlogc']):.3g} p95={float(res['p95_abs_dlogc']):.3g} drift={float(res['drift_slope']):.3g} "
            f"k6(t*)={float(res['k6_tstar']):.3g} tag={res['verdict']}"
        )

        if str(res["verdict"]) == "INCONCLUSIVE":
            any_inconclusive = True

        if not compare_table:
            compare_table.append(["t", "case", "c_meas", "c_pred_even", "dlogc", "k2", "k4", "k6"])
        for i in range(int(cfg.n_steps)):
            compare_table.append(
                [
                    float(t[i]),
                    name,
                    float(c_meas[i]),
                    float(c_pred[i]),
                    float(dlogc[i]),
                    float(k2[i]),
                    float(k4[i]),
                    float(k6[i]),
                ]
            )

    # Certification logic:
    #   - GAUSS must be OK
    #   - K6_HEAVY must be FAIL with sufficiently large median deviation (to avoid "weak" fails)
    gauss_ok = (str(results_by_case["GAUSS"]["verdict"]) == "OK")
    heavy_fail = (str(results_by_case["K6_HEAVY"]["verdict"]) == "FAIL")
    heavy_med = float(results_by_case["K6_HEAVY"]["median_abs_dlogc"])
    heavy_strong = bool(np.isfinite(heavy_med) and (heavy_med >= float(cfg.min_median_abs_dlogc_fail)))

    if any_inconclusive:
        print("[S-0013] INCONCLUSIVE")
        exit_code = 3
    elif gauss_ok and heavy_fail and heavy_strong:
        print("[S-0013] AUDIT PASSED")
        exit_code = 0
    else:
        print(f"[S-0013] AUDIT FAILED: gauss_ok={int(gauss_ok)} heavy_fail={int(heavy_fail)} heavy_strong={int(heavy_strong)}")
        exit_code = 2

    if args.write_outputs:
        out_dir = Path(cfg.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        header_kv = {
            "git_commit": _git_commit_short(),
            "run_utc": _run_utc_iso(),
            "toy_file": Path(__file__).name,
            "source_sha256": _sha256_of_file(Path(__file__)),
        }

        compare_path = out_dir / cfg.out_compare_csv
        _write_csv_with_provenance_header(compare_path, header_kv, compare_table)
        print(f"Wrote (untracked): {compare_path}")

        audit_path = out_dir / cfg.out_audit_csv
        audit_table: List[List[object]] = [
            ["field", "value"],
            ["exit_code", exit_code],
            ["seed", int(cfg.seed)],
            ["dt", float(cfg.dt)],
            ["n_steps", int(cfg.n_steps)],
            ["n_trajectories", int(cfg.n_trajectories)],
            ["t_min", float(cfg.t_min)],
            ["t_max", float(cfg.t_max)],
            ["max_median_abs_dlogc", float(cfg.max_median_abs_dlogc)],
            ["max_p95_abs_dlogc", float(cfg.max_p95_abs_dlogc)],
            ["max_abs_drift_slope", float(cfg.max_abs_drift_slope)],
            ["min_median_abs_dlogc_fail", float(cfg.min_median_abs_dlogc_fail)],
            ["coherence_floor", float(cfg.coherence_floor)],
            ["k6_heavy_b", float(cfg.k6_heavy_b)],
            ["gauss_ok", int(gauss_ok)],
            ["heavy_fail", int(heavy_fail)],
            ["heavy_strong", int(heavy_strong)],
        ]
        for row in audit_rows:
            case = str(row["case"])
            for k in ["coherence_min", "median_abs_dlogc", "p95_abs_dlogc", "drift_slope", "k6_tstar", "mag_ok", "verdict"]:
                audit_table.append([f"{case}.{k}", row[k]])
        _write_csv_with_provenance_header(audit_path, header_kv, audit_table)
        print(f"Wrote (untracked): {audit_path}")

        import matplotlib.pyplot as plt  # noqa: PLC0415

        # Error diagnostic plot (dlogc vs t in window)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        # Reconstruct from compare_table (avoid re-storing large arrays separately)
        t_vals = np.asarray([r[0] for r in compare_table[1:]], dtype=float)
        cases_vals = np.asarray([r[1] for r in compare_table[1:]], dtype=object)
        dlogc_vals = np.asarray([r[4] for r in compare_table[1:]], dtype=float)

        wmask = (t_vals >= float(cfg.t_min)) & (t_vals <= float(cfg.t_max))
        for cname in ["GAUSS", "K6_MILD", "K6_HEAVY"]:
            m = wmask & (cases_vals == cname)
            ax.plot(t_vals[m], dlogc_vals[m], label=cname)

        ax.set_xlabel("t")
        ax.set_ylabel("log(c_meas) - log(c_pred_even)")
        ax.set_title("S-0013 magnitude-closure error (window-restricted)")
        ax.legend()

        err_path = out_dir / cfg.out_error_png
        fig.savefig(err_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote (untracked): {err_path}")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
