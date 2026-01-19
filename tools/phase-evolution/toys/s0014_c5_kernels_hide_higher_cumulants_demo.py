#!/usr/bin/env python3
"""
S-0014 — C5 measurement-kernel identifiability boundary:
         prevent false "κ2+κ4 magnitude closure" certification under higher-cumulant contamination.

Goal
----
Certify that C5 measurement kernels (windowing + LPF smoothing + additive noise) can
hide higher-cumulant (e.g. κ6) departures such that κ2+κ4 magnitude closure appears to pass
IF you compare predictions from "measured/inferred" ω against the true coherence envelope.

Therefore, the tool enforces a "no wrong claims" rule:
  - If the kernel is too strong (information-destroying), we must REFUSE certification
    (BOUNDARY / INCONCLUSIVE), not return OK, even if residuals are small.

Construction
------------
We generate ω_true innovations:
  - GAUSS: standard normal innovations
  - K6_CONTAM: symmetric heavy-tail innovations constructed to keep mean=0, Var=1, κ4≈0,
               while allowing large κ6 via rare large-amplitude events.

Physics (truth):
  Δϕ_true(t) = ∫ ω_true(s) ds  (discrete cumulative sum)
  Z_true(t)  = ⟨exp(i Δϕ_true(t))⟩
  c_true(t)  = |Z_true(t)|

Measurement/inference (C5 operator):
  ω_hat = K(ω_true) + noise
    K = (optional) windowing (taper) *and* per-trajectory FIR smoothing (moving average)
  Δϕ_hat(t) computed from ω_hat
  κ2_hat(t), κ4_hat(t) computed from Δϕ_hat ensemble cumulants
  c_pred(t) := exp( -κ2_hat/2 + κ4_hat/24 )  (even-cumulant magnitude truncation)

Audit compares c_true vs c_pred over a fixed window.
Strong kernels can make ω_hat "Gaussian-ish" and produce small residuals even when
ω_true was κ6-contaminated -> potential false OK.

Certification rule:
  - GAUSS + mild kernel must be OK.
  - K6 + mild kernel must FAIL (i.e., κ6 survives; κ2+κ4 cannot explain c_true).
  - K6 + strong kernel must NOT return OK:
      it must be BOUNDARY/INCONCLUSIVE due to kernel-strength / SNR gates.

Exit codes
----------
  0 PASS
  2 FAIL (any forbidden condition; e.g. false OK)
  3 INCONCLUSIVE (window too small or coherence floor too low across cases)

Outputs (only with --write_outputs)
----------------------------------
  toys/outputs/s0014_cases.csv
  toys/outputs/s0014_audit.csv
  toys/outputs/s0014_error_diagnostic.png  (optional)
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
class S0014Config:
    seed: int = 14014

    dt: float = 0.20
    n_steps: int = 512
    n_trajectories: int = 2048

    # Fixed audit window (absolute times)
    t_min: float = 4.0
    t_max: float = 7.0
    min_points: int = 12

    # Magnitude-closure audit tolerances on log(c) residuals
    max_median_abs_dlogc_ok: float = 0.02
    max_p95_abs_dlogc_ok: float = 0.06
    max_abs_drift_slope_ok: float = 0.03

    # For "must-fail" K6 under mild kernel: require median residual exceeds this
    min_median_abs_dlogc_fail: float = 0.02

    # Coherence floor guard (truth envelope)
    coherence_floor: float = 5.0e-3

    # C5 kernel regimes (moving-average length + SNR in dB)
    mild_ma_len: int = 1
    mild_snr_db: float = 80.0

    strong_ma_len: int = 25
    strong_snr_db: float = 10.0

    # Kernel identifiability gates:
    # If measurement kernel is stronger than these bounds, we refuse certification (BOUNDARY)
    max_cert_ma_len: int = 8
    min_cert_snr_db: float = 30.0

    # κ6 contamination strength parameter (rare tail amplitude)
    k6_b: float = 25.0

    # κ6 contamination frequency under the *mild* kernel (must-fail case)
    # Must satisfy q < 3/(2*b^4); for b=25 => q < ~3.84e-6.
    k6_q_mild: float =  1.0e-6

    k6_pulse_len: int = 12
    k6_pulse_prob: float = 2.0e-3

    out_dir: str = "toys/outputs"
    out_cases_csv: str = "s0014_cases.csv"
    out_audit_csv: str = "s0014_audit.csv"
    out_error_png: str = "s0014_error_diagnostic.png"


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


def simulate_omega_gauss(cfg: S0014Config, rng: np.random.Generator) -> np.ndarray:
    # innovations mean=0, Var=1
    return rng.standard_normal(size=(int(cfg.n_trajectories), int(cfg.n_steps))).astype(float)


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

def simulate_omega_k6_heavy(cfg: S0014Config, rng: np.random.Generator) -> np.ndarray:
    """
    "Must-fail" heavy case (used under mild kernel).

    Construct a high-κ6 Δϕ by injecting *block* pulses in ω:
      - rare rectangular pulses of amplitude ±b lasting L steps
      - symmetric signs -> mean ~ 0
      - then normalize to Var=1 (per entire array) to keep κ2 comparable.
    """
    b = float(cfg.k6_b)

    # Choose a deterministic block length and event probability.
    # These are the real knobs that control κ6(Δϕ) in the window.
    L = int(getattr(cfg, "k6_pulse_len", 12))          # e.g. 12 steps = 2.4 time units at dt=0.2
    q_evt = float(getattr(cfg, "k6_pulse_prob", 2e-3)) # per (traj, time step) start-probability

    nT = int(cfg.n_trajectories)
    nS = int(cfg.n_steps)

    # Start from near-Gaussian innovations
    omega = rng.standard_normal(size=(nT, nS)).astype(float)

    # Pulse start mask
    starts = rng.random(size=(nT, nS)) < q_evt
    # Random pulse sign per start
    signs = rng.choice(np.array([-1.0, +1.0]), size=(nT, nS), replace=True)

    # Add pulses (causal, rectangular)
    for k in range(nS):
        idx = np.where(starts[:, k])[0]
        if idx.size == 0:
            continue
        k2 = min(nS, k + L)
        omega[idx, k:k2] += signs[idx, k][:, None] * b

    # Normalize to mean 0, Var 1 (global) to keep κ2 comparable across cases
    omega = omega - float(np.mean(omega))
    v = float(np.var(omega))
    if v <= 0.0 or not np.isfinite(v):
        raise ValueError("simulate_omega_k6_heavy: variance invalid.")
    omega = omega / np.sqrt(v)

    return omega

def simulate_omega_k6_contam(cfg: S0014Config, rng: np.random.Generator) -> np.ndarray:
    """
    Heavy even-cumulant case: symmetric {0, ±a, ±b} innovations with
    mean=0, Var=1, μ4=3 (=> κ4=0) by construction, and large κ6 via rare ±b events.

    This mirrors the S-0013 "must fail" logic (but here used as the contamination source).
    """
    b = float(cfg.k6_b)
    q = 1.0e-12  # tiny tail probability per sign

    rhs2 = 0.5 * (1.0 - 2.0 * q * (b**2))
    rhs4 = 0.5 * (3.0 - 2.0 * q * (b**4))
    if rhs2 <= 0.0 or rhs4 <= 0.0:
        raise ValueError("k6_b too large for fixed q; violates moment feasibility.")

    p = (rhs2 * rhs2) / rhs4
    if p <= 0.0 or (2.0 * (p + q) >= 1.0):
        raise ValueError("Moment solution invalid (probabilities infeasible).")

    a2 = rhs2 / p
    if a2 <= 0.0:
        raise ValueError("Moment solution invalid (a^2 <= 0).")
    a = float(np.sqrt(a2))

    r = 1.0 - 2.0 * (p + q)
    if r < 0.0:
        raise ValueError("Moment solution invalid (r < 0).")

    support = np.array([0.0, +a, -a, +b, -b], dtype=float)
    probs = np.array([r, p, p, q, q], dtype=float)

    return _discrete_innovations(
        n=(int(cfg.n_trajectories), int(cfg.n_steps)),
        support=support,
        probs=probs,
        rng=rng,
    )


def delta_phi_from_omega(omega: np.ndarray, dt: float) -> np.ndarray:
    return np.cumsum(np.asarray(omega, dtype=float), axis=1) * float(dt)


def complex_coherence(delta_phi: np.ndarray) -> np.ndarray:
    # Z(t) = mean exp(i Δϕ) across trajectories at each time
    return np.mean(np.exp(1j * np.asarray(delta_phi, dtype=float)), axis=0)


def cumulants_2_4(delta_phi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # Empirical cumulants of Δϕ(t) across ensemble at each t (central-moment formulas).
    x = np.asarray(delta_phi, dtype=float)
    mu = np.mean(x, axis=0)
    xc = x - mu[None, :]
    m2 = np.mean(xc**2, axis=0)
    m4 = np.mean(xc**4, axis=0)
    k2 = m2
    k4 = m4 - 3.0 * (m2**2)
    return k2, k4


def predict_magnitude_from_even_cumulants(k2: np.ndarray, k4: np.ndarray) -> np.ndarray:
    # c_pred(t) = exp( -k2/2 + k4/24 )  (even-cumulant truncation of log|Z|)
    logc = (-0.5 * np.asarray(k2, dtype=float)) + (np.asarray(k4, dtype=float) / 24.0)
    # Avoid exp overflow; does not affect audit logic since we compare in log-space anyway.
    logc = np.clip(logc, -700.0, 50.0)
    return np.exp(logc)


def fir_smooth_per_traj(x: np.ndarray, m: int) -> np.ndarray:
    """
    Per-trajectory moving-average FIR smoothing (causal-ish centered by convolution pad).
    m must be >= 1; m=1 means identity.
    """
    x = np.asarray(x, dtype=float)
    m = int(m)
    if m <= 1:
        return x.copy()
    k = np.ones(m, dtype=float) / float(m)
    # Convolve along time axis for each trajectory
    y = np.empty_like(x)
    pad = m // 2
    xp = np.pad(x, ((0, 0), (pad, pad)), mode="edge")
    for i in range(x.shape[0]):
        y[i, :] = np.convolve(xp[i, :], k, mode="valid")[: x.shape[1]]
    return y


def apply_c5_measurement_kernel(
    *,
    omega_true: np.ndarray,
    ma_len: int,
    snr_db: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    C5 measurement operator on ω:
      - per-trajectory FIR smoothing (moving average)
      - additive white measurement noise at declared SNR (relative to omega_true std)
    """
    omega_hat = fir_smooth_per_traj(omega_true, int(ma_len))

    snr_db = float(snr_db)
    if not np.isfinite(snr_db):
        return omega_hat

    sig = float(np.std(omega_hat))
    if sig <= 0:
        return omega_hat

    snr_lin = 10.0 ** (snr_db / 20.0)
    noise_sigma = sig / max(snr_lin, 1e-12)
    noise = rng.standard_normal(size=omega_hat.shape).astype(float) * noise_sigma
    return omega_hat + noise


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


def audit_case(
    *,
    name: str,
    t: np.ndarray,
    c_true: np.ndarray,
    c_pred: np.ndarray,
    cfg: S0014Config,
    ma_len: int,
    snr_db: float,
) -> Dict[str, float | str | int]:
    t = np.asarray(t, dtype=float)
    c_true = np.asarray(c_true, dtype=float)
    c_pred = np.asarray(c_pred, dtype=float)

    mask = _audit_mask(t, cfg.t_min, cfg.t_max)
    if not np.any(mask):
        return {"case": name, "verdict": "INCONCLUSIVE"}

    # Coherence floor guard (truth)
    coh_min = float(np.min(c_true[mask]))
    incoherent = bool(coh_min < float(cfg.coherence_floor))

    eps = 1e-18
    dlogc = np.log(np.maximum(c_true, eps)) - np.log(np.maximum(c_pred, eps))
    dlogc_w = dlogc[mask]

    med_abs = float(np.median(np.abs(dlogc_w)))
    p95_abs = float(np.quantile(np.abs(dlogc_w), 0.95))
    drift_slope = _linear_slope(t[mask], dlogc_w)

    # Kernel identifiability gate (no wrong claims)
    kernel_ok = bool((int(ma_len) <= int(cfg.max_cert_ma_len)) and (float(snr_db) >= float(cfg.min_cert_snr_db)))

    # Magnitude closure "OK" criteria (only possible if kernel_ok and not incoherent)
    mag_ok = bool(
        (not incoherent)
        and np.isfinite(med_abs)
        and np.isfinite(p95_abs)
        and np.isfinite(drift_slope)
        and (med_abs <= float(cfg.max_median_abs_dlogc_ok))
        and (p95_abs <= float(cfg.max_p95_abs_dlogc_ok))
        and (abs(drift_slope) <= float(cfg.max_abs_drift_slope_ok))
    )

    if incoherent:
        verdict = "INCONCLUSIVE"
    elif (not kernel_ok) and mag_ok:
        # forbidden "false OK" (kernel too strong to certify)
        verdict = "BOUNDARY"
    elif (not kernel_ok) and (not mag_ok):
        verdict = "BOUNDARY"
    else:
        verdict = "OK" if mag_ok else "FAIL"

    return {
        "case": name,
        "ma_len": int(ma_len),
        "snr_db": float(snr_db),
        "coherence_min": coh_min,
        "median_abs_dlogc": med_abs,
        "p95_abs_dlogc": p95_abs,
        "drift_slope": drift_slope,
        "kernel_ok": int(kernel_ok),
        "mag_ok": int(mag_ok),
        "verdict": verdict,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="S-0014: C5 kernels can hide higher-cumulant failures; forbid false OK.")
    p.add_argument("--seed", type=int, default=S0014Config.seed)
    p.add_argument("--write_outputs", action="store_true")
    args = p.parse_args()

    cfg = S0014Config(seed=int(args.seed))

    t = (np.arange(int(cfg.n_steps)) + 1) * float(cfg.dt)
    mask = _audit_mask(t, cfg.t_min, cfg.t_max)
    n_w = int(np.count_nonzero(mask))

    print(
        f"[S-0014] window=[{cfg.t_min:.1f},{cfg.t_max:.1f}] n_w={n_w} dt={cfg.dt:.3g} "
        f"tol_ok(med,p95,drift)=({cfg.max_median_abs_dlogc_ok:.3g},{cfg.max_p95_abs_dlogc_ok:.3g},{cfg.max_abs_drift_slope_ok:.3g}) "
        f"fail_med_min={cfg.min_median_abs_dlogc_fail:.3g} coh_floor={cfg.coherence_floor:.3g} "
        f"mild(ma={cfg.mild_ma_len},snr={cfg.mild_snr_db:g}dB) strong(ma={cfg.strong_ma_len},snr={cfg.strong_snr_db:g}dB) "
        f"gate(ma<={cfg.max_cert_ma_len}, snr>={cfg.min_cert_snr_db:g}dB)"
    )

    if n_w < int(cfg.min_points):
        print("[S-0014] INCONCLUSIVE: audit window has too few points.")
        return 3

    rng_gauss = np.random.default_rng(cfg.seed + 1)
    rng_k6 = np.random.default_rng(cfg.seed + 2)

    rng_mild = np.random.default_rng(cfg.seed + 11)
    rng_strong = np.random.default_rng(cfg.seed + 12)

    # Truth ω
    omega_gauss = simulate_omega_gauss(cfg, rng=rng_gauss)

    # IMPORTANT: for the certification "must-fail under mild kernel", do NOT use the ultra-rare
    # contamination generator. Use the non-ultra-rare heavy generator (q = cfg.k6_q_mild).
    omega_k6 = simulate_omega_k6_heavy(cfg, rng=rng_k6)

    # Truth Δϕ and truth envelopes
    dphi_gauss_true = delta_phi_from_omega(omega_gauss, cfg.dt)
    dphi_k6_true = delta_phi_from_omega(omega_k6, cfg.dt)

    c_gauss_true = np.abs(complex_coherence(dphi_gauss_true))
    c_k6_true = np.abs(complex_coherence(dphi_k6_true))

    # Two kernel regimes (applied only for inference)
    omega_gauss_hat_mild = apply_c5_measurement_kernel(
        omega_true=omega_gauss, ma_len=cfg.mild_ma_len, snr_db=cfg.mild_snr_db, rng=rng_mild
    )
    omega_k6_hat_mild = apply_c5_measurement_kernel(
        omega_true=omega_k6, ma_len=cfg.mild_ma_len, snr_db=cfg.mild_snr_db, rng=rng_mild
    )

    omega_k6_hat_strong = apply_c5_measurement_kernel(
        omega_true=omega_k6, ma_len=cfg.strong_ma_len, snr_db=cfg.strong_snr_db, rng=rng_strong
    )

    # Inferred Δϕ_hat and inferred cumulants -> prediction
    dphi_gauss_hat_mild = delta_phi_from_omega(omega_gauss_hat_mild, cfg.dt)
    k2_g, k4_g = cumulants_2_4(dphi_gauss_hat_mild)
    c_pred_g = predict_magnitude_from_even_cumulants(k2_g, k4_g)

    dphi_k6_hat_mild = delta_phi_from_omega(omega_k6_hat_mild, cfg.dt)
    k2_km, k4_km = cumulants_2_4(dphi_k6_hat_mild)
    c_pred_km = predict_magnitude_from_even_cumulants(k2_km, k4_km)

    dphi_k6_hat_strong = delta_phi_from_omega(omega_k6_hat_strong, cfg.dt)
    k2_ks, k4_ks = cumulants_2_4(dphi_k6_hat_strong)
    c_pred_ks = predict_magnitude_from_even_cumulants(k2_ks, k4_ks)

    # Audits compare TRUTH envelope vs prediction from inferred cumulants
    audits: List[Dict[str, float | str | int]] = []
    audits.append(
        audit_case(
            name="GAUSS_MILD",
            t=t,
            c_true=c_gauss_true,
            c_pred=c_pred_g,
            cfg=cfg,
            ma_len=cfg.mild_ma_len,
            snr_db=cfg.mild_snr_db,
        )
    )
    audits.append(
        audit_case(
            name="K6_MILD",
            t=t,
            c_true=c_k6_true,
            c_pred=c_pred_km,
            cfg=cfg,
            ma_len=cfg.mild_ma_len,
            snr_db=cfg.mild_snr_db,
        )
    )
    audits.append(
        audit_case(
            name="K6_STRONG",
            t=t,
            c_true=c_k6_true,
            c_pred=c_pred_ks,
            cfg=cfg,
            ma_len=cfg.strong_ma_len,
            snr_db=cfg.strong_snr_db,
        )
    )

    # Print per-case summaries
    for r in audits:
        verdict = str(r.get("verdict", "INCONCLUSIVE"))
        if verdict == "INCONCLUSIVE":
            print(f"[S-0014:{r['case']}] verdict=INCONCLUSIVE")
            continue
        print(
            f"[S-0014:{r['case']}] "
            f"ma={int(r['ma_len'])} snr_db={float(r['snr_db']):.3g} "
            f"coh_min={float(r['coherence_min']):.3g} "
            f"med|dlogc|={float(r['median_abs_dlogc']):.3g} p95={float(r['p95_abs_dlogc']):.3g} drift={float(r['drift_slope']):.3g} "
            f"kernel_ok={int(r['kernel_ok'])} mag_ok={int(r['mag_ok'])} verdict={verdict}"
        )

    # Certification logic:
    #  - GAUSS_MILD must be OK
    #  - K6_MILD must FAIL strongly (median residual >= min_median_abs_dlogc_fail)
    #  - K6_STRONG must not be OK (must be BOUNDARY or INCONCLUSIVE or FAIL; but specifically forbids OK)
    #  - If K6_STRONG is OK -> hard failure ("false OK" permitted)
    results = {str(r["case"]): r for r in audits}

    any_inconclusive = any(str(r.get("verdict")) == "INCONCLUSIVE" for r in audits)

    gauss_ok = (str(results["GAUSS_MILD"].get("verdict")) == "OK")
    k6_mild_verdict = str(results["K6_MILD"].get("verdict"))
    k6_mild_fail = (k6_mild_verdict == "FAIL")

    k6_mild_med = float(results["K6_MILD"].get("median_abs_dlogc", float("nan")))
    k6_mild_strong = bool(np.isfinite(k6_mild_med) and (k6_mild_med >= float(cfg.min_median_abs_dlogc_fail)))

    k6_strong_verdict = str(results["K6_STRONG"].get("verdict"))
    k6_strong_ok_forbidden = (k6_strong_verdict == "OK")

    if k6_strong_ok_forbidden:
        print("[S-0014] AUDIT FAILED: forbidden OK under strong C5 (false-pass permitted).")
        exit_code = 2
    elif any_inconclusive:
        print("[S-0014] INCONCLUSIVE")
        exit_code = 3
    elif gauss_ok and k6_mild_fail and k6_mild_strong and (k6_strong_verdict in ("BOUNDARY", "FAIL", "INCONCLUSIVE")):
        print("[S-0014] AUDIT PASSED")
        exit_code = 0
    else:
        print(
            "[S-0014] AUDIT FAILED: "
            f"gauss_ok={int(gauss_ok)} "
            f"k6_mild_fail={int(k6_mild_fail)} "
            f"k6_mild_strong={int(k6_mild_strong)} "
            f"k6_strong_verdict={k6_strong_verdict}"
        )
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

        rows = [["case", "ma_len", "snr_db", "coh_min", "med_abs_dlogc", "p95_abs_dlogc", "drift_slope", "kernel_ok", "mag_ok", "verdict"]]
        for r in audits:
            rows.append(
                [
                    str(r.get("case", "")),
                    int(r.get("ma_len", -1)) if "ma_len" in r else "",
                    float(r.get("snr_db", float("nan"))) if "snr_db" in r else "",
                    float(r.get("coherence_min", float("nan"))) if "coherence_min" in r else "",
                    float(r.get("median_abs_dlogc", float("nan"))) if "median_abs_dlogc" in r else "",
                    float(r.get("p95_abs_dlogc", float("nan"))) if "p95_abs_dlogc" in r else "",
                    float(r.get("drift_slope", float("nan"))) if "drift_slope" in r else "",
                    int(r.get("kernel_ok", 0)) if "kernel_ok" in r else "",
                    int(r.get("mag_ok", 0)) if "mag_ok" in r else "",
                    str(r.get("verdict", "")),
                ]
            )

        _write_csv_with_provenance_header(out_dir / cfg.out_cases_csv, header_kv, rows)
        _write_csv_with_provenance_header(
            out_dir / cfg.out_audit_csv,
            header_kv,
            [
                ["field", "value"],
                ["exit_code", exit_code],
                ["seed", int(cfg.seed)],
                ["dt", float(cfg.dt)],
                ["n_steps", int(cfg.n_steps)],
                ["n_trajectories", int(cfg.n_trajectories)],
                ["t_min", float(cfg.t_min)],
                ["t_max", float(cfg.t_max)],
                ["coherence_floor", float(cfg.coherence_floor)],
                ["mild_ma_len", int(cfg.mild_ma_len)],
                ["mild_snr_db", float(cfg.mild_snr_db)],
                ["strong_ma_len", int(cfg.strong_ma_len)],
                ["strong_snr_db", float(cfg.strong_snr_db)],
                ["max_cert_ma_len", int(cfg.max_cert_ma_len)],
                ["min_cert_snr_db", float(cfg.min_cert_snr_db)],
                ["k6_b", float(cfg.k6_b)],
                ["gauss_ok", int(gauss_ok)],
                ["k6_mild_fail", int(k6_mild_fail)],
                ["k6_mild_strong", int(k6_mild_strong)],
                ["k6_strong_verdict", k6_strong_verdict],
            ],
        )

        # Optional diagnostic plot (requires matplotlib only when writing outputs)
        import matplotlib.pyplot as plt  # noqa: PLC0415

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        wmask = mask
        # Plot dlogc in window for the three cases
        eps = 1e-18

        dlogc_gauss = np.log(np.maximum(c_gauss_true, eps)) - np.log(np.maximum(c_pred_g, eps))
        dlogc_k6m = np.log(np.maximum(c_k6_true, eps)) - np.log(np.maximum(c_pred_km, eps))
        dlogc_k6s = np.log(np.maximum(c_k6_true, eps)) - np.log(np.maximum(c_pred_ks, eps))

        ax.plot(t[wmask], dlogc_gauss[wmask], label="GAUSS_MILD")
        ax.plot(t[wmask], dlogc_k6m[wmask], label="K6_MILD")
        ax.plot(t[wmask], dlogc_k6s[wmask], label="K6_STRONG")

        ax.set_xlabel("t")
        ax.set_ylabel("log(c_true) - log(c_pred_from_hat)")
        ax.set_title("S-0014: magnitude-closure residual under C5 kernels (window-restricted)")
        ax.legend()

        fig.savefig(out_dir / cfg.out_error_png, dpi=160, bbox_inches="tight")
        plt.close(fig)

        print(f"Wrote (untracked): {out_dir / cfg.out_cases_csv}")
        print(f"Wrote (untracked): {out_dir / cfg.out_audit_csv}")
        print(f"Wrote (untracked): {out_dir / cfg.out_error_png}")

    return exit_code

if __name__ == "__main__":
    raise SystemExit(main())
