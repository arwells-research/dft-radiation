
#!/usr/bin/env python3
"""
S-0015 — Cross-trajectory coupling can mimic long-memory κ2 scaling (C2 ⟷ C4 confusion),
         with refusal via a window-restricted variance-ratio coupling gate.

Goal
----
Certify classification integrity:

- Provide a true long-memory (C2) baseline whose κ2-slope α lies in the declared LM band
  over a fixed audit window.
- Provide an OU (C1) baseline whose κ2-slope α lies in the declared OU band over the same window.
- Provide a cross-trajectory coupled construction (C4-style shared-mode injection) that can
  yield LM-like κ2-slope, and enforce refusal logic so it cannot be certified as OK_LM.

Key rule (integrity):
- If cross-trajectory coupling is detected (via variance-ratio on ω in the audit window),
  the case must be tagged BOUNDARY (never OK_OU or OK_LM), regardless of κ2-slope.

Construction
------------
Cases (fixed):
  - C1_OU_BASE: independent OU ω per trajectory.
  - C2_LM_TRUE: independent fGn ω per trajectory (Davies-Harte, copied from S-0001 style).
  - C4_COUPLED: independent OU ω plus shared OU latent ω_shared(t) added identically to all trajectories.

Coupling detector (window-restricted on ω):
  r_mean := Var_t( mean_i ω_i(t) ) / mean_i Var_t( ω_i(t) )
Independence scale is ~ 1/nT, so fixed bound:
  coupling_bound = coupling_factor / n_trajectories

Classifier:
  κ2(t) := Var_i[Δϕ_i(t)] where Δϕ = ∫ ω ds (discrete cumulative sum)
  Fit α from log κ2 vs log t over a declared fixed window.
  Admissibility: min_r2, min_k2_end, min_points.

Exit codes
----------
  0 PASS
  2 FAIL (wrong claim or required baselines failed)
  3 INCONCLUSIVE (required baselines inadmissible)

Outputs (only with --write_outputs)
----------------------------------
  toys/outputs/s0015_cases.csv
  toys/outputs/s0015_audit.csv
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
class S0015Config:
    seed: int = 15015

    dt: float = 0.20
    n_steps: int = 512
    n_trajectories: int = 1024

    # Fixed audit window (absolute times; no scanning/optimization)
    t_min: float = 20.0
    t_max: float = 40.0
    min_points: int = 24

    # κ2-slope admissibility
    min_r2: float = 0.985
    min_k2_end: float = 5.0e-3

    # Regime bands (declared, no scanning)
    ou_alpha_min: float = 0.82
    ou_alpha_max: float = 1.18

    lm_alpha_min: float = 1.30
    lm_alpha_max: float = 1.70
    fgn_h: float = 0.75

    # Coupling detector: r_mean > coupling_factor / nT indicates coupling
    coupling_factor: float = 10.0

    # OU parameters (Euler-Maruyama)
    ou_theta: float = 1.0
    ou_sigma: float = 1.0

    # Coupled construction gains (no scanning)
    coupled_shared_gain: float = 1.5
    coupled_independent_gain: float = 0.8

    out_dir: str = "toys/outputs"
    out_cases_csv: str = "s0015_cases.csv"
    out_audit_csv: str = "s0015_audit.csv"


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


def _center_global_dc(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x - float(np.mean(x))


def simulate_ou(cfg: S0015Config, rng: np.random.Generator, *, n_traj: int, n_steps: int) -> np.ndarray:
    """
    Discrete-time OU (Euler-Maruyama) with mean 0:
      x_{k+1} = x_k + (-theta x_k) dt + sigma sqrt(dt) * N(0,1)
    """
    dt = float(cfg.dt)
    theta = float(cfg.ou_theta)
    sigma = float(cfg.ou_sigma)

    x = np.zeros((int(n_traj), int(n_steps)), dtype=float)
    z = rng.standard_normal(size=(int(n_traj), int(n_steps))).astype(float)

    for k in range(1, int(n_steps)):
        x[:, k] = x[:, k - 1] + (-theta * x[:, k - 1]) * dt + sigma * np.sqrt(dt) * z[:, k]
    return x


# --- fGn generator copied in style/behavior from S-0001 (Davies-Harte with nonnegative lam clip) ---

def fgn_davies_harte(n_steps: int, H: float, rng: np.random.Generator) -> np.ndarray:
    if not (0.0 < H < 1.0):
        raise ValueError("H must be in (0,1)")
    if n_steps < 2:
        raise ValueError("n_steps must be >= 2")

    n = int(n_steps)
    k = np.arange(0, n, dtype=float)

    gamma = 0.5 * (
        (np.abs(k + 1.0) ** (2.0 * H))
        + (np.abs(k - 1.0) ** (2.0 * H))
        - 2.0 * (np.abs(k) ** (2.0 * H))
    )

    m = 2 * n
    r = np.zeros(m, dtype=float)
    r[0:n] = gamma
    r[n] = 0.0
    r[n + 1 :] = gamma[1:][::-1]

    lam = np.fft.fft(r).real
    lam[lam < 0.0] = 0.0

    W = rng.normal(size=m) + 1j * rng.normal(size=m)
    X = np.fft.ifft(np.sqrt(lam) * W).real

    x = X[:n]
    std = float(np.std(x))
    if std <= 0.0:
        raise RuntimeError("fGn generation produced non-positive std; numerical issue.")
    return (x / std).astype(float)


def simulate_fgn(cfg: S0015Config, rng: np.random.Generator, *, n_traj: int, n_steps: int) -> np.ndarray:
    """
    Independent fGn ω per trajectory, mean ~ 0, Var ~ 1 (per trajectory).
    """
    H = float(cfg.fgn_h)
    out = np.empty((int(n_traj), int(n_steps)), dtype=float)
    for i in range(int(n_traj)):
        out[i, :] = fgn_davies_harte(int(n_steps), H, rng)
    return out


def delta_phi_from_omega(omega: np.ndarray, dt: float) -> np.ndarray:
    return np.cumsum(np.asarray(omega, dtype=float), axis=1) * float(dt)


def k2_of_delta_phi(delta_phi: np.ndarray) -> np.ndarray:
    x = np.asarray(delta_phi, dtype=float)
    mu = np.mean(x, axis=0)
    xc = x - mu[None, :]
    return np.mean(xc**2, axis=0)


def _linfit_r2(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    good = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(good) < 4:
        return float("nan"), float("nan"), float("nan")
    xv = x[good]
    yv = y[good]
    A = np.vstack([np.ones_like(xv), xv]).T
    coef, *_ = np.linalg.lstsq(A, yv, rcond=None)
    a = float(coef[0])
    b = float(coef[1])
    yhat = a + b * xv
    ss_res = float(np.sum((yv - yhat) ** 2))
    ss_tot = float(np.sum((yv - float(np.mean(yv))) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0.0 else float("nan")
    return b, a, float(r2)


def coupling_ratio_r_mean(omega: np.ndarray, mask: np.ndarray) -> float:
    """
    Window-restricted variance-ratio coupling metric:

      r_mean := Var_t( mean_i ω_i(t) ) / mean_i Var_t( ω_i(t) )
    """
    omega = np.asarray(omega, dtype=float)
    if omega.ndim != 2:
        raise ValueError("omega must be (n_traj, n_steps)")
    w = omega[:, mask]
    if w.shape[1] < 4:
        return float("nan")

    m_t = np.mean(w, axis=0)
    var_mean = float(np.var(m_t))

    var_i = np.var(w, axis=1)
    mean_var = float(np.mean(var_i))

    if mean_var <= 0.0 or (not np.isfinite(mean_var)):
        return float("nan")
    return float(var_mean / mean_var)


def classify_case(
    *,
    name: str,
    omega: np.ndarray,
    t: np.ndarray,
    mask: np.ndarray,
    cfg: S0015Config,
) -> Dict[str, float | int | str]:
    omega = np.asarray(omega, dtype=float)

    # Declared centering: global DC removal only
    omega_c = _center_global_dc(omega)

    r_mean = coupling_ratio_r_mean(omega_c, mask)
    coupling_bound = float(cfg.coupling_factor) / float(cfg.n_trajectories)
    coupling_detected = int(bool(np.isfinite(r_mean) and (r_mean > coupling_bound)))

    dphi = delta_phi_from_omega(omega_c, cfg.dt)
    k2 = k2_of_delta_phi(dphi)

    k2_end = float(k2[mask][-1]) if np.any(mask) else float("nan")

    tw = t[mask]
    k2w = k2[mask]
    good = np.isfinite(tw) & np.isfinite(k2w) & (tw > 0.0) & (k2w > 0.0)
    if np.count_nonzero(good) < int(cfg.min_points):
        return {
            "case": name,
            "verdict": "INCONCLUSIVE",
            "reason": "too_few_points",
            "r_mean": float(r_mean),
            "coupling_bound": coupling_bound,
            "coupling_detected": coupling_detected,
        }

    logt = np.log(tw[good])
    logk2 = np.log(k2w[good])
    alpha, intercept, r2 = _linfit_r2(logt, logk2)

    admissible = int(
        bool(
            np.isfinite(alpha)
            and np.isfinite(r2)
            and np.isfinite(k2_end)
            and (float(r2) >= float(cfg.min_r2))
            and (float(k2_end) >= float(cfg.min_k2_end))
            and (np.count_nonzero(good) >= int(cfg.min_points))
        )
    )

    if not admissible:
        return {
            "case": name,
            "alpha": float(alpha),
            "r2": float(r2),
            "k2_end": float(k2_end),
            "r_mean": float(r_mean),
            "coupling_bound": coupling_bound,
            "coupling_detected": coupling_detected,
            "admissible": admissible,
            "verdict": "INCONCLUSIVE",
            "reason": "inadmissible",
        }

    in_ou = bool(float(cfg.ou_alpha_min) <= float(alpha) <= float(cfg.ou_alpha_max))
    in_lm = bool(float(cfg.lm_alpha_min) <= float(alpha) <= float(cfg.lm_alpha_max))

    # S-0015 integrity rule:
    # If coupling is detected, the case is not certifiable as OK_* (refuse classification).
    if coupling_detected == 1:
        tag = "BOUNDARY"
    else:
        if in_ou:
            tag = "OK_OU"
        elif in_lm:
            tag = "OK_LM"
        else:
            tag = "BOUNDARY"

    return {
        "case": name,
        "alpha": float(alpha),
        "r2": float(r2),
        "k2_end": float(k2_end),
        "r_mean": float(r_mean),
        "coupling_bound": coupling_bound,
        "coupling_detected": coupling_detected,
        "admissible": admissible,
        "verdict": tag,
        "reason": "ok",
    }


def main() -> int:
    p = argparse.ArgumentParser(description="S-0015: prevent false LM under cross-trajectory coupling.")
    p.add_argument("--seed", type=int, default=S0015Config.seed)
    p.add_argument("--write_outputs", action="store_true")
    args = p.parse_args()

    cfg = S0015Config(seed=int(args.seed))

    t = (np.arange(int(cfg.n_steps)) + 1) * float(cfg.dt)
    mask = _audit_mask(t, cfg.t_min, cfg.t_max)
    n_w = int(np.count_nonzero(mask))

    coupling_bound = float(cfg.coupling_factor) / float(cfg.n_trajectories)

    print(
        f"[S-0015] window=[{cfg.t_min:.1f},{cfg.t_max:.1f}] n_w={n_w} dt={cfg.dt:.3g} "
        f"admiss(min_r2={cfg.min_r2:.4f}, min_k2_end={cfg.min_k2_end:g}, min_points={cfg.min_points}) "
        f"OU_band=[{cfg.ou_alpha_min:.2f},{cfg.ou_alpha_max:.2f}] "
        f"LM_band=[{cfg.lm_alpha_min:.1f},{cfg.lm_alpha_max:.1f}] (H={cfg.fgn_h:.2f}) "
        f"coupling_bound={coupling_bound:.7g} (factor={cfg.coupling_factor:g}, nT={cfg.n_trajectories})"
    )

    if n_w < int(cfg.min_points):
        print("[S-0015] INCONCLUSIVE: audit window has too few points.")
        return 3

    rng_ou = np.random.default_rng(cfg.seed + 1)
    rng_lm = np.random.default_rng(cfg.seed + 2)
    rng_cpl_ind = np.random.default_rng(cfg.seed + 3)
    rng_cpl_shared = np.random.default_rng(cfg.seed + 4)

    omega_ou = simulate_ou(cfg, rng_ou, n_traj=int(cfg.n_trajectories), n_steps=int(cfg.n_steps))
    omega_lm = simulate_fgn(cfg, rng_lm, n_traj=int(cfg.n_trajectories), n_steps=int(cfg.n_steps))

    omega_ind = simulate_ou(cfg, rng_cpl_ind, n_traj=int(cfg.n_trajectories), n_steps=int(cfg.n_steps))
    omega_shared_1d = simulate_ou(cfg, rng_cpl_shared, n_traj=1, n_steps=int(cfg.n_steps))[0, :]
    omega_cpl = float(cfg.coupled_independent_gain) * omega_ind + float(cfg.coupled_shared_gain) * omega_shared_1d[None, :]

    rows: List[Dict[str, float | int | str]] = []
    rows.append(classify_case(name="C1_OU_BASE", omega=omega_ou, t=t, mask=mask, cfg=cfg))
    rows.append(classify_case(name="C2_LM_TRUE", omega=omega_lm, t=t, mask=mask, cfg=cfg))
    rows.append(classify_case(name="C4_COUPLED", omega=omega_cpl, t=t, mask=mask, cfg=cfg))

    for r in rows:
        case = str(r.get("case", ""))
        verdict = str(r.get("verdict", "INCONCLUSIVE"))
        if verdict == "INCONCLUSIVE":
            reason = str(r.get("reason", ""))
            alpha = float(r.get("alpha", float("nan"))) if "alpha" in r else float("nan")
            r2 = float(r.get("r2", float("nan"))) if "r2" in r else float("nan")
            k2_end = float(r.get("k2_end", float("nan"))) if "k2_end" in r else float("nan")
            r_mean = float(r.get("r_mean", float("nan"))) if "r_mean" in r else float("nan")
            print(
                f"[S-0015:{case}] verdict=INCONCLUSIVE reason={reason} "
                f"alpha={alpha:.4g} r2={r2:.4g} k2_end={k2_end:.4g} r_mean={r_mean:.4g}"
            )
            continue
        print(
            f"[S-0015:{case}] "
            f"alpha={float(r['alpha']):.4g} r2={float(r['r2']):.4g} k2_end={float(r['k2_end']):.4g} "
            f"r_mean={float(r['r_mean']):.4g} coupling_detected={int(r['coupling_detected'])} tag={verdict}"
        )

    by_case = {str(r["case"]): r for r in rows}

    ou_verdict = str(by_case["C1_OU_BASE"].get("verdict", "INCONCLUSIVE"))
    lm_verdict = str(by_case["C2_LM_TRUE"].get("verdict", "INCONCLUSIVE"))
    c4_verdict = str(by_case["C4_COUPLED"].get("verdict", "INCONCLUSIVE"))

    # Required baselines must be admissible/certified
    any_inconclusive_required = (ou_verdict == "INCONCLUSIVE") or (lm_verdict == "INCONCLUSIVE")

    # Forbidden condition (hard fail): OK_LM when coupling_detected=1 (should be impossible by construction now)
    c4_coupling_detected = int(by_case["C4_COUPLED"].get("coupling_detected", 0))
    forbidden_false_ok = (c4_verdict == "OK_LM") and (c4_coupling_detected == 1)

    if forbidden_false_ok:
        print("[S-0015] AUDIT FAILED: forbidden OK_LM under detected coupling (false C2 certification).")
        exit_code = 2
    elif any_inconclusive_required:
        print("[S-0015] INCONCLUSIVE")
        exit_code = 3
    else:
        ou_ok = (ou_verdict == "OK_OU")
        lm_ok = (lm_verdict == "OK_LM")
        c4_ok = (c4_verdict == "BOUNDARY")

        if ou_ok and lm_ok and c4_ok:
            print("[S-0015] AUDIT PASSED")
            exit_code = 0
        else:
            print(f"[S-0015] AUDIT FAILED: ou={ou_verdict} lm={lm_verdict} c4={c4_verdict}")
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

        cases_rows: List[List[object]] = [
            [
                "case",
                "verdict",
                "reason",
                "alpha",
                "r2",
                "k2_end",
                "r_mean",
                "coupling_bound",
                "coupling_detected",
                "admissible",
            ]
        ]
        for r in rows:
            cases_rows.append(
                [
                    str(r.get("case", "")),
                    str(r.get("verdict", "")),
                    str(r.get("reason", "")),
                    float(r.get("alpha", float("nan"))) if "alpha" in r else "",
                    float(r.get("r2", float("nan"))) if "r2" in r else "",
                    float(r.get("k2_end", float("nan"))) if "k2_end" in r else "",
                    float(r.get("r_mean", float("nan"))) if "r_mean" in r else "",
                    float(r.get("coupling_bound", float("nan"))) if "coupling_bound" in r else "",
                    int(r.get("coupling_detected", 0)) if "coupling_detected" in r else "",
                    int(r.get("admissible", 0)) if "admissible" in r else "",
                ]
            )

        _write_csv_with_provenance_header(out_dir / cfg.out_cases_csv, header_kv, cases_rows)

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
                ["min_r2", float(cfg.min_r2)],
                ["min_k2_end", float(cfg.min_k2_end)],
                ["ou_alpha_min", float(cfg.ou_alpha_min)],
                ["ou_alpha_max", float(cfg.ou_alpha_max)],
                ["lm_alpha_min", float(cfg.lm_alpha_min)],
                ["lm_alpha_max", float(cfg.lm_alpha_max)],
                ["fgn_h", float(cfg.fgn_h)],
                ["coupling_factor", float(cfg.coupling_factor)],
                ["coupling_bound", float(coupling_bound)],
                ["ou_verdict", ou_verdict],
                ["lm_verdict", lm_verdict],
                ["c4_verdict", c4_verdict],
                ["c4_coupling_detected", int(c4_coupling_detected)],
            ],
        )

        print(f"Wrote (untracked): {out_dir / cfg.out_cases_csv}")
        print(f"Wrote (untracked): {out_dir / cfg.out_audit_csv}")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
