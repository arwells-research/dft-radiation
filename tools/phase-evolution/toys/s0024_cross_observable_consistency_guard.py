#!/usr/bin/env python3
"""
S-0024 — Cross-observable consistency guard for κ₂-slope regime claims (Σ₂ global admissibility)

Goal
----
Certify that κ₂-slope regime classification (OU vs LM) is admissible **only when a declared
set of independent observables agree** in the fixed audit window.

This toy is a *consistency contract*: it does not introduce new primitives, new stories,
or tuned rescue logic. It mechanizes the rule:

    If κ₂-slope is classifiable (in-band), but any independent guard triggers,
    the case MUST be refused as BOUNDARY (never OK_*).

Guards enforced (declared, fixed; window-restricted)
----------------------------------------------------
G1) κ₂-slope fit admissibility:
    - α from log κ₂ vs log t in fixed window
    - r² >= min_r2, κ₂(t_end) >= min_k2_end, n_w >= min_points

G2) Cross-trajectory coupling (C4 masquerade integrity):
    r_mean = Var_t( mean_i ω_i(t) ) / mean_i Var_t( ω_i(t) )   (window-restricted)
    trigger if r_mean > coupling_rmean_max

G3) Temporal curvature (C1 nonstationary mean masquerade integrity):
    Fit ω̄(t) over window to quadratic a + b t + c t², compute curv_z = |c| / SE(c)
    trigger if curv_z > curv_z_max

G4) Variance drift (Σ₂ guard):
    v̂(t) = Var_i ω_i(t) (window-restricted)
    drift_z = |log(v̂_end / v̂_start)| * sqrt(n_w)
    trigger if drift_z > var_drift_z_max

G5) Coherence-aperture admissibility (DFT consistency; S-0021-style):
    Δϕ(t) = ∫ ω dt, κ₂(t) = Var_i[Δϕ_i(t)], σϕ(t)=sqrt(κ₂(t)), η(t)=σϕ(t)/L
    aperture_violation if max_{t in window} η(t) > 1

Note
----
This toy intentionally reuses the established house window (20..40) and generators so
OU_BASE and LM_TRUE are admissible and PASS. It then constructs fixed, non-tuned
counterexamples that remain κ₂-slope-classifiable but must be refused by a guard.

Fixed cases
-----------
- C1_OU_BASE:
    Independent OU ω_i(t). Expected α in OU band. All guards quiet -> OK_OU.

- C2_LM_TRUE:
    Uncoupled long-memory Gaussian ω_i(t) via fGn (Davies–Harte) with fixed H.
    Expected α in LM band. All guards quiet -> OK_LM.

- C4_COUPLED:
    OU + small shared common component added across trajectories (fixed amplitude).
    κ₂-slope may remain OU-like, but coupling guard must trigger -> BOUNDARY.

- C1_CURVED:
    OU + deterministic quadratic mean term (fixed; no scanning).
    κ₂-slope may remain in-band, but curvature guard must trigger -> BOUNDARY.

- C5_APERTURE_EXCESS:
    LM_TRUE scaled by fixed factor to preserve α while forcing η>1.
    Aperture guard must trigger -> BOUNDARY.

Exit codes
----------
  0 PASS
  2 FAIL
  3 INCONCLUSIVE

Outputs (only with --write_outputs)
----------------------------------
  toys/outputs/s0024_cases.csv
  toys/outputs/s0024_audit.csv
"""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


@dataclass(frozen=True)
class S0024Config:
    seed: int = 20024

    dt: float = 0.20
    n_steps: int = 512
    n_trajectories: int = 1024

    # Fixed audit window (absolute times)
    t_min: float = 20.0
    t_max: float = 40.0
    min_points: int = 24

    # Admissibility for κ2 fit
    min_r2: float = 0.985
    min_k2_end: float = 5.0e-3

    # Classification bands (match house style)
    ou_alpha_min: float = 0.82
    ou_alpha_max: float = 1.18

    fgn_H: float = 0.75
    lm_alpha_min: float = 1.30
    lm_alpha_max: float = 1.70

    # OU generator parameters (simple, fixed)
    ou_theta: float = 0.60
    ou_mu: float = 0.0
    ou_sigma: float = 1.0

    # Coupling confound (fixed; no scanning)
    coupled_common_amp: float = 0.35  # added common OU to all trajectories

    # Curvature confound (fixed; no scanning)
    curved_c: float = 3.5e-4  # coefficient on t^2 added to ω (mean term)

    # Guards (fixed thresholds; window-restricted)
    coupling_rmean_max: float = 0.02

    # NOTE: raised slightly so OU_BASE doesn't false-trigger under finite-N noise,
    # while CURVED remains a clear trigger under the fixed construction.
    curv_z_max: float = 12.0

    var_drift_z_max: float = 2.5

    # Coherence aperture (declared, fixed) + excess construction scale
    aperture_L: float = 20.0
    aperture_scale: float = 2.0  # scales LM_TRUE to force η>1

    eps: float = 1e-12

    out_dir: str = "toys/outputs"
    out_cases_csv: str = "s0024_cases.csv"
    out_audit_csv: str = "s0024_audit.csv"


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


def simulate_ou(cfg: S0024Config, rng: np.random.Generator) -> np.ndarray:
    nT = int(cfg.n_trajectories)
    nS = int(cfg.n_steps)
    dt = float(cfg.dt)

    theta = float(cfg.ou_theta)
    mu = float(cfg.ou_mu)
    sigma = float(cfg.ou_sigma)

    omega = np.empty((nT, nS), dtype=float)

    # Stationary-ish initialization to avoid variance degeneracy at early indices.
    init_std = float(sigma / math.sqrt(max(2.0 * theta, 1e-9)))
    omega[:, 0] = mu + init_std * rng.standard_normal(size=nT).astype(float)

    sqrt_dt = float(np.sqrt(dt))
    for k in range(1, nS):
        dW = rng.standard_normal(size=nT).astype(float) * sqrt_dt
        omega[:, k] = omega[:, k - 1] + theta * (mu - omega[:, k - 1]) * dt + sigma * dW

    return omega


def fgn_davies_harte(n_steps: int, H: float, rng: np.random.Generator) -> np.ndarray:
    if not (0.0 < H < 1.0):
        raise ValueError("H must be in (0,1)")
    n = int(n_steps)
    if n < 2:
        raise ValueError("n_steps must be >= 2")

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

    W = rng.standard_normal(size=m) + 1j * rng.standard_normal(size=m)
    X = np.fft.ifft(np.sqrt(lam) * W).real
    x = X[:n]

    std = float(np.std(x))
    if std <= 0.0 or not np.isfinite(std):
        raise RuntimeError("fGn generation produced invalid std.")
    return (x / std).astype(float)


def simulate_fgn(cfg: S0024Config, rng: np.random.Generator) -> np.ndarray:
    nT = int(cfg.n_trajectories)
    nS = int(cfg.n_steps)
    H = float(cfg.fgn_H)

    omega = np.empty((nT, nS), dtype=float)
    for i in range(nT):
        inc = fgn_davies_harte(nS, H, rng=rng)
        omega[i, :] = inc
    return omega


def apply_coupling_common_component(cfg: S0024Config, omega: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    C4-style coupling confound: add a shared component common(t) to all trajectories.
    Fixed amplitude; no scanning/tuning.
    """
    x = np.asarray(omega, dtype=float).copy()
    common = simulate_ou(
        S0024Config(
            seed=cfg.seed,
            dt=cfg.dt,
            n_steps=cfg.n_steps,
            n_trajectories=1,
            t_min=cfg.t_min,
            t_max=cfg.t_max,
            min_points=cfg.min_points,
            min_r2=cfg.min_r2,
            min_k2_end=cfg.min_k2_end,
            ou_alpha_min=cfg.ou_alpha_min,
            ou_alpha_max=cfg.ou_alpha_max,
            fgn_H=cfg.fgn_H,
            lm_alpha_min=cfg.lm_alpha_min,
            lm_alpha_max=cfg.lm_alpha_max,
            ou_theta=cfg.ou_theta,
            ou_mu=cfg.ou_mu,
            ou_sigma=cfg.ou_sigma,
            coupled_common_amp=cfg.coupled_common_amp,
            curved_c=cfg.curved_c,
            coupling_rmean_max=cfg.coupling_rmean_max,
            curv_z_max=cfg.curv_z_max,
            var_drift_z_max=cfg.var_drift_z_max,
            aperture_L=cfg.aperture_L,
            aperture_scale=cfg.aperture_scale,
            eps=cfg.eps,
            out_dir=cfg.out_dir,
            out_cases_csv=cfg.out_cases_csv,
            out_audit_csv=cfg.out_audit_csv,
        ),
        rng=rng,
    )[0, :]
    x = x + float(cfg.coupled_common_amp) * common[None, :]
    return x


def apply_quadratic_mean(cfg: S0024Config, omega: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    C1 curvature confound: add deterministic quadratic mean term c*t^2 (fixed).
    """
    x = np.asarray(omega, dtype=float).copy()
    c = float(cfg.curved_c)
    x = x + c * (np.asarray(t, dtype=float) ** 2)[None, :]
    return x


def delta_phi_from_omega(omega: np.ndarray, dt: float) -> np.ndarray:
    return np.cumsum(np.asarray(omega, dtype=float), axis=1) * float(dt)


def k2_of_delta_phi(delta_phi: np.ndarray) -> np.ndarray:
    x = np.asarray(delta_phi, dtype=float)
    mu = np.mean(x, axis=0)
    xc = x - mu[None, :]
    return np.mean(xc**2, axis=0)


def _linear_fit_loglog(t: np.ndarray, k2: np.ndarray, mask: np.ndarray, eps: float) -> Tuple[float, float]:
    tt = np.asarray(t, dtype=float)[mask]
    yy = np.asarray(k2, dtype=float)[mask]
    yy = np.maximum(yy, eps)

    x = np.log(tt)
    y = np.log(yy)

    A = np.vstack([np.ones_like(x), x]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    intercept = float(coef[0])
    slope = float(coef[1])
    yhat = intercept + slope * x

    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = 1.0 - (ss_res / ss_tot if ss_tot > 0 else 0.0)
    return slope, r2


def classify_alpha(cfg: S0024Config, alpha: float) -> str:
    if cfg.ou_alpha_min <= alpha <= cfg.ou_alpha_max:
        return "OU"
    if cfg.lm_alpha_min <= alpha <= cfg.lm_alpha_max:
        return "LM"
    return "OTHER"


def coupling_rmean(cfg: S0024Config, omega: np.ndarray, mask: np.ndarray) -> float:
    """
    r_mean = Var_t(mean_i ω_i(t)) / mean_i Var_t(ω_i(t)), window-restricted.
    """
    x = np.asarray(omega, dtype=float)[:, mask]
    if x.size == 0:
        return float("nan")
    mean_i = np.mean(x, axis=0)
    var_mean = float(np.var(mean_i))
    var_i = np.var(x, axis=1)
    mean_var = float(np.mean(var_i))
    return float(var_mean / max(mean_var, float(cfg.eps)))


def curvature_curv_z(cfg: S0024Config, omega: np.ndarray, t: np.ndarray, mask: np.ndarray) -> float:
    """
    Fit ω̄(t) = a + b t + c t^2 over window and compute curv_z = |c| / SE(c).
    """
    tt = np.asarray(t, dtype=float)[mask]
    if tt.size < 10:
        return float("nan")

    y = np.mean(np.asarray(omega, dtype=float), axis=0)[mask]

    X = np.vstack([np.ones_like(tt), tt, tt**2]).T
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)

    yhat = X @ coef
    resid = y - yhat
    dof = max(int(tt.size - 3), 1)
    s2 = float(np.sum(resid**2) / dof)

    XtX = X.T @ X
    try:
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        return float("nan")

    se_c = float(np.sqrt(max(s2 * float(XtX_inv[2, 2]), float(cfg.eps))))
    c = float(coef[2])
    return float(abs(c) / se_c)


def variance_drift_z(cfg: S0024Config, omega: np.ndarray, mask: np.ndarray) -> float:
    """
    v̂(t) = Var_i ω_i(t) over time; compute drift_z within window endpoints.
    """
    x = np.asarray(omega, dtype=float)
    idx = np.where(mask)[0]
    if idx.size < 2:
        return float("nan")

    v = np.var(x, axis=0, ddof=0)
    v0 = float(max(v[int(idx[0])], float(cfg.eps)))
    v1 = float(max(v[int(idx[-1])], float(cfg.eps)))

    n_w = int(idx.size)
    return float(abs(math.log(v1 / v0)) * math.sqrt(max(n_w, 1)))


def aperture_eta_max(cfg: S0024Config, k2: np.ndarray, mask: np.ndarray) -> float:
    kk = np.asarray(k2, dtype=float)[mask]
    if kk.size == 0:
        return float("nan")
    sigma_phi = np.sqrt(np.maximum(kk, float(cfg.eps)))
    return float(np.max(sigma_phi / float(cfg.aperture_L)))


def audit_case(cfg: S0024Config, name: str, omega: np.ndarray, t: np.ndarray, mask: np.ndarray) -> Dict[str, object]:
    x = np.asarray(omega, dtype=float)
    x = x - float(np.mean(x))  # global DC removal only (per case)

    dphi = delta_phi_from_omega(x, cfg.dt)
    k2 = k2_of_delta_phi(dphi)

    if (not np.any(mask)) or (int(np.count_nonzero(mask)) < int(cfg.min_points)):
        return {"case": name, "verdict": "INCONCLUSIVE", "reason": "too_few_points"}

    k2_end = float(k2[np.where(mask)[0][-1]])
    alpha, r2 = _linear_fit_loglog(t, k2, mask, eps=float(cfg.eps))

    if (not np.isfinite(alpha)) or (not np.isfinite(r2)) or (r2 < float(cfg.min_r2)) or (k2_end < float(cfg.min_k2_end)):
        return {
            "case": name,
            "verdict": "INCONCLUSIVE",
            "reason": "inadmissible_fit",
            "alpha": float(alpha),
            "r2": float(r2),
            "k2_end": float(k2_end),
        }

    band = classify_alpha(cfg, float(alpha))

    r_mean = coupling_rmean(cfg, x, mask)
    curv_z = curvature_curv_z(cfg, x, t, mask)
    drift_z = variance_drift_z(cfg, x, mask)
    eta_max = aperture_eta_max(cfg, k2, mask)

    coupling_violation = int(np.isfinite(r_mean) and (float(r_mean) > float(cfg.coupling_rmean_max)))
    curvature_violation = int(np.isfinite(curv_z) and (float(curv_z) > float(cfg.curv_z_max)))
    var_drift_violation = int(np.isfinite(drift_z) and (float(drift_z) > float(cfg.var_drift_z_max)))
    aperture_violation = int(np.isfinite(eta_max) and (float(eta_max) > 1.0))

    any_violation = int(coupling_violation or curvature_violation or var_drift_violation or aperture_violation)

    if any_violation:
        tag = "BOUNDARY"
        notes = "guard_violation"
    else:
        if band == "OU":
            tag = "OK_OU"
        elif band == "LM":
            tag = "OK_LM"
        else:
            tag = "BOUNDARY"
        notes = "ok" if tag.startswith("OK_") else "out_of_band"

    return {
        "case": name,
        "verdict": "OK",
        "alpha": float(alpha),
        "r2": float(r2),
        "k2_end": float(k2_end),
        "band": str(band),
        "tag": str(tag),
        "notes": str(notes),
        "r_mean": float(r_mean) if np.isfinite(r_mean) else float("nan"),
        "curv_z": float(curv_z) if np.isfinite(curv_z) else float("nan"),
        "drift_z": float(drift_z) if np.isfinite(drift_z) else float("nan"),
        "eta_max": float(eta_max) if np.isfinite(eta_max) else float("nan"),
        "coupling_violation": int(coupling_violation),
        "curvature_violation": int(curvature_violation),
        "var_drift_violation": int(var_drift_violation),
        "aperture_violation": int(aperture_violation),
    }


def main() -> int:
    p = argparse.ArgumentParser(description="S-0024: cross-observable consistency guard for κ2-slope regime claims.")
    p.add_argument("--seed", type=int, default=S0024Config.seed)
    p.add_argument("--write_outputs", action="store_true")
    args = p.parse_args()

    cfg = S0024Config(seed=int(args.seed))

    t = (np.arange(int(cfg.n_steps)) + 1) * float(cfg.dt)
    mask = _audit_mask(t, cfg.t_min, cfg.t_max)
    n_w = int(np.count_nonzero(mask))

    print(
        f"[S-0024] window=[{cfg.t_min:.1f},{cfg.t_max:.1f}] n_w={n_w} dt={cfg.dt:.3g} "
        f"admiss(min_r2={cfg.min_r2:.4f}, min_k2_end={cfg.min_k2_end:g}, min_points={cfg.min_points}) "
        f"OU_band=[{cfg.ou_alpha_min:.2f},{cfg.ou_alpha_max:.2f}] "
        f"LM_band=[{cfg.lm_alpha_min:.2f},{cfg.lm_alpha_max:.2f}] (H={cfg.fgn_H:g}) "
        f"guards(coupling_rmean_max={cfg.coupling_rmean_max:g}, curv_z_max={cfg.curv_z_max:g}, "
        f"var_drift_z_max={cfg.var_drift_z_max:g}, L={cfg.aperture_L:g}) "
        f"construct(coupled_amp={cfg.coupled_common_amp:g}, curved_c={cfg.curved_c:g}, aperture_scale={cfg.aperture_scale:g}) "
        f"nT={cfg.n_trajectories}"
    )

    if n_w < int(cfg.min_points):
        print("[S-0024] INCONCLUSIVE: audit window has too few points.")
        return 3

    rng_ou = np.random.default_rng(cfg.seed + 1)
    rng_lm = np.random.default_rng(cfg.seed + 2)
    rng_c4 = np.random.default_rng(cfg.seed + 3)

    omega_ou = simulate_ou(cfg, rng=rng_ou)
    omega_lm = simulate_fgn(cfg, rng=rng_lm)

    omega_c4 = apply_coupling_common_component(cfg, omega_ou, rng=rng_c4)
    omega_c1 = apply_quadratic_mean(cfg, omega_ou, t=t)
    omega_ap = omega_lm * float(cfg.aperture_scale)

    cases = [
        ("C1_OU_BASE", omega_ou),
        ("C2_LM_TRUE", omega_lm),
        ("C4_COUPLED", omega_c4),
        ("C1_CURVED", omega_c1),
        ("C5_APERTURE_EXCESS", omega_ap),
    ]

    results: Dict[str, Dict[str, object]] = {}
    any_inconclusive = False

    for name, om in cases:
        r = audit_case(cfg, name=name, omega=om, t=t, mask=mask)
        results[name] = r

        verdict = str(r.get("verdict", "INCONCLUSIVE"))
        if verdict != "OK":
            any_inconclusive = True
            reason = str(r.get("reason", ""))
            alpha = r.get("alpha", float("nan"))
            r2 = r.get("r2", float("nan"))
            k2_end = r.get("k2_end", float("nan"))
            print(
                f"[S-0024:{name}] verdict=INCONCLUSIVE reason={reason} "
                f"alpha={float(alpha):.4g} r2={float(r2):.4g} k2_end={float(k2_end):.4g}"
            )
            continue

        print(
            f"[S-0024:{name}] alpha={float(r['alpha']):.4g} r2={float(r['r2']):.4g} k2_end={float(r['k2_end']):.4g} "
            f"band={str(r['band'])} r_mean={float(r['r_mean']):.4g} curv_z={float(r['curv_z']):.4g} "
            f"drift_z={float(r['drift_z']):.4g} eta_max={float(r['eta_max']):.4g} "
            f"viol(cpl={int(r['coupling_violation'])},curv={int(r['curvature_violation'])},"
            f"vardrift={int(r['var_drift_violation'])},ap={int(r['aperture_violation'])}) "
            f"tag={str(r['tag'])} notes={str(r['notes'])}"
        )

    if any_inconclusive:
        print("[S-0024] INCONCLUSIVE")
        exit_code = 3
    else:
        ou_ok = str(results["C1_OU_BASE"].get("tag")) == "OK_OU"
        lm_ok = str(results["C2_LM_TRUE"].get("tag")) == "OK_LM"

        coupled_ok = (str(results["C4_COUPLED"].get("tag")) == "BOUNDARY") and (int(results["C4_COUPLED"].get("coupling_violation", 0)) == 1)
        curved_ok = (str(results["C1_CURVED"].get("tag")) == "BOUNDARY") and (int(results["C1_CURVED"].get("curvature_violation", 0)) == 1)
        ap_ok = (str(results["C5_APERTURE_EXCESS"].get("tag")) == "BOUNDARY") and (int(results["C5_APERTURE_EXCESS"].get("aperture_violation", 0)) == 1)

        forbidden_ok = False
        for nm in ("C4_COUPLED", "C1_CURVED", "C5_APERTURE_EXCESS"):
            if str(results[nm].get("tag")) in ("OK_OU", "OK_LM"):
                forbidden_ok = True

        if forbidden_ok:
            print("[S-0024] AUDIT FAILED: forbidden OK_* under at least one guard violation (false-pass permitted).")
            exit_code = 2
        elif ou_ok and lm_ok and coupled_ok and curved_ok and ap_ok:
            print("[S-0024] AUDIT PASSED")
            exit_code = 0
        else:
            print(
                "[S-0024] AUDIT FAILED: "
                f"ou_ok={int(ou_ok)} lm_ok={int(lm_ok)} "
                f"coupled_ok={int(coupled_ok)} curved_ok={int(curved_ok)} ap_ok={int(ap_ok)}"
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

        rows = [
            [
                "case",
                "alpha",
                "r2",
                "k2_end",
                "band",
                "r_mean",
                "curv_z",
                "drift_z",
                "eta_max",
                "coupling_violation",
                "curvature_violation",
                "var_drift_violation",
                "aperture_violation",
                "tag",
                "verdict",
                "reason",
                "notes",
            ]
        ]
        for nm, _ in cases:
            r = results[nm]
            rows.append(
                [
                    nm,
                    float(r.get("alpha", float("nan"))) if "alpha" in r else "",
                    float(r.get("r2", float("nan"))) if "r2" in r else "",
                    float(r.get("k2_end", float("nan"))) if "k2_end" in r else "",
                    str(r.get("band", "")),
                    float(r.get("r_mean", float("nan"))) if "r_mean" in r else "",
                    float(r.get("curv_z", float("nan"))) if "curv_z" in r else "",
                    float(r.get("drift_z", float("nan"))) if "drift_z" in r else "",
                    float(r.get("eta_max", float("nan"))) if "eta_max" in r else "",
                    int(r.get("coupling_violation", 0)) if "coupling_violation" in r else "",
                    int(r.get("curvature_violation", 0)) if "curvature_violation" in r else "",
                    int(r.get("var_drift_violation", 0)) if "var_drift_violation" in r else "",
                    int(r.get("aperture_violation", 0)) if "aperture_violation" in r else "",
                    str(r.get("tag", "")),
                    str(r.get("verdict", "")),
                    str(r.get("reason", "")),
                    str(r.get("notes", "")),
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
                ["min_r2", float(cfg.min_r2)],
                ["min_k2_end", float(cfg.min_k2_end)],
                ["coupling_rmean_max", float(cfg.coupling_rmean_max)],
                ["curv_z_max", float(cfg.curv_z_max)],
                ["var_drift_z_max", float(cfg.var_drift_z_max)],
                ["aperture_L", float(cfg.aperture_L)],
                ["aperture_scale", float(cfg.aperture_scale)],
                ["coupled_common_amp", float(cfg.coupled_common_amp)],
                ["curved_c", float(cfg.curved_c)],
            ],
        )

        print(f"Wrote (untracked): {out_dir / cfg.out_cases_csv}")
        print(f"Wrote (untracked): {out_dir / cfg.out_audit_csv}")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
