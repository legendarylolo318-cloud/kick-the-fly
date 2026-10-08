"""Lab individuality assay and consistency validation suite.

Measures behavioural individuality across repeated sessions with varying noise seeds,
comparing diverse individuals against identical-brain controls (noise only).

Metrics:
- Intraclass Correlation (ICC(1,1)) with 95% CI for:
    - Looming escape latency / peak DNp01 response
    - Sugar feeding drive (MN9 response ratio)
    - Turning bias (DNa01/DNa02 right vs left difference)
    - T-maze olfactory conditioning (Performance Index PI)
- Pre-fixed validation test (Kain et al. 2012; Linneweber et al. 2020):
    Individuals are more consistent with themselves across sessions than with other individuals.
    Evaluated with paired Wilcoxon signed-rank test on Euclidean behavioral distances.
- Adult validated behavior pass rates across individuality settings (off, subtle, strong).
"""
from __future__ import annotations

import math
from typing import Any
import numpy as np
from scipy import stats

from kickthefly.core import simcore
from kickthefly.core.individuality import compute_fly_gains
from kickthefly.lab import assays


def compute_icc_1_1(data: np.ndarray) -> tuple[float, tuple[float, float], float]:
    """Compute ICC(1, 1) one-way random effects model with 95% CI.
    
    data: shape (n_individuals, k_sessions)
    Returns: (icc, (ci_lower, ci_upper), p_value)
    """
    n, k = data.shape
    if n < 2 or k < 2:
        return 0.0, (0.0, 0.0), 1.0

    mean_indiv = np.mean(data, axis=1)
    grand_mean = np.mean(data)

    ss_total = np.sum((data - grand_mean) ** 2)
    ss_between = k * np.sum((mean_indiv - grand_mean) ** 2)
    ss_within = ss_total - ss_between

    df_b = n - 1
    df_w = n * (k - 1)

    ms_b = ss_between / max(df_b, 1)
    ms_w = ss_within / max(df_w, 1)

    if (ms_b + (k - 1) * ms_w) <= 1e-12:
        return 0.0, (0.0, 0.0), 1.0

    icc = float((ms_b - ms_w) / (ms_b + (k - 1) * ms_w))
    f_stat = ms_b / max(ms_w, 1e-12)
    p_val = float(1.0 - stats.f.cdf(f_stat, df_b, df_w))

    # 95% Confidence Interval using F distribution
    alpha = 0.05
    f_lower = f_stat / max(1e-12, float(stats.f.ppf(1.0 - alpha / 2.0, df_b, df_w)))
    f_upper = f_stat * float(stats.f.ppf(1.0 - alpha / 2.0, df_w, df_b))

    ci_l = float((f_lower - 1.0) / (f_lower + k - 1.0))
    ci_u = float((f_upper - 1.0) / (f_upper + k - 1.0))
    ci_l = float(np.clip(ci_l, -1.0, 1.0))
    ci_u = float(np.clip(ci_u, -1.0, 1.0))

    return icc, (ci_l, ci_u), p_val


def measure_fly_session(
    fly_seed: int,
    session: int,
    mode: str = "subtle",
    backend: str = "cpu",
) -> dict[str, float]:
    """Run one fly through the 4 assay readouts in a specific session."""
    # Individual fly gets a unique brain seed, and session adds noise perturbation
    noise_seed = (int(fly_seed) * 1009 + int(session) * 31337) & 0x7FFFFFFF

    # Create brain with this fly's individuality setting
    br = simcore.new_brain(
        seed=noise_seed,
        brain="adult",
        individuality=mode,
        backend=backend,
        warmup=60,
    )
    g = assays.groups(br)

    # 1. Looming: drive LPLC2/LC4, measure peak DNp01 response ratio
    res_loom = assays.pathway_response(br, g["loom"], {"dnp01": g["dnp01"]}, pre=100, stim=100)
    base_l, stim_l = res_loom["dnp01"]
    loom_ratio = float(stim_l / max(0.1, base_l))

    # 2. Sugar feeding: drive head gustatory neurons, measure MN9 response ratio
    res_sugar = assays.pathway_response(br, g["head_gust"][:30], {"mn9": g["mn9"]}, pre=100, stim=100)
    base_s, stim_s = res_sugar["mn9"]
    sugar_ratio = float(stim_s / max(0.1, base_s))

    # 3. Turning bias: compare right vs left steering DNs (DNa01 & DNa02)
    res_turn = assays.pathway_response(br, g["sensory"][:20], {"dn": g["dn"]}, pre=100, stim=100)
    rates = br.sim.activity.rates()
    # DNa01 and DNa02 indices
    t = br.types.astype(str)
    inst = br.graph.instance.astype(str)
    dna_r = np.flatnonzero(np.char.endswith(inst, "_R") & ((t == "DNa01") | (t == "DNa02")))
    dna_l = np.flatnonzero(np.char.endswith(inst, "_L") & ((t == "DNa01") | (t == "DNa02")))
    rate_r = float(np.mean(rates[dna_r])) if len(dna_r) else 0.01
    rate_l = float(np.mean(rates[dna_l])) if len(dna_l) else 0.01
    turn_bias = float((rate_r - rate_l) / max(0.005, rate_r + rate_l))

    # 4. T-maze conditioning: single conditioning cycle test
    tmaze_res = assays.tmaze_fly(noise_seed, brain=br, cycles=2, test_trials=10)
    tmaze_pi = float(tmaze_res["pi"])

    return {
        "looming": loom_ratio,
        "sugar": sugar_ratio,
        "turning": turn_bias,
        "tmaze": tmaze_pi,
    }


def run_individuality_assay(
    n_individuals: int = 10,
    n_sessions: int = 4,
    mode: str = "subtle",
    control_mode: str = "off",
    backend: str = "cpu",
) -> dict[str, Any]:
    """Run repeated assay for n_individuals across n_sessions vs identical controls."""
    assays_names = ("looming", "sugar", "turning", "tmaze")

    # Arrays for data: shape (n_individuals, n_sessions)
    data_exp = {name: np.zeros((n_individuals, n_sessions), dtype=np.float32) for name in assays_names}
    data_ctrl = {name: np.zeros((n_individuals, n_sessions), dtype=np.float32) for name in assays_names}

    # Run Experimental (diverse individuals)
    for i in range(n_individuals):
        fly_seed = 100 + i
        for s in range(n_sessions):
            res = measure_fly_session(fly_seed, s, mode=mode, backend=backend)
            for k in assays_names:
                data_exp[k][i, s] = res[k]

    # Run Control (identical connectome, noise variation only)
    for i in range(n_individuals):
        fly_seed = 100 + i
        for s in range(n_sessions):
            # In control, fly_seed is fixed or mode is "off"
            res = measure_fly_session(999, s + i * n_sessions, mode=control_mode, backend=backend)
            for k in assays_names:
                data_ctrl[k][i, s] = res[k]

    # Compute ICC for each assay
    results_icc = {}
    for k in assays_names:
        icc_exp, ci_exp, p_exp = compute_icc_1_1(data_exp[k])
        icc_ctrl, ci_ctrl, p_ctrl = compute_icc_1_1(data_ctrl[k])
        results_icc[k] = {
            "exp_icc": icc_exp,
            "exp_ci": ci_exp,
            "exp_p": p_exp,
            "ctrl_icc": icc_ctrl,
            "ctrl_ci": ci_ctrl,
            "ctrl_p": p_ctrl,
        }

    # Consistency Test: Within-individual distance vs Between-individual distance
    # Z-score normalize across all measurements for fair Euclidean combination
    mat_exp = np.stack([data_exp[k] for k in assays_names], axis=-1)  # (n, k, 4)
    std_vals = np.std(mat_exp, axis=(0, 1), keepdims=True)
    std_vals = np.where(std_vals < 1e-6, 1.0, std_vals)
    norm_mat = (mat_exp - np.mean(mat_exp, axis=(0, 1), keepdims=True)) / std_vals

    within_dists = []
    between_dists = []

    for i in range(n_individuals):
        # Within-individual distances between sessions
        for s1 in range(n_sessions):
            for s2 in range(s1 + 1, n_sessions):
                within_dists.append(float(np.linalg.norm(norm_mat[i, s1] - norm_mat[i, s2])))

        # Between-individual distances
        for j in range(i + 1, n_individuals):
            for s in range(n_sessions):
                between_dists.append(float(np.linalg.norm(norm_mat[i, s] - norm_mat[j, s])))

    mean_within = float(np.mean(within_dists)) if within_dists else 0.0
    mean_between = float(np.mean(between_dists)) if between_dists else 0.0

    # Paired test across individuals
    indiv_within = [
        np.mean([np.linalg.norm(norm_mat[i, s1] - norm_mat[i, s2])
                 for s1 in range(n_sessions) for s2 in range(s1 + 1, n_sessions)])
        for i in range(n_individuals)
    ]
    indiv_between = [
        np.mean([np.linalg.norm(norm_mat[i, s] - norm_mat[j, s])
                 for j in range(n_individuals) if j != i for s in range(n_sessions)])
        for i in range(n_individuals)
    ]

    try:
        w_stat, p_val = stats.wilcoxon(indiv_within, indiv_between, alternative="less")
        p_str = "< 0.001" if p_val < 0.001 else f"{p_val:.4f}"
    except Exception:
        w_stat, p_val, p_str = 0.0, 1.0, "1.0"

    consistency_pass = mean_within < mean_between and p_val < 0.05

    return {
        "n_individuals": n_individuals,
        "n_sessions": n_sessions,
        "mode": mode,
        "icc": results_icc,
        "within_distance": mean_within,
        "between_distance": mean_between,
        "wilcoxon_stat": float(w_stat),
        "wilcoxon_p": p_val,
        "wilcoxon_p_str": p_str,
        "consistency_pass": consistency_pass,
    }


def evaluate_validated_behaviors_pass_rates(
    seeds: tuple[int, ...] = tuple(range(1000, 1010)),
    modes: tuple[str, ...] = ("off", "subtle", "strong"),
) -> dict[str, dict[str, float]]:
    """Report pass rate of each adult validated behavior across individuality settings."""
    results: dict[str, dict[str, float]] = {}

    for mode in modes:
        loom_passes = 0
        sugar_passes = 0
        jo_passes = 0
        tmaze_passes = 0
        n_seeds = len(seeds)

        for seed in seeds:
            br = simcore.new_brain(seed=seed, individuality=mode, warmup=60)
            g = assays.groups(br)

            # 1. Looming -> GF DNp01 (pass if ratio >= 1.5)
            r_loom = assays.pathway_response(br, g["loom"], {"dnp01": g["dnp01"]}, pre=100, stim=100)
            if r_loom["dnp01"][1] / max(0.1, r_loom["dnp01"][0]) >= 1.5:
                loom_passes += 1

            # 2. Sugar -> MN9 (pass if ratio >= 1.5)
            r_sugar = assays.pathway_response(br, g["head_gust"][:30], {"mn9": g["mn9"]}, pre=100, stim=100)
            if r_sugar["mn9"][1] / max(0.1, r_sugar["mn9"][0]) >= 1.5:
                sugar_passes += 1

            # 3. JO-C/E -> aDN (pass if ratio >= 1.5)
            r_jo = assays.pathway_response(br, g["jo_ce"][:30], {"adn": g["adn"]}, pre=100, stim=100)
            if r_jo["adn"][1] / max(0.1, r_jo["adn"][0]) >= 1.5:
                jo_passes += 1

            # 4. T-maze conditioning (pass if PI >= 0.20)
            tm = assays.tmaze_fly(seed, brain=br, cycles=2, test_trials=10)
            if tm["pi"] >= 0.20:
                tmaze_passes += 1

        results[mode] = {
            "looming_gf": loom_passes / n_seeds,
            "sugar_mn9": sugar_passes / n_seeds,
            "jo_adn": jo_passes / n_seeds,
            "tmaze": tmaze_passes / n_seeds,
        }

    return results
