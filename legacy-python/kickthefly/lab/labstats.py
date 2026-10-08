"""Statistics for Lab results: mean with a 95% confidence interval, and comparisons with a control.

Repeated trials pair each perturbed fly with an unperturbed fly of the same seed (same noise, same stimuli), so the
default comparison is paired: the Wilcoxon signed-rank test (no normality assumption), with the paired t-test reported
alongside. For yes/no outcomes pooled over flies (escapes, proboscis extensions) Fisher's exact test on the 2x2 counts
is reported too.
"""
from __future__ import annotations

import math

import numpy as np


def mean_ci(values, level: float = 0.95) -> dict:
    """Mean, SD, n and a t-distribution confidence interval (nan-safe; n < 2 gives no interval)."""
    from scipy import stats

    v = np.asarray([x for x in values if x is not None and not (isinstance(x, float) and math.isnan(x))], float)
    n = len(v)
    if n == 0:
        return dict(mean=float("nan"), sd=float("nan"), n=0, lo=float("nan"), hi=float("nan"))
    m = float(v.mean())
    if n < 2:
        return dict(mean=m, sd=float("nan"), n=1, lo=float("nan"), hi=float("nan"))
    sd = float(v.std(ddof=1))
    half = float(stats.t.ppf(0.5 + level / 2, n - 1) * sd / math.sqrt(n))
    return dict(mean=m, sd=sd, n=n, lo=m - half, hi=m + half)


def paired(treated, control) -> dict:
    """Paired comparison over matched seeds. p-values are two-sided."""
    from scipy import stats

    a, b = np.asarray(treated, float), np.asarray(control, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    a, b = a[ok], b[ok]
    d = a - b
    out = dict(n=int(len(d)), mean_difference=float(d.mean()) if len(d) else float("nan"),
               test="Wilcoxon signed-rank (paired, two-sided)")
    if len(d) < 2 or np.allclose(d, 0):
        out.update(p_value=1.0 if len(d) else float("nan"), t_p_value=float("nan"))
        return out
    out["p_value"] = float(stats.wilcoxon(a, b, zero_method="wilcox").pvalue)
    t = stats.ttest_rel(a, b)
    out["t_p_value"] = float(t.pvalue) if np.isfinite(t.pvalue) else float("nan")
    out["difference_ci"] = mean_ci(d)
    return out


def fisher(yes_a: int, n_a: int, yes_b: int, n_b: int) -> float:
    from scipy import stats

    return float(stats.fisher_exact([[yes_a, n_a - yes_a], [yes_b, n_b - yes_b]]).pvalue)


def fmt_p(p: float) -> str:
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "p = n/a"
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
