//! The statistics the tournament, race and validation analyses call (scipy.stats): exact binomial test (two-sided, p = 1/2), Spearman
//! correlation with its t-test p-value, one-sided Wilcoxon signed-rank (`zero_method="wilcox"`), Holm correction. Verified against scipy
//! on fixed inputs (`tests/golden/make_stats.py`).

fn ln_factorials(n: usize) -> Vec<f64> {
    let mut v = vec![0.0; n + 1];
    for i in 1..=n {
        v[i] = v[i - 1] + (i as f64).ln();
    }
    v
}

/// `binom.pmf(k, n, 0.5)` for every k.
fn pmf_half(n: usize) -> Vec<f64> {
    let lf = ln_factorials(n);
    (0..=n).map(|k| (lf[n] - lf[k] - lf[n - k] - n as f64 * std::f64::consts::LN_2).exp()).collect()
}

/// `scipy.stats.binomtest(k, n, 0.5).pvalue` (two-sided).
pub fn binom_two_sided(k: usize, n: usize) -> f64 {
    let pmf = pmf_half(n);
    let d = pmf[k];
    let center = n as f64 * 0.5;
    let rel = 1.0 + 1e-7;
    let cdf = |upto: isize| -> f64 { if upto < 0 { 0.0 } else { pmf[..=(upto as usize).min(n)].iter().sum() } };
    let sf = |above: isize| -> f64 {
        // P(X > above)
        if above >= n as isize { 0.0 } else { pmf[(above + 1).max(0) as usize..].iter().sum() }
    };
    if (k as f64 - center).abs() < f64::EPSILON {
        return 1.0;
    }
    let p = if (k as f64) < center {
        let lo = center.ceil() as usize;
        let i = (lo..=n).filter(|&y| pmf[y] <= d * rel).count() as isize;
        cdf(k as isize) + sf(n as isize - i)
    } else {
        let hi = center.floor() as usize;
        let i = (0..=hi).filter(|&y| pmf[y] <= d * rel).count() as isize;
        cdf(i - 1) + sf(k as isize - 1)
    };
    p.min(1.0)
}

fn ln_gamma(x: f64) -> f64 {
    // Lanczos (g = 7, n = 9)
    const C: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        return (std::f64::consts::PI / (std::f64::consts::PI * x).sin()).ln() - ln_gamma(1.0 - x);
    }
    let x = x - 1.0;
    let mut a = C[0];
    let t = x + 7.5;
    for (i, c) in C.iter().enumerate().skip(1) {
        a += c / (x + i as f64);
    }
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
}

fn betacf(a: f64, b: f64, x: f64) -> f64 {
    let (qab, qap, qam) = (a + b, a + 1.0, a - 1.0);
    let tiny = 1e-300;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < tiny {
        d = tiny;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..500 {
        let m = m as f64;
        let m2 = 2.0 * m;
        let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < tiny {
            d = tiny;
        }
        c = 1.0 + aa / c;
        if c.abs() < tiny {
            c = tiny;
        }
        d = 1.0 / d;
        h *= d * c;
        let aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < tiny {
            d = tiny;
        }
        c = 1.0 + aa / c;
        if c.abs() < tiny {
            c = tiny;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < 1e-15 {
            break;
        }
    }
    h
}

/// Regularized incomplete beta `I_x(a, b)`.
pub fn betainc(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let bt = (ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b) + a * x.ln() + b * (1.0 - x).ln()).exp();
    if x < (a + 1.0) / (a + b + 2.0) { bt * betacf(a, b, x) / a } else { 1.0 - bt * betacf(b, a, 1.0 - x) / b }
}

/// Two-sided p of a Student t statistic with `df` degrees of freedom.
pub fn t_two_sided(t: f64, df: f64) -> f64 {
    betainc(df / 2.0, 0.5, df / (df + t * t))
}

/// Complementary error function (series for small |x|, continued fraction beyond).
pub fn erfc(x: f64) -> f64 {
    let z = x.abs();
    let r = if z < 2.5 {
        // erf series
        let mut sum = z;
        let mut term = z;
        for n in 1..200 {
            term *= -z * z / n as f64;
            let add = term / (2 * n + 1) as f64;
            sum += add;
            if add.abs() < 1e-17 * sum.abs() {
                break;
            }
        }
        1.0 - 2.0 / std::f64::consts::PI.sqrt() * sum
    } else {
        // Lentz continued fraction for erfc
        let mut f = 0.0;
        for k in (1..80).rev() {
            f = k as f64 / 2.0 / (z + f);
        }
        (-z * z).exp() / std::f64::consts::PI.sqrt() / (z + f)
    };
    if x < 0.0 { 2.0 - r } else { r }
}

/// Standard normal CDF.
pub fn norm_cdf(z: f64) -> f64 {
    0.5 * erfc(-z / std::f64::consts::SQRT_2)
}

/// Average ranks (1-based, ties share the mean rank) and the sizes of the tie groups.
pub fn rank_avg(x: &[f64]) -> (Vec<f64>, Vec<usize>) {
    let mut idx: Vec<usize> = (0..x.len()).collect();
    idx.sort_by(|&a, &b| x[a].total_cmp(&x[b]));
    let mut ranks = vec![0.0; x.len()];
    let mut ties = Vec::new();
    let mut i = 0;
    while i < idx.len() {
        let mut j = i;
        while j + 1 < idx.len() && x[idx[j + 1]] == x[idx[i]] {
            j += 1;
        }
        let r = (i + j) as f64 / 2.0 + 1.0;
        for &k in &idx[i..=j] {
            ranks[k] = r;
        }
        ties.push(j - i + 1);
        i = j + 1;
    }
    (ranks, ties)
}

/// `scipy.stats.spearmanr(x, y)`: `(rho, two-sided p)`; NaN when either input is constant or n < 3.
pub fn spearman(x: &[f64], y: &[f64]) -> (f64, f64) {
    let n = x.len();
    if n < 2 || y.len() != n {
        return (f64::NAN, f64::NAN);
    }
    let (rx, _) = rank_avg(x);
    let (ry, _) = rank_avg(y);
    let (mx, my) = (rx.iter().sum::<f64>() / n as f64, ry.iter().sum::<f64>() / n as f64);
    let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
    for i in 0..n {
        sxy += (rx[i] - mx) * (ry[i] - my);
        sxx += (rx[i] - mx).powi(2);
        syy += (ry[i] - my).powi(2);
    }
    if sxx == 0.0 || syy == 0.0 {
        return (f64::NAN, f64::NAN);
    }
    let rho = (sxy / (sxx * syy).sqrt()).clamp(-1.0, 1.0);
    if n < 3 {
        return (rho, f64::NAN);
    }
    if rho.abs() == 1.0 {
        return (rho, 0.0);
    }
    let t = rho * ((n as f64 - 2.0) / (1.0 - rho * rho)).sqrt();
    (rho, t_two_sided(t, n as f64 - 2.0))
}

/// `scipy.stats.wilcoxon(d, alternative="less", zero_method="wilcox").pvalue`; 1.0 when every difference is zero.
pub fn wilcoxon_less(d: &[f64]) -> f64 {
    let zeros = d.iter().filter(|v| **v == 0.0).count();
    let nz: Vec<f64> = d.iter().copied().filter(|v| *v != 0.0).collect();
    let n = nz.len();
    if n == 0 {
        return 1.0;
    }
    let abs: Vec<f64> = nz.iter().map(|v| v.abs()).collect();
    let (ranks, ties) = rank_avg(&abs);
    let r_plus: f64 = nz.iter().zip(&ranks).filter(|(v, _)| **v > 0.0).map(|(_, r)| r).sum();
    let clean = zeros == 0 && ties.iter().all(|&t| t == 1);
    // scipy 1.18: exact for n <= 50 when nothing is tied or zero; with ties or zeros the sign-flip enumeration only while it is small
    // (observed exact at 5 and 11 non-zero differences, normal approximation at 23: the cut-off 13 is 2^n <= 10,000, not verified)
    if (clean && n <= 50) || (!clean && n <= 13) {
        // scipy's exact method: the distribution of the sum over every sign assignment of the (average) ranks; doubled ranks are integers
        let w: Vec<usize> = ranks.iter().map(|r| (r * 2.0).round() as usize).collect();
        let max: usize = w.iter().sum();
        let mut c = vec![0.0f64; max + 1];
        c[0] = 1.0;
        for &x in &w {
            for s in (x..=max).rev() {
                c[s] += c[s - x];
            }
        }
        let total: f64 = c.iter().sum();
        let upto = ((r_plus * 2.0).round() as usize).min(max);
        return c[..=upto].iter().sum::<f64>() / total;
    }
    let nf = n as f64;
    let mn = nf * (nf + 1.0) / 4.0;
    let tie_term: f64 = ties.iter().map(|&t| (t as f64).powi(3) - t as f64).sum::<f64>() / 48.0;
    let se = (nf * (nf + 1.0) * (2.0 * nf + 1.0) / 24.0 - tie_term).sqrt();
    norm_cdf((r_plus - mn) / se)
}

/// Holm step-down adjusted p-values (`tournament.holm`).
pub fn holm(ps: &[f64]) -> Vec<f64> {
    let mut order: Vec<usize> = (0..ps.len()).collect();
    order.sort_by(|&a, &b| ps[a].total_cmp(&ps[b]));
    let mut out = vec![0.0; ps.len()];
    let mut run = 0.0f64;
    for (rank, &i) in order.iter().enumerate() {
        run = run.max(((ps.len() - rank) as f64 * ps[i]).min(1.0));
        out[i] = run;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normal_cdf_and_t_are_right_at_known_points() {
        assert!((norm_cdf(0.0) - 0.5).abs() < 1e-14);
        assert!((norm_cdf(1.959_963_984_540_054) - 0.975).abs() < 1e-9);
        assert!((t_two_sided(2.228_138_851_986_274, 10.0) - 0.05).abs() < 1e-7);
    }

    #[test]
    fn holm_is_monotone_and_capped() {
        let h = holm(&[0.01, 0.04, 0.03]);
        assert_eq!(h, vec![0.03, 0.06, 0.06]);
        assert_eq!(holm(&[0.6, 0.9]), vec![1.0, 0.9_f64.max(1.0)]);
    }
}

// --- Lab result statistics (`lab/labstats.py`) ---------------------------------------------------------------------------------------
/// Student t CDF.
pub fn t_cdf(t: f64, df: f64) -> f64 {
    let p = 0.5 * betainc(df / 2.0, 0.5, df / (df + t * t));
    if t > 0.0 { 1.0 - p } else { p }
}

/// `scipy.stats.t.ppf(q, df)` by bisection on the CDF.
pub fn t_ppf(q: f64, df: f64) -> f64 {
    let (mut lo, mut hi) = (-1000.0, 1000.0);
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if t_cdf(mid, df) < q {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MeanCi {
    pub mean: f64,
    pub sd: f64,
    pub n: usize,
    pub lo: f64,
    pub hi: f64,
}

/// `labstats.mean_ci`: mean, SD, n and a t-distribution 95 % interval (NaN-safe; n < 2 gives no interval).
pub fn mean_ci(values: &[f64]) -> MeanCi {
    let v: Vec<f64> = values.iter().copied().filter(|x| !x.is_nan()).collect();
    let n = v.len();
    if n == 0 {
        return MeanCi { mean: f64::NAN, sd: f64::NAN, n: 0, lo: f64::NAN, hi: f64::NAN };
    }
    let m = crate::validation::mean(&v);
    if n < 2 {
        return MeanCi { mean: m, sd: f64::NAN, n: 1, lo: f64::NAN, hi: f64::NAN };
    }
    let sd = crate::validation::sd(&v);
    let half = t_ppf(0.975, (n - 1) as f64) * sd / (n as f64).sqrt();
    MeanCi { mean: m, sd, n, lo: m - half, hi: m + half }
}

/// Two-sided paired Wilcoxon signed-rank p (`zero_method="wilcox"`).
pub fn wilcoxon_two_sided(d: &[f64]) -> f64 {
    let nz: Vec<f64> = d.iter().copied().filter(|v| *v != 0.0).collect();
    let n = nz.len();
    if n == 0 {
        return 1.0;
    }
    let abs: Vec<f64> = nz.iter().map(|v| v.abs()).collect();
    let (ranks, ties) = rank_avg(&abs);
    let r_plus: f64 = nz.iter().zip(&ranks).filter(|(v, _)| **v > 0.0).map(|(_, r)| r).sum();
    let total_rank = n as f64 * (n as f64 + 1.0) / 2.0;
    let r_min = r_plus.min(total_rank - r_plus);
    let clean = ties.iter().all(|&t| t == 1) && d.iter().all(|v| *v != 0.0);
    if (clean && n <= 50) || (!clean && n <= 13) {
        let w: Vec<usize> = ranks.iter().map(|r| (r * 2.0).round() as usize).collect();
        let max: usize = w.iter().sum();
        let mut c = vec![0.0f64; max + 1];
        c[0] = 1.0;
        for &x in &w {
            for s in (x..=max).rev() {
                c[s] += c[s - x];
            }
        }
        let tot: f64 = c.iter().sum();
        let upto = ((r_min * 2.0).round() as usize).min(max);
        return (2.0 * c[..=upto].iter().sum::<f64>() / tot).min(1.0);
    }
    let nf = n as f64;
    let mn = nf * (nf + 1.0) / 4.0;
    let tie_term: f64 = ties.iter().map(|&t| (t as f64).powi(3) - t as f64).sum::<f64>() / 48.0;
    let se = (nf * (nf + 1.0) * (2.0 * nf + 1.0) / 24.0 - tie_term).sqrt();
    (2.0 * norm_cdf((r_min - mn) / se)).min(1.0)
}

/// `scipy.stats.ttest_rel(a, b).pvalue` (two-sided); NaN when the differences have no spread.
pub fn ttest_rel_p(a: &[f64], b: &[f64]) -> f64 {
    let d: Vec<f64> = a.iter().zip(b).map(|(x, y)| x - y).collect();
    let n = d.len();
    if n < 2 {
        return f64::NAN;
    }
    let (m, s) = (crate::validation::mean(&d), crate::validation::sd(&d));
    if s == 0.0 {
        return f64::NAN;
    }
    t_two_sided(m / (s / (n as f64).sqrt()), (n - 1) as f64)
}

/// `scipy.stats.fisher_exact([[yes_a, n_a - yes_a], [yes_b, n_b - yes_b]]).pvalue` (two-sided).
pub fn fisher_exact(yes_a: u64, n_a: u64, yes_b: u64, n_b: u64) -> f64 {
    let (k_total, n_total, row) = (yes_a + yes_b, n_a + n_b, n_a);
    // hypergeometric: P(X = x) with the first row's size n_a, successes k_total, population n_total
    let lf = |n: u64| -> f64 { (1..=n).map(|i| (i as f64).ln()).sum() };
    let ln_choose = |n: u64, k: u64| lf(n) - lf(k) - lf(n - k);
    let pmf = |x: u64| -> f64 { (ln_choose(k_total, x) + ln_choose(n_total - k_total, row - x) - ln_choose(n_total, row)).exp() };
    let lo = row.saturating_sub(n_total - k_total);
    let hi = k_total.min(row);
    let p_obs = pmf(yes_a);
    let rel = 1.0 + 1e-7;
    (lo..=hi).map(pmf).filter(|&p| p <= p_obs * rel).sum::<f64>().min(1.0)
}
