//! The brain autopsy (`Game._autopsy`): when the fly dies, the last two seconds of every population and detail group against its calm
//! baseline, a 23 s timeline of three summaries, and the pain index over that timeline. Computed from the history ring and the baseline
//! saved at the moment of death.

use crate::groups::{Groups, TOUCH, pain_index, population_names};

pub const PRETTY: [(&str, &str); 12] = [
    ("head", "touch: head"),
    ("body", "touch: body"),
    ("legs", "touch: legs"),
    ("wing", "touch: wings"),
    ("heat", "heat sensors"),
    ("cold", "cold sensors"),
    ("smell", "smell ORNs"),
    ("taste", "taste neurons"),
    ("reward", "PAM reward DANs"),
    ("jump", "DNs: jump group"),
    ("run", "DNs: run group"),
    ("fly", "DNs: DNg02 flight"),
];

#[derive(Debug, Clone)]
pub struct Row {
    pub name: String,
    pub base: f32,
    pub final_: f32,
    pub after: f32,
    pub ratio: f32,
    /// `pop` (whole brain and the populations) or `detail`.
    pub section: &'static str,
}

#[derive(Debug, Clone)]
pub struct Autopsy {
    pub rows: Vec<Row>,
    /// (label, activity over calm) for whole brain, touch neurons, descending.
    pub series: Vec<(String, Vec<f32>)>,
    pub death_at: usize,
    pub n: usize,
    pub top: Row,
    pub pain: Vec<f32>,
    pub pain_last: f32,
}

fn mean(v: &[f32]) -> f32 {
    if v.is_empty() { 0.0 } else { v.iter().sum::<f32>() / v.len() as f32 }
}

/// `death_sample` is the history sample count at death, `base` the calm baselines then; `pain_level` the brain's pain level.
pub fn compute(g: &Groups, death_sample: usize, base: &[f32], pain_level: usize) -> Option<Autopsy> {
    let ds = death_sample;
    let col = |n: &str| g.col(n);
    let hist = |c: usize, a: usize, b: usize| g.history_col(c, a, b);
    let mut names: Vec<String> = vec!["whole brain".into()];
    names.extend(population_names().iter().map(|s| (*s).to_string()));
    names.extend(["head", "body", "legs", "wing", "heat", "cold", "smell", "taste", "reward", "jump", "run", "fly"].map(String::from));
    let pops: Vec<&str> = population_names();
    let mut rows = vec![];
    for n in &names {
        let i = col(n)?;
        let b = base[i];
        let f = mean(&hist(i, ds.saturating_sub(100), ds));
        let a = mean(&hist(i, ds + 50, ds + 150));
        let pretty = PRETTY.iter().find(|p| p.0 == n).map_or(n.as_str(), |p| p.1);
        rows.push(Row {
            name: pretty.to_string(),
            base: b,
            final_: f,
            after: a,
            ratio: (f + 0.05) / (b + 0.05),
            section: if pops.contains(&n.as_str()) || n == "whole brain" { "pop" } else { "detail" },
        });
    }
    let tl_first = ds.saturating_sub(1000);
    let n_tl = hist(col("whole brain")?, tl_first, ds + 150).len();
    let start = tl_first.max(g.hist_n.saturating_sub(crate::groups::HIST));
    let col_series = |name: &str| -> Vec<f32> { hist(col(name).unwrap_or(0), tl_first, ds + 150) };
    let whole = col_series("whole brain");
    let desc = col_series("descending");
    let touch_cols: Vec<usize> = TOUCH.iter().filter_map(|n| col(n)).collect();
    let touch_series: Vec<Vec<f32>> = TOUCH.iter().map(|n| col_series(n)).collect();
    let sizes: Vec<f32> = TOUCH.iter().map(|n| g.size(n) as f32).collect();
    let tsz: f32 = sizes.iter().sum();
    let touch: Vec<f32> = (0..n_tl)
        .map(|k| touch_series.iter().zip(&sizes).map(|(s, z)| s.get(k).copied().unwrap_or(0.0) * z).sum::<f32>() / tsz.max(1.0))
        .collect();
    let touch_base = touch_cols.iter().zip(&sizes).map(|(&c, z)| base[c] * z).sum::<f32>() / tsz.max(1.0);
    let norm = |v: &[f32], b: f32| v.iter().map(|x| x / b.max(1e-3)).collect::<Vec<f32>>();
    let series = vec![
        ("whole brain".to_string(), norm(&whole, base[col("whole brain")?])),
        ("touch neurons".to_string(), norm(&touch, touch_base)),
        ("descending".to_string(), norm(&desc, base[col("descending")?])),
    ];
    let top = rows[1..].iter().max_by(|a, b| a.ratio.total_cmp(&b.ratio))?.clone();
    // the pain index along the timeline (`pain_parts(tl, base)`, `hold_alarm`, `pain_index`)
    let get = |n: &str| col_series(n);
    let (touch_names,): (Vec<&str>,) = ({
        let mut t: Vec<&str> = TOUCH.to_vec();
        if pain_level > 0 {
            t.push("body_extra");
        }
        t
    },);
    let tn: Vec<(Vec<f32>, f32, f32)> = touch_names.iter().filter_map(|n| col(n).map(|c| (get(n), base[c], g.size(n) as f32))).collect();
    let above = |series: &[(Vec<f32>, f32, f32)], k: usize, full: f32| -> f32 {
        let (mut num, mut den) = (0.0f32, 0.0f32);
        for (s, b, z) in series {
            num += (s.get(k).copied().unwrap_or(0.0) - b) * z;
            den += z;
        }
        if den == 0.0 { 0.0 } else { num / den / full }
    };
    let one = |n: &str| -> Vec<(Vec<f32>, f32, f32)> { col(n).map(|c| vec![(get(n), base[c], g.size(n) as f32)]).unwrap_or_default() };
    let (heat, cold, sm_ta) = (one("heat"), one("cold"), {
        let mut v = one("smell");
        v.extend(one("taste"));
        v
    });
    let (desc_b, asc) = (base[col("descending")?].max(1.0), col("ascending").map(|c| (get("ascending"), base[c].max(1.0))));
    let mut parts: Vec<[f32; 5]> = (0..n_tl)
        .map(|k| {
            let touch = above(&tn, k, 35.0);
            let thermal = above(&heat, k, 30.0).max(above(&cold, k, 30.0));
            let chemical = (above(&sm_ta, k, 1.0) - 8.0) / 27.0;
            let alarm = (desc.get(k).copied().unwrap_or(0.0) / desc_b - 1.06) / 0.3;
            let relay =
                if pain_level >= 2 { asc.as_ref().map_or(0.0, |(s, b)| (s.get(k).copied().unwrap_or(0.0) / b - 1.06) / 0.35) } else { 0.0 };
            [touch, thermal, chemical, alarm, relay].map(|v| v.clamp(0.0, 1.0))
        })
        .collect();
    for k in 1..parts.len() {
        parts[k][3] = parts[k][3].max(parts[k - 1][3] * 0.95);
    }
    let pain: Vec<f32> = parts.iter().map(pain_index).collect();
    let death_at = ds.saturating_sub(start);
    let pain_last = if pain.is_empty() { 0.0 } else { mean(&pain[death_at.saturating_sub(100).min(pain.len())..death_at.min(pain.len())]) };
    Some(Autopsy { rows, series, death_at, n: n_tl, top, pain, pain_last })
}
