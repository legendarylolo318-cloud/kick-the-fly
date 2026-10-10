//! Hemifield and unilateral lesions (`lab/lesions.py`): silencing one hemisphere's visual projection neurons or a whole hemisphere.

use crate::pack::Labels;

const VISUAL_PREFIXES: [&str; 5] = ["LC", "LPLC", "LPTC", "VS", "HS"];

/// `np.nanmedian(soma[:, 0])` of a float32 column.
fn midline(labels: &Labels) -> Option<f32> {
    let mut x: Vec<f32> = labels.soma.iter().map(|s| s[0]).filter(|v| !v.is_nan()).collect();
    if x.is_empty() {
        return None;
    }
    x.sort_by(|a, b| a.total_cmp(b));
    let n = x.len();
    Some(if n % 2 == 1 { x[n / 2] } else { (x[n / 2 - 1] + x[n / 2]) / 2.0 })
}

#[allow(clippy::needless_range_loop)]
fn side_flags(labels: &Labels, side: char) -> Vec<bool> {
    let target = format!("_{}", side.to_ascii_uppercase());
    let n = labels.types.len();
    let inst = |i: usize| labels.instance.get(i).map_or("", String::as_str);
    let mut m: Vec<bool> = (0..n).map(|i| inst(i).ends_with(&target)).collect();
    if let (Some(mid), false) = (midline(labels), labels.soma.is_empty()) {
        for i in 0..n {
            let has_any = inst(i).ends_with("_L") || inst(i).ends_with("_R");
            let x = labels.soma[i][0];
            if !has_any && !x.is_nan() {
                m[i] |= if side.eq_ignore_ascii_case(&'L') { x > mid } else { x <= mid };
            }
        }
    }
    m
}

/// Rows of one hemisphere's visual cell types (LC10, LPLC2, LC4, LPTC/VS/HS, ...).
pub fn hemifield_visual_rows(labels: &Labels, side: char) -> Vec<u32> {
    let side_match = side_flags(labels, side);
    (0..labels.types.len())
        .filter(|&i| VISUAL_PREFIXES.iter().any(|p| labels.types[i].starts_with(p)) && side_match[i])
        .map(|i| i as u32)
        .collect()
}

/// Rows of a whole hemisphere (all left or all right neurons).
pub fn hemisphere_rows(labels: &Labels, side: char) -> Vec<u32> {
    let m = side_flags(labels, side);
    (0..m.len()).filter(|&i| m[i]).map(|i| i as u32).collect()
}
