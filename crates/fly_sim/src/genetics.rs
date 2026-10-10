//! Genetic toolkit (`lab/genetics.py`): choose neurons by split-GAL4 driver line. The line -> cell-type table is the literature's
//! (Meissner et al. 2025, eLife 13:RP98405, CC BY 4.0); a cell-type name matches the connectome only when spelled identically.

use std::collections::{BTreeMap, HashMap};
use std::sync::OnceLock;

use serde_json::{Value, json};

use crate::pack::Labels;

#[derive(Debug, Clone)]
pub struct Line {
    pub name: String,
    pub quality: String,
    pub sex_difference: String,
    pub vnc_only: String,
    pub brain_only: String,
    pub cell_types: Vec<String>,
    pub doi: String,
    pub cite: String,
}

struct Table {
    source: Value,
    /// In file order (the order Python's dict iterates).
    lines: Vec<Line>,
    by_name: HashMap<String, usize>,
}

fn table() -> &'static Table {
    static T: OnceLock<Table> = OnceLock::new();
    T.get_or_init(|| {
        let doc: Value = serde_json::from_str(include_str!("../data/driver_lines.json")).expect("driver_lines.json is valid");
        let s = |v: &Value| v.as_str().unwrap_or("").to_string();
        let lines: Vec<Line> = doc["lines"]
            .as_array()
            .into_iter()
            .flatten()
            .map(|d| Line {
                name: s(&d["line"]),
                quality: s(&d["quality"]),
                sex_difference: s(&d["sex_difference"]),
                vnc_only: s(&d["vnc_only"]),
                brain_only: s(&d["brain_only"]),
                cell_types: d["cell_types"].as_array().into_iter().flatten().map(&s).collect(),
                doi: s(&d["doi"]),
                cite: s(&d["cite"]),
            })
            .collect();
        let by_name = lines.iter().enumerate().map(|(i, l)| (l.name.clone(), i)).collect();
        Table { source: doc["source"].clone(), lines, by_name }
    })
}

pub fn source() -> &'static Value {
    &table().source
}

pub fn lines() -> &'static [Line] {
    &table().lines
}

/// `get`: exact name, else case-insensitive.
pub fn get(name: &str) -> Result<&'static Line, String> {
    let t = table();
    let key = name.trim();
    if let Some(&i) = t.by_name.get(key) {
        return Ok(&t.lines[i]);
    }
    let low = key.to_lowercase();
    // a dict of lower-cased names: the last of equal lower-cased names wins
    if let Some(l) = t.lines.iter().rev().find(|l| l.name.to_lowercase() == low) {
        return Ok(l);
    }
    Err(format!(
        "unknown driver line {name:?} (the table lists {} adult split-GAL4 lines; Lab > Genetic toolkit searches them)",
        crate::netsci::group(t.lines.len() as f64)
    ))
}

/// `search`: lines whose name or one of whose cell types contains the text, best first.
pub fn search(query: &str, limit: usize) -> Vec<&'static Line> {
    let q = query.trim().to_lowercase();
    if q.is_empty() {
        return Vec::new();
    }
    let (mut exact, mut prefix, mut other) = (Vec::new(), Vec::new(), Vec::new());
    for l in lines() {
        let n = l.name.to_lowercase();
        if n == q {
            exact.push(l);
        } else if n.starts_with(&q) {
            prefix.push(l);
        } else if l.cell_types.iter().any(|t| t.to_lowercase().contains(&q)) {
            other.push(l);
        }
    }
    exact.into_iter().chain(prefix).chain(other).take(limit).collect()
}

/// `{MaleCNS type: neuron count}`.
pub fn type_counts(labels: &Labels) -> BTreeMap<String, usize> {
    let mut m = BTreeMap::new();
    for t in &labels.types {
        *m.entry(t.clone()).or_insert(0) += 1;
    }
    m
}

/// One line against one connectome.
pub struct Matched {
    pub line: &'static Line,
    pub matched: Vec<(String, usize)>,
    pub unmatched: Vec<String>,
    pub rows: Vec<u32>,
}

pub fn match_line(line: &'static Line, labels: &Labels, counts: &BTreeMap<String, usize>) -> Matched {
    let (mut matched, mut unmatched) = (Vec::new(), Vec::new());
    for t in &line.cell_types {
        match counts.get(t) {
            Some(&n) => matched.push((t.clone(), n)),
            None => unmatched.push(t.clone()),
        }
    }
    let rows: Vec<u32> = if matched.is_empty() {
        Vec::new()
    } else {
        (0..labels.types.len() as u32).filter(|&i| matched.iter().any(|(t, _)| *t == labels.types[i as usize])).collect()
    };
    Matched { line, matched, unmatched, rows }
}

/// `rows_of_lines`: the rows of a `line:A,B` spec.
pub fn rows_of_lines(labels: &Labels, spec: &str) -> Result<Vec<u32>, String> {
    let names: Vec<&str> = spec.split_once(':').map_or("", |x| x.1).split(',').map(str::trim).filter(|s| !s.is_empty()).collect();
    if names.is_empty() {
        return Err("line: needs at least one driver line name, for example line:SS00727".into());
    }
    let counts = type_counts(labels);
    let (mut rows, mut problems) = (Vec::new(), Vec::new());
    for n in names {
        let m = match_line(get(n)?, labels, &counts);
        if m.rows.is_empty() {
            let types = if m.line.cell_types.is_empty() { "no listed cell type".to_string() } else { m.line.cell_types.join(", ") };
            problems.push(format!(
                "{} labels {types}; none of those is spelled like a cell type in this connectome, so it selects no neurons",
                m.line.name
            ));
        }
        rows.extend(m.rows);
    }
    if !problems.is_empty() {
        return Err(problems.join("; "));
    }
    rows.sort_unstable();
    rows.dedup();
    Ok(rows)
}

fn quality_text(q: &str) -> Option<&'static str> {
    Some(match q {
        "1" => "one cell type, strongly and specifically; occasional weak expression may be seen in other cells",
        "2" => "two cell types; occasional weak expression may be seen in other cells",
        "3" => "three or more cell types",
        "4" => "specific expression but weak or variable labeling",
        "5" => "not stabilized: groups of neurons are visible but the target cell type was not labeled cleanly",
        _ => return None,
    })
}

/// `off_target`: `(level, sentence)`.
pub fn off_target(line: &Line) -> (&'static str, String) {
    let q = line.quality.trim();
    let base = quality_text(q).unwrap_or("");
    match q {
        "1" => ("minimal", format!("quality 1: {base}.")),
        "2" => {
            let extra = if line.cell_types.len() == 1 {
                " The table lists one cell type for this line, so the second labeled type is off-target."
            } else {
                " Both labeled types are the ones the table lists."
            };
            ("some", format!("quality 2: {base}.{extra}"))
        }
        "3" => ("yes", format!("quality 3: {base}; expression beyond the listed types is expected.")),
        "4" => ("weak", format!("quality 4: {base}.")),
        "5" => ("unstable", format!("quality 5: {base}.")),
        _ => (
            "unknown",
            format!(
                "quality \"{q}\" is recorded in the source table, but its meaning isn't given in the text this table was built from, so nothing is claimed about off-target expression."
            ),
        ),
    }
}

/// `describe`: everything the page shows for a line.
pub fn describe(name: &str, labels: &Labels) -> Result<Value, String> {
    let line = get(name)?;
    let counts = type_counts(labels);
    let m = match_line(line, labels, &counts);
    let (level, text) = off_target(line);
    let mut matched = serde_json::Map::new();
    for (t, n) in &m.matched {
        matched.insert(t.clone(), json!(n));
    }
    Ok(json!({
        "line": line.name, "quality": line.quality, "cell_types": line.cell_types, "matched": matched, "unmatched": m.unmatched,
        "neurons": m.rows.len(), "off_target": level, "off_target_text": text, "cite": line.cite, "doi": line.doi,
        "sex_difference": line.sex_difference, "brain_only": line.brain_only, "vnc_only": line.vnc_only,
    }))
}

/// `coverage`.
pub fn coverage(labels: &Labels) -> Value {
    let counts = type_counts(labels);
    let usable = lines().iter().filter(|l| l.cell_types.iter().any(|t| counts.contains_key(t))).count();
    let mut names: Vec<&String> = lines().iter().flat_map(|l| &l.cell_types).collect();
    names.sort();
    names.dedup();
    let hit = names.iter().filter(|t| counts.contains_key(**t)).count();
    json!({"lines": lines().len(), "lines_with_a_match": usable, "cell_type_names": names.len(), "cell_type_names_matched": hit})
}
