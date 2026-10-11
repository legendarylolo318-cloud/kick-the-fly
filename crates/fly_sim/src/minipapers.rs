//! Guided mini-papers (`lab/minipapers.py`): short experiments that reproduce a classic paper step by step.
//!
//! Nothing is re-implemented: every adult experiment is a validation test run through [`crate::validation::run`], so its numbers are the
//! validation suite's own. The paper texts are data (`data/minipapers.json`, written from the Python table by `tests/golden/make_minipapers.py`).
//! "What the paper found" is LITERATURE, "your result" is a MODEL PREDICTION. The Buridan rig paper needs the rig assays, which are not
//! ported yet; the larva pair needs the larva pack (absent) and shows the recorded validation numbers instead, as Python does.

use std::sync::{Arc, OnceLock};

use serde_json::{Value, json};

use crate::csr::Weights;
use crate::pack::Labels;
use crate::sensitivity::fmt_gp;

pub const QUICK_SEEDS: [u64; 4] = [0, 1, 2, 3];
pub const FULL_SEEDS: [u64; 10] = [1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009];
/// `validation.RATIO_MIN, P_MAX, PI_MIN, CONTROL_PI_MAX`.
pub const RATIO_MIN: f64 = 1.5;
pub const P_MAX: f64 = 0.01;
pub const PI_MIN: f64 = 0.5;
pub const CONTROL_PI_MAX: f64 = 0.25;

#[derive(Debug, Clone)]
pub struct Question {
    pub test: String,
    pub prompt: String,
    /// `(key, label)`; the first is the direction the paper reports.
    pub options: Vec<(String, String)>,
    pub expected: String,
    pub pass_note: String,
    pub fail_why: String,
}

#[derive(Debug, Clone)]
pub struct Paper {
    pub id: String,
    pub title: String,
    pub short: String,
    pub citation: String,
    pub doi: String,
    pub read: String,
    pub summary: String,
    pub found: String,
    pub found_numbers: String,
    pub cannot_check: String,
    pub questions: Vec<Question>,
    pub brain: String,
    /// Recorded validation numbers, used when the brain pack is missing (larva): test id -> measured dict.
    pub recorded: Value,
    /// `(title, text, kind)`; kinds: read, hypothesis, run, plot, compare.
    pub steps: Vec<(String, String, String)>,
}

fn s(v: &Value) -> String {
    v.as_str().unwrap_or("").to_string()
}

/// The papers in `ORDER`.
pub fn papers() -> &'static [Paper] {
    static P: OnceLock<Vec<Paper>> = OnceLock::new();
    P.get_or_init(|| {
        let v: Value = serde_json::from_str(include_str!("../data/minipapers.json")).expect("minipapers.json");
        v["papers"]
            .as_array()
            .into_iter()
            .flatten()
            .map(|p| Paper {
                id: s(&p["id"]),
                title: s(&p["title"]),
                short: s(&p["short"]),
                citation: s(&p["citation"]),
                doi: s(&p["doi"]),
                read: s(&p["read"]),
                summary: s(&p["summary"]),
                found: s(&p["found"]),
                found_numbers: s(&p["found_numbers"]),
                cannot_check: s(&p["cannot_check"]),
                brain: s(&p["brain"]),
                recorded: p["recorded"].clone(),
                questions: p["questions"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .map(|q| Question {
                        test: s(&q["test"]),
                        prompt: s(&q["prompt"]),
                        options: q["options"].as_array().into_iter().flatten().map(|o| (s(&o[0]), s(&o[1]))).collect(),
                        expected: s(&q["expected"]),
                        pass_note: s(&q["pass_note"]),
                        fail_why: s(&q["fail_why"]),
                    })
                    .collect(),
                steps: p["steps"].as_array().into_iter().flatten().map(|t| (s(&t[0]), s(&t[1]), s(&t[2]))).collect(),
            })
            .collect()
    })
}

pub fn paper(id: &str) -> Option<&'static Paper> {
    papers().iter().find(|p| p.id == id)
}

/// A short note of how much of the paper was read, for the list (`read_label`).
pub fn read_label(p: &Paper) -> &'static str {
    let r = p.read.to_lowercase();
    if r.starts_with("full text") {
        "full text"
    } else if r.contains("full text") && r.contains("searched") {
        "abstract, full text searched"
    } else {
        "abstract only"
    }
}

/// `available`: whether the paper can run live here, and if not why not.
pub fn available(p: &Paper, adult_pack: bool) -> (bool, String) {
    if p.questions.first().is_some_and(|q| q.test.starts_with("rig:")) {
        return (false, "the rig assays (Lab > Rigs) are not part of this build yet".into());
    }
    if p.brain == "larva" {
        return (
            false,
            "the larva brain pack is not built (python -m kickthefly.sim.connectome.larva_loader build downloads Data S1 once; docs/larva.md); \
the recorded validation numbers are shown instead"
                .into(),
        );
    }
    if !adult_pack {
        return (false, "the adult brain pack is not built (python -m kickthefly.sim.brainpack build)".into());
    }
    (true, String::new())
}

/// The order Python's dicts had (serde sorts keys): the T-maze numbers, then the pathway numbers, other keys, `p_value`, `n`, `passed`.
pub fn ordered(m: &Value) -> Vec<(String, Value)> {
    const HEAD: [&str; 14] = [
        "pi_mean",
        "pi_sd",
        "control_pi_mean",
        "control_pi_sd",
        "fear_cs_plus",
        "fear_cs_minus",
        "approach_mbon_cs_plus_hz",
        "approach_mbon_cs_minus_hz",
        "drive_ratio_mean",
        "drive_ratio_sd",
        "control_ratio_mean",
        "control_ratio_sd",
        "readout_base_hz",
        "readout_driven_hz",
    ];
    const TAIL: [&str; 3] = ["p_value", "n", "passed"];
    let Some(o) = m.as_object() else { return Vec::new() };
    let mut out: Vec<(String, Value)> = Vec::new();
    for k in HEAD {
        if let Some(v) = o.get(k) {
            out.push((k.to_string(), v.clone()));
        }
    }
    for (k, v) in o {
        if !HEAD.contains(&k.as_str()) && !TAIL.contains(&k.as_str()) {
            out.push((k.clone(), v.clone()));
        }
    }
    for k in TAIL {
        if let Some(v) = o.get(k) {
            out.push((k.to_string(), v.clone()));
        }
    }
    out
}

fn ratio_pairs(t: &Value) -> Vec<Value> {
    t["per_seed"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|sd| {
            if sd.get("drive").is_some() {
                let num = |v: &Value| v["ratio"].as_f64().or_else(|| v.as_f64()).unwrap_or(f64::NAN);
                json!({"seed": sd["seed"], "drive": num(&sd["drive"]), "control": num(&sd["control"])})
            } else {
                json!({"seed": sd["seed"], "drive": sd["pi"], "control": sd["control_pi"]})
            }
        })
        .collect()
}

/// `_verdict`: reproduced or not, by validation's criteria on a full run; on a quick run by effect and direction only.
fn verdict(test_id: &str, t: &Value, full: bool) -> Value {
    let m = &t["measured"];
    let f = |k: &str| m[k].as_f64().unwrap_or(f64::NAN);
    let eff = if test_id == "mb_conditioning" {
        f("pi_mean") >= PI_MIN && f("control_pi_mean").abs() <= CONTROL_PI_MAX && f("pi_mean") > f("control_pi_mean")
    } else {
        f("drive_ratio_mean") >= RATIO_MIN && f("drive_ratio_mean") > f("control_ratio_mean")
    };
    let n = m["n"].as_i64().unwrap_or_else(|| t["per_seed"].as_array().map_or(0, |a| a.len() as i64));
    let p = m["p_value"].as_f64();
    let (ok, basis) = if full {
        (
            t["passed"].as_bool().unwrap_or(eff && p.is_some_and(|p| p < P_MAX)),
            "validation's criteria on the held-out seeds (n = 10, p < 0.01)".to_string(),
        )
    } else {
        (
            eff,
            format!(
                "effect size and direction only: n = {n} is too few flies for the validation's p < 0.01 (the smallest one-sided Wilcoxon p at n = {n} is {})",
                fmt_gp(1.0 / 2f64.powi(n as i32), 3)
            ),
        )
    };
    json!({"reproduced": ok, "basis": basis, "measured": m, "n": n})
}

/// `run_paper`: run a mini-paper's experiment(s). `shared` is the loaded adult brain (None: not available). A paper that cannot run live
/// returns the recorded numbers when it has them, marked `source = "recorded"`. `progress(done, total, label)` returns false to cancel.
pub fn run_paper(
    paper_id: &str,
    seeds: &[u64],
    shared: Option<(&Arc<Weights>, &Arc<Labels>)>,
    workers: usize,
    progress: impl FnMut(usize, usize, &str) -> bool,
    created: &str,
) -> Result<Value, String> {
    let p = paper(paper_id).ok_or_else(|| format!("unknown mini-paper {paper_id:?}"))?;
    let mut full = seeds == FULL_SEEDS;
    let t0 = std::time::Instant::now();
    let (ok, why) = available(p, shared.is_some());
    let mut got: std::collections::BTreeMap<String, Value> = Default::default();
    let mut source = "live";
    let has_recorded = p.recorded.as_object().is_some_and(|o| !o.is_empty());
    if !ok && !has_recorded {
        return Err(why);
    } else if !ok {
        source = "recorded";
        for q in &p.questions {
            let rec = p.recorded[&q.test].clone();
            got.insert(q.test.clone(), json!({"measured": rec, "passed": rec["passed"], "per_seed": [], "id": q.test}));
        }
        full = true; // the recorded numbers are the held-out run's
    } else if let Some((w, l)) = shared {
        let include: Vec<&str> = p.questions.iter().map(|q| q.test.as_str()).collect();
        let res = crate::validation::run(w, l, seeds, Some(&include), workers.max(1), progress)?;
        for t in res["tests"].as_array().into_iter().flatten() {
            if let Some(id) = t["id"].as_str().filter(|id| include.contains(id)) {
                got.insert(id.to_string(), t.clone());
            }
        }
    }
    let qs: Vec<Value> = p
        .questions
        .iter()
        .map(|q| {
            let t = got.get(&q.test).cloned().unwrap_or(Value::Null);
            let v = verdict(&q.test, &t, full);
            let reproduced = v["reproduced"].as_bool().unwrap_or(false);
            json!({"test": q.test, "prompt": q.prompt, "expected": q.expected, "verdict": v,
                   "pairs": if t.get("per_seed").is_some() { ratio_pairs(&t) } else { Vec::new() },
                   "note": if reproduced { &q.pass_note } else { &q.fail_why }})
        })
        .collect();
    Ok(json!({
        "kind": "minipaper", "paper": paper_id, "title": p.title, "citation": p.citation, "doi": p.doi, "read": p.read, "seeds": seeds, "full": full,
        "source": source, "unavailable": why, "seconds": (t0.elapsed().as_secs_f64() * 10.0).round() / 10.0, "questions": qs, "created": created,
    }))
}

/// `compare`: per question, your hypothesis, the paper's direction, the model's result and whether they agree.
pub fn compare(res: &Value, answers: &std::collections::BTreeMap<String, String>) -> Vec<Value> {
    let Some(p) = paper(res["paper"].as_str().unwrap_or("")) else { return Vec::new() };
    p.questions
        .iter()
        .zip(res["questions"].as_array().into_iter().flatten())
        .map(|(q, r)| {
            let mine = answers.get(&q.test);
            let label = |k: &str| q.options.iter().find(|o| o.0 == k).map(|o| o.1.clone());
            let reproduced = r["verdict"]["reproduced"].as_bool().unwrap_or(false);
            json!({
                "test": q.test, "question": q.prompt, "your_hypothesis": mine.and_then(|m| label(m)),
                "paper_direction": label(&q.expected), "model_reproduces_paper": reproduced,
                "you_matched_the_paper": mine.map(|m| *m == q.expected), "you_matched_the_model": mine.map(|m| (*m == q.expected) == reproduced),
                "basis": r["verdict"]["basis"], "note": r["note"], "measured": r["verdict"]["measured"],
            })
        })
        .collect()
}

/// A number as `f"{v:.3g}"`, anything else as `str(v)`.
fn num3(v: &Value) -> String {
    match v {
        Value::Number(n) if n.is_f64() => fmt_gp(n.as_f64().unwrap_or(0.0), 3),
        Value::Number(n) => n.to_string(),
        Value::Bool(b) => if *b { "True" } else { "False" }.into(),
        Value::Null => "None".into(),
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// `render`: the text of the comparison.
pub fn render(res: &Value, answers: &std::collections::BTreeMap<String, String>) -> String {
    let Some(p) = paper(res["paper"].as_str().unwrap_or("")) else { return String::new() };
    let seeds: Vec<i64> = res["seeds"].as_array().into_iter().flatten().filter_map(Value::as_i64).collect();
    let mut lines = vec![
        format!("MINI-PAPER: {}", p.title),
        format!("  {}", p.citation),
        format!("  doi:{}   read: {}", p.doi, p.read),
        String::new(),
        format!("  The paper reports: {}", p.found),
        format!("  Numbers the paper states: {}", p.found_numbers),
        String::new(),
    ];
    let src = if res["source"].as_str() == Some("recorded") {
        format!("RECORDED validation numbers (not run now): {}", res["unavailable"].as_str().unwrap_or(""))
    } else {
        format!(
            "RUN NOW on {} flies (seeds {}-{}{})",
            seeds.len(),
            seeds.first().copied().unwrap_or(0),
            seeds.last().copied().unwrap_or(0),
            if res["full"].as_bool().unwrap_or(false) { "" } else { ", exploration seeds: a first look" }
        )
    };
    lines.push(format!("  Your result: {src}"));
    for row in compare(res, answers) {
        let nums: Vec<String> =
            ordered(&row["measured"]).iter().filter(|(k, _)| k != "passed").map(|(k, v)| format!("{k} {}", num3(v))).collect();
        lines.push(String::new());
        lines.push(format!("  Q ({}): {}", row["test"].as_str().unwrap_or(""), row["question"].as_str().unwrap_or("")));
        lines.push(format!("    the direction the paper's finding points to: {}", row["paper_direction"].as_str().unwrap_or("None")));
        lines.push(format!("    your hypothesis: {}", row["your_hypothesis"].as_str().unwrap_or("not given")));
        lines.push(format!(
            "    the model {} that ({})",
            if row["model_reproduces_paper"].as_bool().unwrap_or(false) { "REPRODUCES" } else { "DOES NOT REPRODUCE" },
            row["basis"].as_str().unwrap_or("")
        ));
        lines.push(format!("    numbers: {}", nums.join(", ")));
        lines.push(format!("    {}", row["note"].as_str().unwrap_or("")));
    }
    lines.push(String::new());
    lines.push(format!("  What this model cannot check: {}", p.cannot_check));
    lines.join("\n")
}

/// `svg_pairs`: a paired dot plot per question, plain SVG.
pub fn svg_pairs(res: &Value, width: i32, row_h: i32) -> String {
    let qs: Vec<&Value> =
        res["questions"].as_array().into_iter().flatten().filter(|q| q["pairs"].as_array().is_some_and(|a| !a.is_empty())).collect();
    let h = row_h * qs.len().max(1) as i32 + 20;
    let mut out = vec![format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{h}\" font-family=\"sans-serif\" font-size=\"11\">"
    )];
    for (i, q) in qs.iter().enumerate() {
        let top = 10 + i as i32 * row_h;
        let pairs = q["pairs"].as_array().cloned().unwrap_or_default();
        let vals: Vec<f64> =
            pairs.iter().flat_map(|p| [p["drive"].as_f64().unwrap_or(0.0), p["control"].as_f64().unwrap_or(0.0)]).collect();
        let mut lo = vals.iter().copied().fold(0.0f64, f64::min);
        let mut hi = vals.iter().copied().fold(0.0f64, f64::max);
        let pad = {
            let d = (hi - lo) * 0.1;
            if d == 0.0 { 1.0 } else { d }
        };
        lo = if lo < 0.0 { lo - pad } else { lo };
        hi += pad;
        let y = |v: f64| f64::from(top + row_h - 40) - (v - lo) / (hi - lo) * f64::from(row_h - 70);
        out.push(format!("<text x=\"10\" y=\"{}\">{}</text>", top + 12, q["test"].as_str().unwrap_or("")));
        out.push(format!("<line x1=\"150\" y1=\"{:.1}\" x2=\"{}\" y2=\"{:.1}\" stroke=\"#888\"/>", y(0.0), width - 20, y(0.0)));
        let xs = (260, 440);
        for pr in &pairs {
            let (d, c) = (pr["drive"].as_f64().unwrap_or(0.0), pr["control"].as_f64().unwrap_or(0.0));
            out.push(format!("<line x1=\"{}\" y1=\"{:.1}\" x2=\"{}\" y2=\"{:.1}\" stroke=\"#999\"/>", xs.0, y(c), xs.1, y(d)));
            out.push(format!(
                "<circle cx=\"{}\" cy=\"{:.1}\" r=\"4\" fill=\"#6b7a8f\"/><circle cx=\"{}\" cy=\"{:.1}\" r=\"4\" fill=\"#d9822b\"/>",
                xs.0,
                y(c),
                xs.1,
                y(d)
            ));
        }
        out.push(format!(
            "<text x=\"{}\" y=\"{}\">control</text><text x=\"{}\" y=\"{}\">drive</text>",
            xs.0 - 20,
            top + row_h - 20,
            xs.1 - 14,
            top + row_h - 20
        ));
    }
    out.push("</svg>".into());
    out.join("\n")
}

/// `save`: the JSON, the text and (when there are per-fly pairs) the SVG. The JSON's keys are sorted (Python kept insertion order).
pub fn save(
    res: &Value,
    folder: &std::path::Path,
    answers: &std::collections::BTreeMap<String, String>,
) -> std::io::Result<Vec<std::path::PathBuf>> {
    std::fs::create_dir_all(folder)?;
    let id = res["paper"].as_str().unwrap_or("");
    let mut out = Vec::new();
    let p = folder.join(format!("minipaper_{id}.json"));
    let doc = json!({"result": res, "comparison": compare(res, answers), "hypotheses": answers});
    std::fs::write(&p, serde_json::to_string_pretty(&doc).unwrap_or_default())?;
    out.push(p);
    let p = folder.join(format!("minipaper_{id}.txt"));
    std::fs::write(&p, render(res, answers) + "\n")?;
    out.push(p);
    if res["questions"].as_array().into_iter().flatten().any(|q| q["pairs"].as_array().is_some_and(|a| !a.is_empty())) {
        let p = folder.join(format!("minipaper_{id}.svg"));
        std::fs::write(&p, svg_pairs(res, 560, 160))?;
        out.push(p);
    }
    Ok(out)
}
