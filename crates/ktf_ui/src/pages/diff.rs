//! Lab > Connectome diff mode (`lab/labdiff.py`): two flies, same seed and inputs, diffed region by region.
//!
//! Host: query `lab.diff_state` = `{job: {frac, note} | null, error, result}`; commands `lab.diff_run {scenario, mod}`, `lab.diff_export`.

use serde_json::{Value, json};

use crate::canvas::{Canvas, Rect};
use crate::menu::{Act, Anchor, Ctx, Menu, Page, Style};
use crate::theme::*;

const SCENARIOS: [(&str, &str); 4] =
    [("looming", "Looming escape"), ("sugar", "Sugar feeding"), ("antenna", "Antennal touch"), ("calm", "Calm spontaneous")];
const MODS: [(&str, &str); 7] = [
    ("lesion_loom", "Silence looming (LPLC2, LC4)"),
    ("lesion_gf", "Silence giant fiber (DNp01)"),
    ("hemi_vis", "Silence left visual hemifield"),
    ("hemi_all", "Silence left hemisphere"),
    ("thresh_10", "Threshold: drop <10 synapses"),
    ("flips", "Flip uncertain signs (50% @ 0.5)"),
    ("inhibition_0", "Picrotoxin: 100% inhibition block"),
];

fn num(v: &Value) -> f64 {
    v.as_f64().unwrap_or(f64::NAN)
}

pub struct LabDiff {
    scenario: &'static str,
    modification: &'static str,
}

impl Default for LabDiff {
    fn default() -> Self {
        Self { scenario: "looming", modification: "lesion_loom" }
    }
}

impl LabDiff {
    pub fn set_view(&mut self, scenario: &str, modification: &str) {
        if let Some(s) = SCENARIOS.iter().find(|s| s.0 == scenario) {
            self.scenario = s.0;
        }
        if let Some(s) = MODS.iter().find(|s| s.0 == modification) {
            self.modification = s.0;
        }
    }

    fn results(&self, m: &mut Menu, cv: &mut Canvas, area: Rect, res: &Value) {
        let f = m.fonts();
        let div_t = res["diverge_time_s"].as_f64();
        let div_str = div_t.map_or("no major divergence".to_string(), |t| format!("diverged at {:.0} ms", t * 1000.0));
        m.text(
            cv,
            &format!("DIVERGENCE TIMELINE: {div_str}  ·  seed {}  ·  stimulus: {}", res["seed"], res["stimulus"].as_str().unwrap_or("")),
            (area.x, area.y),
            INK,
            &f.bold,
            Anchor::TopLeft,
        );
        let tl: Vec<f64> = res["divergence_timeline"].as_array().into_iter().flatten().map(num).collect();
        let (tl_w, tl_h) = ((area.w - 20).min(680), 32);
        let r = Rect::new(area.x, area.y + 20, tl_w, tl_h);
        cv.fill_rect(r, [16, 20, 28], 6);
        cv.stroke_rect(r, [40, 48, 64], 6);
        if tl.len() > 1 {
            let max_div = tl.iter().copied().fold(1e-4, f64::max);
            let pts: Vec<(i32, i32)> = tl
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    (
                        r.x + (i as f64 * f64::from(tl_w - 4) / (tl.len() - 1) as f64) as i32 + 2,
                        r.bottom() - ((v / max_div) * f64::from(tl_h - 6)) as i32 - 3,
                    )
                })
                .collect();
            for w in pts.windows(2) {
                cv.line(w[0], w[1], 2, AMBER, 255);
            }
            if let Some(step) = res["diverge_step"].as_f64() {
                let dx = r.x + (step / (tl.len() as f64 * 5.0) * f64::from(tl_w)) as i32;
                cv.line((dx, r.y), (dx, r.bottom()), 2, BAD, 255);
                m.text(cv, "divergence", (dx + 4, r.y + 4), BAD, &f.small, Anchor::TopLeft);
            }
        }
        let mut y = r.bottom() + 14;
        m.text(
            cv,
            "Region-by-Region Activity Diff (Autopsy diverging bars: red = B > A, blue = B < A):",
            (area.x, y),
            INK,
            &f.bold,
            Anchor::TopLeft,
        );
        y += 20;
        let chart_w = area.w - 420;
        let cx0 = area.x + 190;
        let mid = cx0 + chart_w / 2;
        m.text(cv, "Neuropil Region", (area.x, y), LABEL, &f.small, Anchor::TopLeft);
        m.text(cv, "1/4x", (mid - chart_w / 4, y), LABEL, &f.small, Anchor::MidTop);
        m.text(cv, "same", (mid, y), LABEL, &f.small, Anchor::MidTop);
        m.text(cv, "4x", (mid + chart_w / 4, y), LABEL, &f.small, Anchor::MidTop);
        m.text(cv, "Fly A (Hz)", (cx0 + chart_w + 30, y), LABEL, &f.small, Anchor::TopLeft);
        m.text(cv, "Fly B (Hz)", (cx0 + chart_w + 110, y), LABEL, &f.small, Anchor::TopLeft);
        m.text(cv, "Diff Δ", (cx0 + chart_w + 190, y), LABEL, &f.small, Anchor::TopLeft);
        y += 18;
        cv.line((mid, y - 2), (mid, y + 160), 1, [80, 80, 76], 255);
        for row in res["rows"].as_array().into_iter().flatten().take(9) {
            let name: String = row["name"].as_str().unwrap_or("").chars().take(22).collect();
            m.text(cv, &name, (area.x, y), TEXT, &f.small, Anchor::TopLeft);
            let lr = num(&row["log2_ratio"]);
            let frac = (lr / 2.0).clamp(-1.0, 1.0);
            let bl = (frac.abs() * f64::from(chart_w / 2 - 4)) as i32;
            let col = if frac > 0.0 { [230, 100, 90] } else { [60, 135, 230] };
            if bl >= 2 {
                let rx = if frac > 0.0 { mid } else { mid - bl };
                cv.fill_rect(Rect::new(rx, y + 2, bl, 11), col, 2);
            } else {
                cv.circle((mid, y + 7), 2, [60, 60, 58]);
            }
            m.text(cv, &format!("{:.1}", num(&row["rate_a"])), (cx0 + chart_w + 30, y), LABEL, &f.small, Anchor::TopLeft);
            m.text(cv, &format!("{:.1}", num(&row["rate_b"])), (cx0 + chart_w + 110, y), INK, &f.small, Anchor::TopLeft);
            let d = num(&row["diff"]);
            m.text(cv, &format!("{d:+.1}"), (cx0 + chart_w + 190, y), if d.abs() > 2.0 { BAD } else { TEXT }, &f.small, Anchor::TopLeft);
            y += 16;
        }
    }
}

impl Page for LabDiff {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn draw(&mut self, m: &mut Menu, cv: &mut Canvas, rect: Rect, ctx: &mut Ctx) {
        let f = m.fonts();
        let q = ctx.host.query("lab.diff_state");
        m.text(cv, "CONNECTOME DIFF MODE", (rect.x + 24, rect.y + 16), INK, &f.head, Anchor::TopLeft);
        let mut y = m.subtitle(
            cv,
            rect,
            "Run two flies with different configurations from the exact same seed and inputs, and diff their activity region by region live.",
            None,
            2,
        );
        let right = rect.right() - 24;
        let items: Vec<_> = SCENARIOS
            .iter()
            .map(|(k, l)| (l.to_string(), Act::Custom(format!("scen:{k}")), format!("d_scen:{k}"), self.scenario == *k, None))
            .collect();
        y = m.flow_buttons(cv, rect.x + 24, y, right, &items, None, None, 8, Some("Shared Stimulus:")) + 8;
        let items: Vec<_> = MODS
            .iter()
            .map(|(k, l)| (l.to_string(), Act::Custom(format!("mod:{k}")), format!("d_mod:{k}"), self.modification == *k, None))
            .collect();
        y = m.flow_buttons(cv, rect.x + 24, y, right, &items, None, None, 8, Some("Fly B Configuration:")) - 28;
        y += 40;
        let job = &q["job"];
        let running = job.is_object();
        if running {
            cv.fill_rect(Rect::new(rect.x + 24, y + 10, rect.w - 300, 12), [30, 36, 48], 6);
            let frac = num(&job["frac"]).clamp(0.0, 1.0);
            cv.fill_rect(Rect::new(rect.x + 24, y + 10, 8.max(((rect.w - 300) as f64 * frac) as i32), 12), AMBER, 6);
            m.text(cv, &format!("{} …", job["note"].as_str().unwrap_or("")), (rect.x + 24, y + 26), LABEL, &f.small, Anchor::TopLeft);
        } else {
            m.button(
                cv,
                Rect::new(rect.x + 24, y, 260, 36),
                "Run Connectome Diff",
                Act::Custom("run".into()),
                "run_diff",
                Style::Primary,
                true,
                Some("Simulates Fly A (control) and Fly B (perturbed) with identical seed and inputs, measuring divergence region by region."),
                false,
                None,
            );
        }
        if let Some(e) = q["error"].as_str().filter(|e| !e.is_empty()) {
            m.text(cv, &format!("error: {e}"), (rect.x + 24, y + 10), BAD, &f.small, Anchor::TopLeft);
        }
        let res = &q["result"];
        let body = Rect::new(rect.x + 24, y + 46, rect.w - 48, rect.bottom() - y - 46 - 66);
        if res.is_object() {
            self.results(m, cv, body, res);
            m.button(
                cv,
                Rect::new(rect.x + 24, rect.bottom() - 58, m.bw("Export CSV + JSON", 200), 42),
                "Export CSV + JSON",
                Act::Custom("export".into()),
                "diff_export",
                Style::Normal,
                true,
                None,
                false,
                None,
            );
        } else {
            m.text(
                cv,
                if running { "" } else { "Click 'Run Connectome Diff' to compare region-by-region activity and divergence timeline." },
                (body.x, body.y + 12),
                LABEL,
                &f.small,
                Anchor::TopLeft,
            );
        }
        m.button(
            cv,
            Rect::new(rect.right() - 164, rect.bottom() - 58, 140, 42),
            "Back",
            Act::Back,
            "diff:back",
            Style::Primary,
            true,
            None,
            false,
            None,
        );
    }

    fn custom(&mut self, _m: &mut Menu, name: &str, _text: Option<&str>, ctx: &mut Ctx) {
        if let Some(k) = name.strip_prefix("scen:") {
            self.set_view(k, self.modification);
        } else if let Some(k) = name.strip_prefix("mod:") {
            self.set_view(self.scenario, k);
        } else if name == "run" {
            ctx.host.command("lab.diff_run", json!({"scenario": self.scenario, "mod": self.modification}));
        } else if name == "export" {
            ctx.host.command("lab.diff_export", Value::Null);
        }
    }
}
