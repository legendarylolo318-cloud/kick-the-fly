//! Lab > Neural clamp (`lab/labclamp.py`): replay a recorded spike train into a modified connectome under dynamic clamp.
//!
//! Host: query `lab.clamp_state` = `{job: {frac, note} | null, error, result}`; commands `lab.clamp_run {scenario, mod}`,
//! `lab.clamp_export`.

use serde_json::{Value, json};

use crate::canvas::{Canvas, Rect};
use crate::menu::{Act, Anchor, Ctx, Menu, Page, Style};
use crate::theme::*;

const SCENARIOS: [(&str, &str); 3] = [("looming", "Looming escape"), ("sugar", "Sugar feeding"), ("antenna", "Antennal touch")];
const MODS: [(&str, &str); 5] = [
    ("lesion_loom", "Silence looming detectors (LPLC2, LC4)"),
    ("lesion_gf", "Silence giant fiber (DNp01)"),
    ("thresh_10", "Threshold: drop <10 synapses"),
    ("flips", "Flip uncertain neurotransmitters (50% @ 0.5)"),
    ("inhibition_0", "Picrotoxin: 100% inhibition block"),
];

fn num(v: &Value) -> f64 {
    v.as_f64().unwrap_or(f64::NAN)
}

pub struct LabClamp {
    scenario: &'static str,
    modification: &'static str,
}

impl Default for LabClamp {
    fn default() -> Self {
        Self { scenario: "looming", modification: "lesion_gf" }
    }
}

impl Page for LabClamp {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn draw(&mut self, m: &mut Menu, cv: &mut Canvas, rect: Rect, ctx: &mut Ctx) {
        let f = m.fonts();
        let q = ctx.host.query("lab.clamp_state");
        m.text(cv, "NEURAL CLAMP (DYNAMIC CLAMPING)", (rect.x + 24, rect.y + 16), INK, &f.head, Anchor::TopLeft);
        let y = m.subtitle(
            cv,
            rect,
            "Record a run's exact spike train and replay it into a modified connectome to isolate wiring effects from behavioral input variation.",
            None,
            2,
        );
        let note = "DYNAMIC CLAMP: Forced spikes override the network's own state and break feedback loops (such as proprioception). This is dynamic clamping, not an autonomous run.";
        let n = Menu::fit_lines(&f.small, note, rect.w - 68, 4).0.len() as i32;
        let banner = Rect::new(rect.x + 24, y, rect.w - 48, n * f.small.linesize + 10);
        cv.fill_rect(banner, [40, 24, 20], 6);
        cv.stroke_rect(banner, [220, 110, 60], 6);
        m.wrapped(cv, note, (banner.x + 10, banner.y + 5), banner.w - 20, [255, 180, 120], &f.small, 4);
        let mut y = banner.bottom() + 10;
        let right = rect.right() - 24;
        let items: Vec<_> = SCENARIOS
            .iter()
            .map(|(k, l)| (l.to_string(), Act::Custom(format!("scen:{k}")), format!("c_scen:{k}"), self.scenario == *k, None))
            .collect();
        y = m.flow_buttons(cv, rect.x + 24, y, right, &items, None, None, 8, Some("Reference scenario:")) + 8;
        let items: Vec<_> = MODS
            .iter()
            .map(|(k, l)| (l.to_string(), Act::Custom(format!("mod:{k}")), format!("c_mod:{k}"), self.modification == *k, None))
            .collect();
        y = m.flow_buttons(cv, rect.x + 24, y, right, &items, None, None, 8, Some("Connectome modification:")) - 30;
        y += 44;
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
                Rect::new(rect.x + 24, y, 280, 38),
                "Run Neural Clamp",
                Act::Custom("run".into()),
                "run_clamp",
                Style::Primary,
                true,
                Some("Records reference run with the chosen stimulus, then clamps those spikes into the modified connectome to directly observe downstream divergence."),
                false,
                None,
            );
        }
        if let Some(e) = q["error"].as_str().filter(|e| !e.is_empty()) {
            m.text(cv, &format!("error: {e}"), (rect.x + 24, y + 10), BAD, &f.small, Anchor::TopLeft);
        }
        let res = &q["result"];
        let body = Rect::new(rect.x + 24, y + 48, rect.w - 48, rect.bottom() - y - 48 - 66);
        if res.is_object() {
            self.results(m, cv, body, res);
            m.button(
                cv,
                Rect::new(rect.x + 24, rect.bottom() - 58, m.bw("Export CSV + JSON", 200), 42),
                "Export CSV + JSON",
                Act::Custom("export".into()),
                "clamp_export",
                Style::Normal,
                true,
                None,
                false,
                None,
            );
        } else {
            m.text(
                cv,
                if running { "" } else { "Click 'Run Neural Clamp' to compare free-running vs clamped activity side-by-side." },
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
            "clamp:back",
            Style::Primary,
            true,
            None,
            false,
            None,
        );
    }

    fn custom(&mut self, _m: &mut Menu, name: &str, _text: Option<&str>, ctx: &mut Ctx) {
        if let Some(k) = name.strip_prefix("scen:") {
            if let Some(s) = SCENARIOS.iter().find(|s| s.0 == k) {
                self.scenario = s.0;
            }
        } else if let Some(k) = name.strip_prefix("mod:") {
            if let Some(s) = MODS.iter().find(|s| s.0 == k) {
                self.modification = s.0;
            }
        } else if name == "run" {
            ctx.host.command("lab.clamp_run", json!({"scenario": self.scenario, "mod": self.modification}));
        } else if name == "export" {
            ctx.host.command("lab.clamp_export", Value::Null);
        }
    }
}

impl LabClamp {
    /// Restore the page's choices (the tests, and a returning player).
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
        let steps = num(&res["steps"]);
        m.text(
            cv,
            &format!(
                "Side-by-side Diff: Free-Running vs Clamped  ·  Stimulus: {}  ·  {} steps ({:.1} s)  ·  clamped {} neurons",
                res["stimulus"].as_str().unwrap_or(""),
                steps,
                steps * 0.005,
                super::day4::grp(num(&res["clamped_neurons"]), 0)
            ),
            (area.x, area.y),
            INK,
            &f.small,
            Anchor::TopLeft,
        );
        let mut y = area.y + 24;
        let xs = [area.x, area.x + 220, area.x + 380, area.x + 540];
        for (hx, label) in xs.iter().zip(["Key Readout", "Free-running (Hz)", "Clamped (Hz)", "Wiring Effect Δ (Hz)"]) {
            m.text(cv, label, (*hx, y), LABEL, &f.small, Anchor::TopLeft);
        }
        y += 18;
        let colour = |d: f64| {
            if d < -1.0 {
                BAD
            } else if d > 1.0 {
                AMBER
            } else {
                GOOD
            }
        };
        // Python's insertion order (serde_json sorts the keys)
        let order = ["giant_fiber_DNp01", "sugar_MN9", "steering_DNa01_02"];
        for name in order.into_iter().filter(|n| res["readouts"].get(*n).is_some()) {
            let v = &res["readouts"][name];
            let diff = num(&v["diff_hz"]);
            m.text(cv, &name.replace('_', " "), (xs[0], y), TEXT, &f.small, Anchor::TopLeft);
            m.text(cv, &format!("{:.1}", num(&v["free_hz"])), (xs[1], y), LABEL, &f.small, Anchor::TopLeft);
            m.text(cv, &format!("{:.1}", num(&v["clamped_hz"])), (xs[2], y), INK, &f.small, Anchor::TopLeft);
            m.text(cv, &format!("{diff:+.1}"), (xs[3], y), colour(diff), &f.small, Anchor::TopLeft);
            y += 18;
        }
        y += 12;
        m.text(cv, "Neuropil Region Rates (spikes/s per neuron):", (area.x, y), INK, &f.small, Anchor::TopLeft);
        y += 18;
        for (hx, label) in xs.iter().zip(["Neuropil Region", "Free-running (Hz)", "Clamped (Hz)", "Diff Δ (Hz)"]) {
            m.text(cv, label, (*hx, y), LABEL, &f.small, Anchor::TopLeft);
        }
        y += 18;
        let d = res["diff_region_rates"].as_object().cloned().unwrap_or_default();
        let mut regs: Vec<(&String, f64)> = d.iter().map(|(k, v)| (k, num(v))).collect();
        regs.sort_by(|a, b| b.1.abs().total_cmp(&a.1.abs())); // stable: ties keep the (sorted) region order, as `reverse=True` does
        for (reg, diff) in regs.into_iter().take(8) {
            let name: String = reg.chars().take(26).collect();
            m.text(cv, &name, (xs[0], y), TEXT, &f.small, Anchor::TopLeft);
            m.text(cv, &format!("{:.1}", num(&res["free_region_rates"][reg])), (xs[1], y), LABEL, &f.small, Anchor::TopLeft);
            m.text(cv, &format!("{:.1}", num(&res["clamped_region_rates"][reg])), (xs[2], y), INK, &f.small, Anchor::TopLeft);
            m.text(cv, &format!("{diff:+.1}"), (xs[3], y), colour(diff), &f.small, Anchor::TopLeft);
            y += 18;
        }
    }
}
