//! Tool loadouts (`core/loadout.py`): which of the game's tools are on the hotbar, in which slots. Pure data and logic.
//!
//! Rules (GAME RULE, all UI conveniences, no neuron reads them): the hotbar has 10 slots on keys 1-9 and 0; a longer loadout pages, and
//! the previous/next page keys turn the page; the hand is in every loadout, in slot 1, and can't be removed; Lab-only tools never appear
//! outside Lab mode; larva mode hides tools with no larval sensory mapping; presets base, chaos, chemist, lab, all, pet and "custom".

use crate::config::{self, Config};

pub const HAND: &str = "hand";

fn info(name: &str) -> Option<&'static config::ToolInfo> {
    config::schema().loadout.catalog.iter().find(|t| t.name == name)
}

/// Whether a tool may be on the hotbar in this mode and brain.
pub fn available(name: &str, lab: bool, larva: bool) -> bool {
    let Some(t) = info(name) else { return false };
    !(t.lab_only && !lab) && !(larva && !t.larva)
}

pub fn available_tools(lab: bool, larva: bool) -> Vec<String> {
    config::schema().loadout.tool_names.iter().filter(|n| available(n, lab, larva)).cloned().collect()
}

/// A loadout's tools with unknown, duplicate and unavailable ones dropped and the hand first.
pub fn clean(names: &[String], lab: bool, larva: bool) -> Vec<String> {
    let mut out: Vec<String> = vec![];
    for n in names {
        if !out.contains(n) && available(n, lab, larva) {
            out.push(n.clone());
        }
    }
    out.retain(|n| n != HAND);
    let mut v = vec![HAND.to_string()];
    v.extend(out);
    v
}

/// The tools on the hotbar, in order, and which page of ten is showing.
#[derive(Debug, Clone, PartialEq)]
pub struct Loadout {
    pub preset: String,
    pub lab: bool,
    pub larva: bool,
    pub tools: Vec<String>,
    pub page: usize,
}

impl Loadout {
    pub fn new(tools: &[String], preset: &str, lab: bool, larva: bool) -> Self {
        Loadout { preset: preset.into(), lab, larva, tools: clean(tools, lab, larva), page: 0 }
    }

    fn size() -> usize {
        config::schema().loadout.page_size
    }

    pub fn len(&self) -> usize {
        self.tools.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    pub fn contains(&self, name: &str) -> bool {
        self.tools.iter().any(|t| t == name)
    }

    pub fn n_pages(&self) -> usize {
        self.tools.len().div_ceil(Self::size()).max(1)
    }

    pub fn page_tools(&self) -> &[String] {
        let (a, b) = (self.page * Self::size(), (self.page + 1) * Self::size());
        &self.tools[a.min(self.tools.len())..b.min(self.tools.len())]
    }

    /// The tool in hotbar slot 0-9 (keys 1-9, 0) on the page showing.
    pub fn slot_tool(&self, slot: usize) -> Option<&str> {
        self.page_tools().get(slot).map(String::as_str)
    }

    /// The slot (0-9) of this tool on the page showing.
    pub fn slot_of(&self, name: &str) -> Option<usize> {
        self.page_tools().iter().position(|t| t == name)
    }

    /// Previous (-1) / next (+1) page. False, and nothing happens, when everything fits on one page.
    pub fn turn_page(&mut self, d: i32) -> bool {
        let n = self.n_pages() as i32;
        if n <= 1 {
            return false;
        }
        self.page = (self.page as i32 + d).rem_euclid(n) as usize;
        true
    }

    /// Turn to the page that has this tool.
    pub fn show(&mut self, name: &str) {
        if let Some(i) = self.tools.iter().position(|t| t == name) {
            self.page = i / Self::size();
        }
    }

    /// The next (+1) or previous (-1) tool, wrapping (mouse wheel and gamepad bumpers). A tool that isn't in it starts from the front.
    pub fn neighbor(&self, name: &str, d: i32) -> &str {
        match self.tools.iter().position(|t| t == name) {
            None => &self.tools[if d > 0 { 0 } else { self.tools.len() - 1 }],
            Some(i) => &self.tools[(i as i32 + d).rem_euclid(self.tools.len() as i32) as usize],
        }
    }

    pub fn equip(&mut self, name: &str, index: Option<usize>) -> bool {
        if !available(name, self.lab, self.larva) || self.contains(name) {
            return false;
        }
        let i = index.map_or(self.tools.len(), |i| i.clamp(1, self.tools.len()));
        self.tools.insert(i, name.into());
        true
    }

    pub fn unequip(&mut self, name: &str) -> bool {
        if name == HAND || !self.contains(name) {
            return false;
        }
        self.tools.retain(|t| t != name);
        self.page = self.page.min(self.n_pages() - 1);
        true
    }

    /// Reorder. The hand stays in slot 1, and nothing goes ahead of it.
    pub fn move_to(&mut self, name: &str, index: usize) -> bool {
        if name == HAND || !self.contains(name) {
            return false;
        }
        self.tools.retain(|t| t != name);
        let i = index.clamp(1, self.tools.len());
        self.tools.insert(i, name.into());
        true
    }
}

/// The loadout for a preset name (or "auto") in this mode. "custom" uses the saved custom slots.
pub fn build(preset: &str, custom: &[String], lab: bool, larva: bool, mode: &str) -> Loadout {
    let ls = &config::schema().loadout;
    let name = if preset == "auto" { ls.mode_default.get(mode).map_or("base", String::as_str) } else { preset };
    let tools: Vec<String> = if name == "custom" {
        if custom.is_empty() { ls.presets["base"].clone() } else { custom.to_vec() }
    } else {
        ls.presets.get(name).unwrap_or(&ls.presets["base"]).clone()
    };
    Loadout::new(&tools, name, lab, larva)
}

/// The loadout a Config asks for, in its current mode.
pub fn resolve(cfg: &Config) -> Loadout {
    build(cfg.get_str("controls.loadout_preset"), &cfg.custom_loadout, cfg.lab(), cfg.larva(), cfg.get_str("brain.mode"))
}
