//! What the game tells the player: floating popups, the reaction log with its CONNECTOME / GAME RULE tags, and sound requests. Systems
//! write these messages; the HUD and audio plugins read them, so no gameplay code draws or plays anything itself.

use std::collections::VecDeque;

use bevy::prelude::*;

/// A floating word at a world position ("SWAT!", "DODGE!").
#[derive(Message, Debug, Clone)]
pub struct Popup {
    pub pos: Vec3,
    pub text: String,
    pub color: Color,
}

impl Popup {
    pub fn new(pos: Vec3, text: &str, rgb: (u8, u8, u8)) -> Self {
        Self { pos, text: text.to_string(), color: Color::srgb_u8(rgb.0, rgb.1, rgb.2) }
    }
}

/// Which honesty tag a log line carries (`docs/connectome-and-game-rules.md`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tag {
    Connectome,
    GameRule,
    Model,
}

impl Tag {
    pub fn label(self) -> &'static str {
        match self {
            Tag::Connectome => "CONNECTOME",
            Tag::GameRule => "GAME RULE",
            Tag::Model => "MODEL PREDICTION",
        }
    }

    /// The tag of a reaction word, from the Python `REACTION_SOURCE` table: REAL means live descending-neuron firing crossed a threshold.
    pub fn of_reaction(word: &str) -> Tag {
        match crate::reactions::reaction_source(word) {
            Some("real") => Tag::Connectome,
            _ => Tag::GameRule,
        }
    }
}

#[derive(Resource, Default)]
pub struct Notes {
    pub lines: VecDeque<(f32, String, Tag)>,
    /// Texts pushed since a Lab recording last drained them (`Game.note` logs a "note" event into a running recording).
    pub fresh: Vec<String>,
}

impl Notes {
    pub fn push(&mut self, now: f32, text: impl Into<String>, tag: Tag) {
        let text = text.into();
        if self.fresh.len() < 256 {
            self.fresh.push(text.clone());
        }
        self.lines.push_back((now, text, tag));
        while self.lines.len() > 8 {
            self.lines.pop_front();
        }
    }
}

/// A sound to play, by name (see `audio.rs`), with a volume 0..1 and the world position it comes from (None: not positional).
#[derive(Message, Debug, Clone)]
pub struct Sfx {
    pub name: &'static str,
    pub volume: f32,
}

impl Sfx {
    pub fn new(name: &'static str) -> Self {
        Self { name, volume: 1.0 }
    }

    pub fn vol(name: &'static str, volume: f32) -> Self {
        Self { name, volume }
    }
}

/// The screen shake after a hit or an explosion: seconds left and strength.
#[derive(Resource, Default)]
pub struct Shake {
    pub until: f32,
    pub strength: f32,
}

/// `Game.saved_msg`: the one-line toast under the status card ("1v1 duel: ON ...", "time: normal speed"), shown for 4 real seconds.
#[derive(Resource, Default)]
pub struct SavedMsg(pub Option<(String, f32)>);

impl SavedMsg {
    pub const SHOWN_S: f32 = 4.0;

    /// Show `text` from real time `real_now` on.
    pub fn set(&mut self, text: impl Into<String>, real_now: f32) {
        self.0 = Some((text.into(), real_now));
    }

    /// The text still on screen at `real_now`.
    pub fn current(&self, real_now: f32) -> Option<&str> {
        self.0.as_ref().filter(|(_, t)| real_now - t < Self::SHOWN_S).map(|(s, _)| s.as_str())
    }
}

pub struct FeedbackPlugin;

impl Plugin for FeedbackPlugin {
    fn build(&self, app: &mut App) {
        app.add_message::<Popup>().add_message::<Sfx>().init_resource::<Notes>().init_resource::<Shake>().init_resource::<SavedMsg>();
    }
}
