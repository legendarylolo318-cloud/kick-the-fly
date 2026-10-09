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
}

impl Notes {
    pub fn push(&mut self, now: f32, text: impl Into<String>, tag: Tag) {
        self.lines.push_back((now, text.into(), tag));
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

pub struct FeedbackPlugin;

impl Plugin for FeedbackPlugin {
    fn build(&self, app: &mut App) {
        app.add_message::<Popup>().add_message::<Sfx>().init_resource::<Notes>().init_resource::<Shake>();
    }
}
