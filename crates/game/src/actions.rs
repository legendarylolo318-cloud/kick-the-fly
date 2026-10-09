//! The rebindable hotkeys (`config.ACTIONS`): the game asks "was `big_view` pressed", not "was B pressed", so Settings > Controls changes
//! really change the keys. Built each frame from the keyboard and the player's `[keys]` table, after the menu has taken the keys it wants
//! (while the menu is open the input is cleared, so no action fires).

use std::collections::HashSet;

use bevy::prelude::*;

use crate::menu_ui::{UserConfig, pygame_key_name};

#[derive(Resource, Default)]
pub struct Actions {
    just: HashSet<String>,
    held: HashSet<String>,
}

impl Actions {
    pub fn just(&self, action: &str) -> bool {
        self.just.contains(action)
    }

    pub fn held(&self, action: &str) -> bool {
        self.held.contains(action)
    }
}

fn collect(keys: Res<ButtonInput<KeyCode>>, cfg: Res<UserConfig>, mut act: ResMut<Actions>) {
    act.just.clear();
    act.held.clear();
    for k in keys.get_just_pressed() {
        if let Some(name) = pygame_key_name(*k) {
            act.just.extend(cfg.0.actions_for(name));
        }
    }
    for k in keys.get_pressed() {
        if let Some(name) = pygame_key_name(*k) {
            act.held.extend(cfg.0.actions_for(name));
        }
    }
}

pub struct ActionsPlugin;

impl Plugin for ActionsPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Actions>().add_systems(PreUpdate, collect.after(crate::menu_ui::MenuInputSet));
    }
}
