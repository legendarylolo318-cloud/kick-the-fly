//! Names the player gives flies (C with the mouse free opens the rename page for the fly you are looking at, unless the player bound C to something) and the tags drawn over them.
//! Names belong to a brain slot and last for the session; whether tags show is `[names] tags` in config.toml.

use std::collections::HashMap;

use bevy::prelude::*;

#[derive(Resource, Default)]
pub struct FlyNames(pub HashMap<u32, String>);

/// Set or clear (empty) the name of a slot; names are one line, trimmed, at most 16 characters.
pub fn set_name(names: &mut FlyNames, slot: u32, name: &str) {
    let name: String = name.split_whitespace().collect::<Vec<_>>().join(" ").chars().take(16).collect();
    if name.is_empty() {
        names.0.remove(&slot);
    } else {
        names.0.insert(slot, name);
    }
}

pub struct FlyNamesPlugin;

impl Plugin for FlyNamesPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<FlyNames>();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_are_cleaned_and_empty_removes() {
        let mut n = FlyNames::default();
        set_name(&mut n, 0, "  Sir   Buzz-a-lot the Third  ");
        assert_eq!(n.0[&0], "Sir Buzz-a-lot t");
        set_name(&mut n, 0, "   ");
        assert!(n.0.is_empty());
    }
}
