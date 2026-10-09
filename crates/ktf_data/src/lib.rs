//! Files the Python game wrote and the Rust game must keep reading, plus the per-OS folders they live in.
//!
//! * [`sharecode`]: `KTF1-<KIND>-…` share codes and `.ktfshare` files (Python: `core/sharecode.py`).
//! * [`config`]: `config.toml`, schema v3, with the same migrations, clamping and tolerance (Python: `core/config.py`).
//! * [`savestate`]: `.ktfsave` files (zip of `meta.json` + `arrays.npz`), read and written (Python: `core/savestate.py`).
//! * [`stores`]: `neurodex.json`, `contraptions.json`, `lesion_loadouts.json`, `fly-memory.npz` + `training-log.json`.
//! * [`paths`]: where config, saves and data go (Python: `core/paths.py`), through the `directories` crate.
//!
//! No Bevy here: the crate builds and tests in seconds, and the game and the Lab use it the same way.

pub mod config;
pub mod paths;
pub mod savestate;
pub mod sharecode;
pub mod stores;
