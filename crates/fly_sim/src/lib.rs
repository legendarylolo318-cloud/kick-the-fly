//! `fly_sim`: the Kick the Fly connectome simulation, without a game engine.
//!
//! * [`pack`] loads the brain pack (`data/kick_brain.npz`, the same file the Python game uses) into CSR sparse form.
//! * [`lif`] is the leaky integrate-and-fire model, step for step the Python reference
//!   (`kickthefly/sim/connectome/sim.py` + `backends.py:CPUBackend`), parallelised with rayon. Given the same pack, seed and
//!   stimulus it produces the same spikes as NumPy.
//! * [`rng`] reimplements NumPy's `default_rng` stream (SeedSequence, PCG64, ziggurat normals) so the noise matches too.
//! * [`brain`] selects neurons by spec (`type:DNp01`, `prefix:KC`, ...) and holds the held-current state the game and assays use.
//!
//! Nothing here depends on Bevy or on a window; everything runs headless.

pub mod activity;
pub mod assay_jobs;
pub mod assays;
pub mod autopsy;
pub mod benchmark;
pub mod brain;
pub mod cards;
pub mod challenge;
pub mod csr;
pub mod duel;
pub mod groups;
pub mod inspect;
pub mod lif;
pub mod memory;
pub mod mirror;
pub mod netsci;
pub mod neurodex;
pub mod npy;
pub mod outdoors;
pub mod pack;
pub mod race;
pub mod rng;
pub mod robustness;
pub mod sensitivity;
pub mod sleepdep;
pub mod stats;
pub mod surgery;
pub mod synthetic;
pub mod tournament;
pub mod training;
pub mod util;
pub mod validation;
pub mod wiring;
mod ziggurat_tables;

use std::fmt;

/// Errors from loading a pack or building a sim.
#[derive(Debug)]
pub enum Error {
    Io(std::io::Error),
    /// The file is not what it should be (not a zip, a missing array, an unsupported dtype).
    Format(String),
    /// A neuron spec that cannot be resolved.
    Spec(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io(e) => write!(f, "I/O error: {e}"),
            Error::Format(m) => write!(f, "bad brain pack: {m}"),
            Error::Spec(m) => write!(f, "bad neuron spec: {m}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}

pub use activity::ActivityBuffer;
pub use brain::Brain;
pub use lif::{LifParams, LifSim};
pub use pack::Pack;
