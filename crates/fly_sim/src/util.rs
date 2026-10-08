//! Small helpers shared by the loader, the tests and the benchmark.

use sha2::{Digest, Sha256};

/// Lower-case hex SHA-256 of `bytes`.
pub fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|b| format!("{b:02x}")).collect()
}

/// First 16 hex digits of the SHA-256: the fingerprint the Python golden script (`make_golden.py:sha`) writes.
pub fn sha16(bytes: &[u8]) -> String {
    sha256_hex(bytes)[..16].to_string()
}

/// Fingerprint of a list of neuron indices as little-endian `u32` (Python: `idx.astype("<u4").tobytes()`).
pub fn sha16_u32(idx: &[u32]) -> String {
    let mut bytes = Vec::with_capacity(idx.len() * 4);
    for i in idx {
        bytes.extend_from_slice(&i.to_le_bytes());
    }
    sha16(&bytes)
}

/// Fingerprint of an `f32` array by its bit patterns (Python: `arr.astype("<f4").tobytes()`).
pub fn sha16_f32(values: &[f32]) -> String {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for v in values {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    sha16(&bytes)
}
