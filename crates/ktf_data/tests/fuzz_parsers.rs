//! A short, repeatable fuzz of everything that reads a file or a pasted text a player can hand to the game: `.ktfsave`, share codes
//! (`KTF1-...`, `.ktfshare`), protocol YAML, the pet's save and bundle zips. Damaged input must come back as an error (or be
//! quarantined), never as a panic. The mutations are seeded, so a failure names its case and can be run again.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::PathBuf;

use ktf_data::bundle::{Bundle, create_live};
use ktf_data::savestate::{SaveFile, read_meta};
use ktf_data::{pet, protocol, sharecode};
use serde_json::json;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n.max(1) as u64) as usize
    }
}

/// Flip, overwrite, insert, delete or cut: a handful of edits to a copy of `src`.
fn mutate(src: &[u8], r: &mut Rng) -> Vec<u8> {
    let mut b = src.to_vec();
    for _ in 0..1 + r.below(6) {
        if b.is_empty() {
            break;
        }
        let i = r.below(b.len());
        match r.below(6) {
            0 => b[i] ^= 1 << r.below(8),
            1 => b[i] = r.next() as u8,
            2 => b.insert(i, r.next() as u8),
            3 => {
                b.remove(i);
            }
            4 => b.truncate(i),
            _ => {
                let j = (i + r.below(64)).min(b.len());
                let chunk = b[i..j].to_vec();
                b.splice(i..i, chunk);
            }
        }
    }
    b
}

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/python_files")
}

fn scratch(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("ktf-fuzz-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// Run `f` on `cases` mutations of `seed_input`; collect the cases that panicked.
fn fuzz(label: &str, seed_input: &[u8], cases: usize, mut f: impl FnMut(&[u8])) {
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let mut bad = Vec::new();
    let mut r = Rng(0x9E37_79B9_7F4A_7C15 ^ label.len() as u64);
    for case in 0..cases {
        let input = mutate(seed_input, &mut r);
        if catch_unwind(AssertUnwindSafe(|| f(&input))).is_err() {
            bad.push(case);
        }
    }
    std::panic::set_hook(hook);
    assert!(bad.is_empty(), "{label}: {} of {cases} damaged inputs panicked (cases {:?})", bad.len(), &bad[..bad.len().min(10)]);
}

#[test]
fn damaged_saves_never_panic() {
    let d = scratch("save");
    let good = std::fs::read(fixtures().join("savestate/real.ktfsave")).unwrap();
    let p = d.join("x.ktfsave");
    fuzz("ktfsave", &good, 120, |bytes| {
        std::fs::write(&p, bytes).unwrap();
        let _ = read_meta(&p);
        if let Ok(s) = SaveFile::read(&p) {
            for i in 0..s.flies().min(4) {
                let _ = s.sim_state(i);
            }
        }
    });
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn damaged_share_codes_never_panic() {
    let ctx = sharecode::Context::default();
    for name in ["surgery.txt", "loadout.txt", "protocol.txt", "challenge.txt", "lab.txt", "contraption.txt", "messy.txt"] {
        let good = std::fs::read(fixtures().join("share").join(name)).unwrap();
        fuzz(name, &good, 400, |bytes| {
            let text = String::from_utf8_lossy(bytes);
            for code in [sharecode::decode(&text), sharecode::read_any(&text)].into_iter().flatten() {
                if sharecode::validate(&code, &ctx).is_none() {
                    let _ = sharecode::preview(&code, &ctx);
                }
            }
        });
    }
    // a `.ktfshare` file on disk
    let d = scratch("share");
    let good = std::fs::read(fixtures().join("share/big.ktfshare")).unwrap();
    let p = d.join("x.ktfshare");
    fuzz("ktfshare", &good, 150, |bytes| {
        std::fs::write(&p, bytes).unwrap();
        let _ = sharecode::read_any(&p.to_string_lossy());
    });
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn damaged_protocols_never_panic() {
    for (name, text) in protocol::bundled().iter().take(12) {
        fuzz(name, text.as_bytes(), 150, |bytes| {
            if let Ok(p) = protocol::load_text(&String::from_utf8_lossy(bytes), "fuzz") {
                let _ = protocol::summary(&p);
            }
        });
    }
}

#[test]
fn a_damaged_pet_save_never_panics() {
    let d = scratch("pet");
    let mut p = pet::Pet::new(d.clone(), 1_000.0);
    p.create_new(Some(7), "adult", 1_000.0);
    p.feed(0.3, "sugar", true, 1_050.0);
    let good = std::fs::read(p.save_file()).unwrap();
    let (file, backup) = (p.save_file(), p.backup_file());
    fuzz("pet", &good, 400, |bytes| {
        std::fs::write(&file, bytes).unwrap();
        let _ = std::fs::remove_file(&backup);
        let mut q = pet::Pet::new(d.clone(), 2_000.0);
        q.load_or_create("adult", 2_000.0);
        q.apply_catchup(90_000.0);
        let _ = q.mood_word();
    });
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn damaged_bundles_never_panic() {
    let d = scratch("bundle");
    let f1 = d.join("recording-spikes.csv");
    std::fs::write(&f1, "time_ms,row\r\n5.0,1\r\n").unwrap();
    let z = d.join("good.zip");
    create_live(&[f1], &json!({"seeds": [7]}), &z, &json!({}), "4.0.0", "2026-01-01T00:00:00").unwrap();
    let good = std::fs::read(&z).unwrap();
    let p = d.join("x.zip");
    fuzz("bundle", &good, 400, |bytes| {
        std::fs::write(&p, bytes).unwrap();
        if let Ok(mut b) = Bundle::open(&p) {
            let _ = b.verify();
            for n in b.names() {
                let _ = b.read(&n);
            }
        }
    });
    let _ = std::fs::remove_dir_all(&d);
}
