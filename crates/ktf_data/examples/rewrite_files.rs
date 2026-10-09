//! `cargo run -p ktf_data --example rewrite_files -- <dir with python stores> <out dir>`: reads the Python-written memory, Neurodex and
//! a .ktfsave and writes them again, to check that Python reads what Rust writes.
use std::path::Path;
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (src, out) = (Path::new(&a[1]), Path::new(&a[2]));
    std::fs::create_dir_all(out).unwrap();
    let sig = [41495.0, 50.26125336];
    ktf_data::stores::FlyMemory::load(&src.join("stores/fly-memory.npz"), &sig).unwrap().save(&out.join("fly-memory.npz")).unwrap();
    let mut p = ktf_data::stores::Progress::open(src.join("stores/neurodex.json"));
    p.path = out.join("neurodex.json");
    p.save();
    ktf_data::savestate::SaveFile::read(&src.join("savestate/real.ktfsave")).unwrap().write(&out.join("rewritten.ktfsave")).unwrap();
}
