//! `cargo run -p ktf_data --example share_encode -- <kind> '<json payload>'` prints a share code (used to check that Python reads ours).
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let payload: serde_json::Value = serde_json::from_str(&a[2]).expect("payload JSON");
    match ktf_data::sharecode::encode(&a[1], &payload, None) {
        Ok(c) => println!("{c}"),
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(2);
        }
    }
}
