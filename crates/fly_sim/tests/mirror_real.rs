//! The mirror-averaged matrix of the real pack equals `simcore.symmetrize_weights` (Python: nnz 13317281, sum 11661.67984092337).

use fly_sim::pack::{self, Pack};

#[test]
fn real_pack_symmetrised_like_python() {
    let Some(path) = pack::find_default() else { return };
    let pk = Pack::load(&path).unwrap();
    let s = fly_sim::mirror::symmetrize(&pk.weights.csr, &pk.labels.instance);
    assert_eq!(s.nnz(), 13_317_281);
    let sum: f64 = s.data.iter().map(|&v| f64::from(v)).sum();
    assert!((sum - 11661.67984092337).abs() < 1e-3, "{sum}");
}
