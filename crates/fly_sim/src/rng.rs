//! NumPy's `default_rng(seed)` stream, reimplemented: `SeedSequence` → PCG64 (XSL-RR, 128-bit state) → the float32 and float64
//! ziggurat normals and the Lemire bounded-integer sampler.
//!
//! The Python simulator draws its membrane noise from this generator (`np.random.default_rng(seed).standard_normal(n * 16,
//! dtype=np.float32)`, then one `integers(0, size - n)` offset per step). Reproducing the stream bit for bit is what lets the Rust
//! step reproduce Python's spikes bit for bit (see `tests/golden.rs`). Each routine follows NumPy's C source
//! (`numpy/random/src/{pcg64,distributions}`, `bit_generator.pyx`); the golden `rng.json` pins them against the real library.

// --- SeedSequence ------------------------------------------------------------------------------------------------------------
const INIT_A: u32 = 0x43b0_d7e5;
const MULT_A: u32 = 0x931e_8875;
const INIT_B: u32 = 0x8b51_f9dd;
const MULT_B: u32 = 0x58f3_8ded;
const MIX_MULT_L: u32 = 0xca01_f9dd;
const MIX_MULT_R: u32 = 0x4973_f715;
const XSHIFT: u32 = 16;
const POOL: usize = 4;

fn hashmix(value: u32, hash_const: &mut u32) -> u32 {
    let mut v = value ^ *hash_const;
    *hash_const = hash_const.wrapping_mul(MULT_A);
    v = v.wrapping_mul(*hash_const);
    v ^ (v >> XSHIFT)
}

fn mix(x: u32, y: u32) -> u32 {
    let r = MIX_MULT_L.wrapping_mul(x).wrapping_sub(MIX_MULT_R.wrapping_mul(y));
    r ^ (r >> XSHIFT)
}

/// `np.random.SeedSequence(seed)` for a non-negative integer seed (no spawn key).
#[derive(Debug, Clone)]
pub struct SeedSequence {
    pool: [u32; POOL],
}

impl SeedSequence {
    pub fn new(seed: u64) -> Self {
        let mut entropy = vec![seed as u32];
        if seed > u32::MAX as u64 {
            entropy.push((seed >> 32) as u32);
        }
        let mut pool = [0u32; POOL];
        let mut hash_const = INIT_A;
        for (i, slot) in pool.iter_mut().enumerate() {
            *slot = hashmix(entropy.get(i).copied().unwrap_or(0), &mut hash_const);
        }
        for i_src in 0..POOL {
            for i_dst in 0..POOL {
                if i_src != i_dst {
                    let h = hashmix(pool[i_src], &mut hash_const);
                    pool[i_dst] = mix(pool[i_dst], h);
                }
            }
        }
        for &e in entropy.iter().skip(POOL) {
            for slot in pool.iter_mut() {
                let h = hashmix(e, &mut hash_const);
                *slot = mix(*slot, h);
            }
        }
        Self { pool }
    }

    /// `generate_state(n_words, np.uint64)`.
    pub fn generate_state_u64(&self, n_words: usize) -> Vec<u64> {
        let mut hash_const = INIT_B;
        let mut words = Vec::with_capacity(n_words * 2);
        for i in 0..n_words * 2 {
            let mut d = self.pool[i % POOL] ^ hash_const;
            hash_const = hash_const.wrapping_mul(MULT_B);
            d = d.wrapping_mul(hash_const);
            d ^= d >> XSHIFT;
            words.push(d);
        }
        words.as_chunks::<2>().0.iter().map(|w| u64::from(w[0]) | (u64::from(w[1]) << 32)).collect()
    }
}

// --- PCG64 ---------------------------------------------------------------------------------------------------------------------
const PCG_MULT: u128 = 0x2360_ed05_1fc6_5da4_4385_df64_9fcc_f645;

/// The complete state of a `PCG64`, as NumPy reports it in `bit_generator.state` (a save file carries it).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PcgState {
    pub state: u128,
    pub inc: u128,
    pub has_uint32: bool,
    pub uinteger: u32,
}

/// `np.random.PCG64`, including the buffered 32-bit half that `next_uint32` shares with `next_uint64`.
#[derive(Debug, Clone)]
pub struct Pcg64 {
    state: u128,
    inc: u128,
    has_u32: bool,
    buffered: u32,
}

impl Pcg64 {
    pub fn new(seed: u64) -> Self {
        let val = SeedSequence::new(seed).generate_state_u64(4);
        let initstate = (u128::from(val[0]) << 64) | u128::from(val[1]);
        let initseq = (u128::from(val[2]) << 64) | u128::from(val[3]);
        let mut rng = Self { state: 0, inc: (initseq << 1) | 1, has_u32: false, buffered: 0 };
        rng.step();
        rng.state = rng.state.wrapping_add(initstate);
        rng.step();
        rng
    }

    /// `bit_generator.state`.
    pub fn export_state(&self) -> PcgState {
        PcgState { state: self.state, inc: self.inc, has_uint32: self.has_u32, uinteger: self.buffered }
    }

    /// `bit_generator.state = ...`.
    pub fn import_state(&mut self, s: PcgState) {
        self.state = s.state;
        self.inc = s.inc;
        self.has_u32 = s.has_uint32;
        self.buffered = s.uinteger;
    }

    #[inline]
    fn step(&mut self) {
        self.state = self.state.wrapping_mul(PCG_MULT).wrapping_add(self.inc);
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.step();
        let s = self.state;
        let xored = ((s >> 64) as u64) ^ (s as u64);
        xored.rotate_right((s >> 122) as u32)
    }

    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        if self.has_u32 {
            self.has_u32 = false;
            return self.buffered;
        }
        let next = self.next_u64();
        self.has_u32 = true;
        self.buffered = (next >> 32) as u32;
        next as u32
    }

    #[inline]
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 * (1.0 / 16_777_216.0)
    }

    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
    }
}

// --- ziggurat tables -----------------------------------------------------------------------------------------------------------
// The 256-layer tables are NumPy's own constants (src/ziggurat_tables.rs, extracted by tools/extract_ziggurat.py): they cannot be
// rebuilt bit for bit from the textbook algorithm in f64.
use crate::ziggurat_tables::{FI_D, FI_F, KI_D, KI_F, WI_D, WI_F};

const NOR_R: f64 = 3.654_152_885_361_009;
const NOR_INV_R: f64 = 0.273_661_237_329_758_3;
const NOR_R_F: f32 = 3.654_152_9;
const NOR_INV_R_F: f32 = 0.273_661_23;

// --- Generator -----------------------------------------------------------------------------------------------------------------
/// `np.random.Generator(PCG64(seed))`, the parts the simulator uses.
#[derive(Debug, Clone)]
pub struct Generator {
    bits: Pcg64,
}

impl Generator {
    pub fn new(seed: u64) -> Self {
        Self { bits: Pcg64::new(seed) }
    }

    pub fn export_state(&self) -> PcgState {
        self.bits.export_state()
    }

    pub fn import_state(&mut self, s: PcgState) {
        self.bits.import_state(s);
    }

    /// `bit_generator.random_raw()`.
    pub fn random_raw(&mut self) -> u64 {
        self.bits.next_u64()
    }

    /// `rng.standard_normal(dtype=np.float32)` (one value).
    pub fn standard_normal_f32(&mut self) -> f32 {
        let b = &mut self.bits;
        loop {
            let r = b.next_u32();
            let idx = (r & 0xff) as usize;
            let sign = (r >> 8) & 1;
            let rabs = (r >> 9) & 0x007f_ffff;
            let mut x = rabs as f32 * f32::from_bits(WI_F[idx]);
            if sign == 1 {
                x = -x;
            }
            if rabs < KI_F[idx] {
                return x;
            }
            if idx == 0 {
                loop {
                    let xx = -NOR_INV_R_F * (-b.next_f32()).ln_1p();
                    let yy = -(-b.next_f32()).ln_1p();
                    if yy + yy > xx * xx {
                        return if (rabs >> 8) & 1 == 1 { -(NOR_R_F + xx) } else { NOR_R_F + xx };
                    }
                }
            } else {
                let (f_prev, f_here) = (f32::from_bits(FI_F[idx - 1]), f32::from_bits(FI_F[idx]));
                let lhs = (f_prev - f_here) * b.next_f32() + f_here;
                if f64::from(lhs) < (-0.5f64 * f64::from(x) * f64::from(x)).exp() {
                    return x;
                }
            }
        }
    }

    /// `rng.standard_normal()` (float64, one value).
    pub fn standard_normal_f64(&mut self) -> f64 {
        let b = &mut self.bits;
        loop {
            let mut r = b.next_u64();
            let idx = (r & 0xff) as usize;
            r >>= 8;
            let sign = r & 1;
            let rabs = (r >> 1) & 0x000f_ffff_ffff_ffff;
            let mut x = rabs as f64 * f64::from_bits(WI_D[idx]);
            if sign == 1 {
                x = -x;
            }
            if rabs < KI_D[idx] {
                return x;
            }
            if idx == 0 {
                loop {
                    let xx = -NOR_INV_R * (-b.next_f64()).ln_1p();
                    let yy = -(-b.next_f64()).ln_1p();
                    if yy + yy > xx * xx {
                        return if (rabs >> 8) & 1 == 1 { -(NOR_R + xx) } else { NOR_R + xx };
                    }
                }
            } else if (f64::from_bits(FI_D[idx - 1]) - f64::from_bits(FI_D[idx])) * b.next_f64() + f64::from_bits(FI_D[idx])
                < (-0.5 * x * x).exp()
            {
                return x;
            }
        }
    }

    /// `rng.normal(loc, scale)` (float64).
    pub fn normal(&mut self, loc: f64, scale: f64) -> f64 {
        loc + scale * self.standard_normal_f64()
    }

    /// `rng.standard_normal(out.len(), dtype=np.float32)`.
    pub fn fill_standard_normal_f32(&mut self, out: &mut [f32]) {
        for o in out {
            *o = self.standard_normal_f32();
        }
    }

    /// `rng.integers(low, high)` (int64, high exclusive): NumPy's buffered Lemire sampler on 32 bits when the range fits.
    pub fn integers(&mut self, low: u64, high: u64) -> u64 {
        assert!(high > low, "empty range");
        let rng = high - low - 1;
        if rng == 0 {
            return low;
        }
        if rng <= 0xFFFF_FFFF {
            if rng == 0xFFFF_FFFF {
                return low + u64::from(self.bits.next_u32());
            }
            let rng = rng as u32;
            let rng_excl = rng + 1;
            let mut m = u64::from(self.bits.next_u32()) * u64::from(rng_excl);
            let mut leftover = (m & 0xFFFF_FFFF) as u32;
            if leftover < rng_excl {
                let threshold = (u32::MAX - rng) % rng_excl;
                while leftover < threshold {
                    m = u64::from(self.bits.next_u32()) * u64::from(rng_excl);
                    leftover = (m & 0xFFFF_FFFF) as u32;
                }
            }
            return low + (m >> 32);
        }
        // 64-bit Lemire
        let rng_excl = rng.wrapping_add(1);
        let mut m = u128::from(self.bits.next_u64()) * u128::from(rng_excl);
        let mut leftover = m as u64;
        if rng_excl != 0 && leftover < rng_excl {
            let threshold = (u64::MAX - rng) % rng_excl;
            while leftover < threshold {
                m = u128::from(self.bits.next_u64()) * u128::from(rng_excl);
                leftover = m as u64;
            }
        }
        low + (m >> 64) as u64
    }

    /// `Generator.random_interval(max)`: uniform on `0..=max` by masked rejection (what `shuffle` and `permutation` use).
    fn random_interval(&mut self, max: u64) -> u64 {
        if max == 0 {
            return 0;
        }
        let mut mask = max;
        mask |= mask >> 1;
        mask |= mask >> 2;
        mask |= mask >> 4;
        mask |= mask >> 8;
        mask |= mask >> 16;
        mask |= mask >> 32;
        if max <= 0xFFFF_FFFF {
            loop {
                let v = u64::from(self.bits.next_u32()) & mask;
                if v <= max {
                    return v;
                }
            }
        }
        loop {
            let v = self.bits.next_u64() & mask;
            if v <= max {
                return v;
            }
        }
    }

    /// `rng.shuffle(x)` for a 1-D array: Fisher-Yates from the back with `random_interval`, as NumPy does.
    pub fn shuffle<T>(&mut self, x: &mut [T]) {
        for i in (1..x.len()).rev() {
            let j = self.random_interval(i as u64) as usize;
            x.swap(i, j);
        }
    }
}
