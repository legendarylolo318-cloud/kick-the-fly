//! CPython's `random.Random(seed).shuffle(list)`: Mersenne Twister (MT19937) seeded the way `random.seed(int)` does, and the same
//! Fisher-Yates over `_randbelow` with `getrandbits`. The Neuron of the Day's order is "a fixed shuffle seeded by the cycle number", so
//! the Rust game must shuffle exactly as Python did for the card of the day to be the same.

pub struct Mt {
    mt: [u32; 624],
    idx: usize,
}

impl Mt {
    fn init_genrand(s: u32) -> Mt {
        let mut mt = [0u32; 624];
        mt[0] = s;
        for i in 1..624 {
            mt[i] = 1_812_433_253u32.wrapping_mul(mt[i - 1] ^ (mt[i - 1] >> 30)).wrapping_add(i as u32);
        }
        Mt { mt, idx: 624 }
    }

    /// `random.seed(n)` for a non-negative integer: `init_by_array` over its 32-bit little-endian chunks.
    pub fn seeded(seed: u64) -> Mt {
        let mut key: Vec<u32> = vec![(seed & 0xFFFF_FFFF) as u32];
        if seed >> 32 != 0 {
            key.push((seed >> 32) as u32);
        }
        let mut m = Mt::init_genrand(19_650_218);
        let (mut i, mut j) = (1usize, 0usize);
        for _ in 0..624.max(key.len()) {
            m.mt[i] = (m.mt[i] ^ (m.mt[i - 1] ^ (m.mt[i - 1] >> 30)).wrapping_mul(1_664_525)).wrapping_add(key[j]).wrapping_add(j as u32);
            i += 1;
            j += 1;
            if i >= 624 {
                m.mt[0] = m.mt[623];
                i = 1;
            }
            if j >= key.len() {
                j = 0;
            }
        }
        for _ in 0..623 {
            m.mt[i] = (m.mt[i] ^ (m.mt[i - 1] ^ (m.mt[i - 1] >> 30)).wrapping_mul(1_566_083_941)).wrapping_sub(i as u32);
            i += 1;
            if i >= 624 {
                m.mt[0] = m.mt[623];
                i = 1;
            }
        }
        m.mt[0] = 0x8000_0000;
        m.idx = 624;
        m
    }

    fn next_u32(&mut self) -> u32 {
        if self.idx >= 624 {
            for k in 0..624 {
                let y = (self.mt[k] & 0x8000_0000) | (self.mt[(k + 1) % 624] & 0x7FFF_FFFF);
                let mut v = self.mt[(k + 397) % 624] ^ (y >> 1);
                if y & 1 != 0 {
                    v ^= 0x9908_B0DF;
                }
                self.mt[k] = v;
            }
            self.idx = 0;
        }
        let mut y = self.mt[self.idx];
        self.idx += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9D2C_5680;
        y ^= (y << 15) & 0xEFC6_0000;
        y ^ (y >> 18)
    }

    /// `getrandbits(k)` for k <= 32.
    pub fn getrandbits(&mut self, k: u32) -> u32 {
        self.next_u32() >> (32 - k)
    }

    /// `_randbelow(n)`
    pub fn randbelow(&mut self, n: u32) -> u32 {
        if n == 0 {
            return 0;
        }
        let k = 32 - n.leading_zeros();
        loop {
            let r = self.getrandbits(k);
            if r < n {
                return r;
            }
        }
    }

    /// `random.shuffle(x)`
    pub fn shuffle<T>(&mut self, x: &mut [T]) {
        for i in (1..x.len()).rev() {
            let j = self.randbelow(i as u32 + 1) as usize;
            x.swap(i, j);
        }
    }
}
