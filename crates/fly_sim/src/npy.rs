//! A small reader for NumPy `.npy` arrays inside `.npz` zip archives: exactly the dtypes the brain pack uses.
//!
//! `numpy.savez_compressed` writes one `.npy` per array into a deflate-compressed zip. Supported element types: `bool`, `i2`, `i4`,
//! `i8`, `f4`, `f8` and fixed-width unicode `<U<n>` (UCS-4, NUL padded). Arrays are C-ordered, little-endian.

use std::collections::HashMap;
use std::io::{Read, Seek};

use crate::Error;

/// One decoded array.
#[derive(Debug, Clone)]
pub enum Data {
    Bool(Vec<bool>),
    I16(Vec<i16>),
    I32(Vec<i32>),
    I64(Vec<i64>),
    F32(Vec<f32>),
    F64(Vec<f64>),
    Str(Vec<String>),
}

#[derive(Debug, Clone)]
pub struct Array {
    pub shape: Vec<usize>,
    pub data: Data,
}

impl Array {
    pub fn len(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn kind(&self) -> &'static str {
        match self.data {
            Data::Bool(_) => "bool",
            Data::I16(_) => "i16",
            Data::I32(_) => "i32",
            Data::I64(_) => "i64",
            Data::F32(_) => "f32",
            Data::F64(_) => "f64",
            Data::Str(_) => "str",
        }
    }
}

macro_rules! accessor {
    ($name:ident, $variant:ident, $ty:ty) => {
        pub fn $name(self) -> Result<Vec<$ty>, Error> {
            let kind = self.kind();
            match self.data {
                Data::$variant(v) => Ok(v),
                _ => Err(Error::Format(format!("expected {} array, found {kind}", stringify!($ty)))),
            }
        }
    };
}

impl Array {
    accessor!(into_i16, I16, i16);
    accessor!(into_i32, I32, i32);
    accessor!(into_i64, I64, i64);
    accessor!(into_f32, F32, f32);
    accessor!(into_strings, Str, String);
}

/// All arrays of an `.npz`, read eagerly (the pack is ~40 MB compressed).
pub struct Npz {
    arrays: HashMap<String, Array>,
}

impl Npz {
    pub fn read<R: Read + Seek>(reader: R) -> Result<Self, Error> {
        let mut zip = zip::ZipArchive::new(reader).map_err(|e| Error::Format(format!("not a zip archive: {e}")))?;
        let mut arrays = HashMap::new();
        for i in 0..zip.len() {
            let mut file = zip.by_index(i).map_err(|e| Error::Format(format!("bad zip entry: {e}")))?;
            let name = file.name().trim_end_matches(".npy").to_string();
            let mut bytes = Vec::with_capacity(file.size() as usize);
            file.read_to_end(&mut bytes)?;
            arrays.insert(name, parse_npy(&bytes)?);
        }
        Ok(Self { arrays })
    }

    pub fn contains(&self, name: &str) -> bool {
        self.arrays.contains_key(name)
    }

    pub fn take(&mut self, name: &str) -> Result<Array, Error> {
        self.arrays.remove(name).ok_or_else(|| Error::Format(format!("array '{name}' is missing from the pack")))
    }

    pub fn take_opt(&mut self, name: &str) -> Option<Array> {
        self.arrays.remove(name)
    }
}

fn parse_npy(bytes: &[u8]) -> Result<Array, Error> {
    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
        return Err(Error::Format("not an .npy array".into()));
    }
    let major = bytes[6];
    let (header_len, header_start) = match major {
        1 => (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10),
        2 | 3 => (u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize, 12),
        v => return Err(Error::Format(format!("unsupported .npy version {v}"))),
    };
    let header = std::str::from_utf8(&bytes[header_start..header_start + header_len])
        .map_err(|_| Error::Format("non-UTF-8 .npy header".into()))?;
    let descr = dict_value(header, "descr").ok_or_else(|| Error::Format("no descr in .npy header".into()))?;
    let descr = descr.trim_matches(|c| c == '\'' || c == '"').to_string();
    if dict_value(header, "fortran_order").is_some_and(|v| v.starts_with("True")) {
        return Err(Error::Format("Fortran-ordered arrays are not supported".into()));
    }
    let shape = parse_shape(&dict_value(header, "shape").ok_or_else(|| Error::Format("no shape in .npy header".into()))?)?;
    let count: usize = shape.iter().product();
    let body = &bytes[header_start + header_len..];

    let need = |width: usize| -> Result<(), Error> {
        if body.len() < count * width {
            Err(Error::Format(format!("array body too short: {} bytes for {count} x {width}", body.len())))
        } else {
            Ok(())
        }
    };
    macro_rules! numeric {
        ($t:ty, $variant:ident) => {{
            const W: usize = std::mem::size_of::<$t>();
            need(W)?;
            let v = body[..count * W].chunks_exact(W).map(|c| <$t>::from_le_bytes(c.try_into().unwrap())).collect();
            Data::$variant(v)
        }};
    }
    let data = match descr.as_str() {
        "|b1" | "?" => {
            need(1)?;
            Data::Bool(body[..count].iter().map(|&b| b != 0).collect())
        }
        "<i2" => numeric!(i16, I16),
        "<i4" => numeric!(i32, I32),
        "<i8" => numeric!(i64, I64),
        "<f4" => numeric!(f32, F32),
        "<f8" => numeric!(f64, F64),
        d if d.starts_with("<U") => {
            let chars: usize = d[2..].parse().map_err(|_| Error::Format(format!("bad dtype {d}")))?;
            let width = chars * 4;
            need(width)?;
            let mut out = Vec::with_capacity(count);
            for item in body[..count * width].chunks_exact(width.max(1)) {
                let mut s = String::new();
                for ch in item.as_chunks::<4>().0.iter() {
                    let cp = u32::from_le_bytes(*ch);
                    if cp == 0 {
                        break;
                    }
                    s.push(char::from_u32(cp).unwrap_or('\u{fffd}'));
                }
                out.push(s);
            }
            if chars == 0 {
                out = vec![String::new(); count];
            }
            Data::Str(out)
        }
        other => return Err(Error::Format(format!("unsupported dtype {other}"))),
    };
    Ok(Array { shape, data })
}

/// The text after `'key':` up to the next top-level comma (tuples keep their commas).
fn dict_value(header: &str, key: &str) -> Option<String> {
    let pat = format!("'{key}':");
    let start = header.find(&pat)? + pat.len();
    let rest = header[start..].trim_start();
    let mut depth = 0i32;
    let mut end = rest.len();
    for (i, c) in rest.char_indices() {
        match c {
            '(' | '[' => depth += 1,
            ')' | ']' => depth -= 1,
            ',' | '}' if depth == 0 => {
                end = i;
                break;
            }
            _ => {}
        }
    }
    Some(rest[..end].trim().to_string())
}

fn parse_shape(s: &str) -> Result<Vec<usize>, Error> {
    let inner = s.trim().trim_start_matches('(').trim_end_matches(')');
    inner
        .split(',')
        .map(str::trim)
        .filter(|p| !p.is_empty())
        .map(|p| p.parse::<usize>().map_err(|_| Error::Format(format!("bad shape {s}"))))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn npy(descr: &str, shape: &str, body: &[u8]) -> Vec<u8> {
        let mut header = format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape}, }}");
        while (10 + header.len() + 1) % 64 != 0 {
            header.push(' ');
        }
        header.push('\n');
        let mut out = b"\x93NUMPY\x01\x00".to_vec();
        out.extend((header.len() as u16).to_le_bytes());
        out.extend(header.as_bytes());
        out.extend(body);
        out
    }

    #[test]
    fn reads_numeric_and_string_arrays() {
        let a = parse_npy(&npy("<i2", "(3,)", &[1, 0, 255, 255, 2, 0])).unwrap();
        assert_eq!(a.into_i16().unwrap(), vec![1, -1, 2]);
        let f = parse_npy(&npy("<f4", "(2, 1)", &[0, 0, 128, 63, 0, 0, 0, 64])).unwrap();
        assert_eq!(f.shape, vec![2, 1]);
        assert_eq!(f.into_f32().unwrap(), vec![1.0, 2.0]);
        let mut s = Vec::new();
        for text in ["ab", "c"] {
            let mut chars: Vec<u32> = text.chars().map(|c| c as u32).collect();
            chars.resize(3, 0);
            for c in chars {
                s.extend(c.to_le_bytes());
            }
        }
        let a = parse_npy(&npy("<U3", "(2,)", &s)).unwrap();
        assert_eq!(a.into_strings().unwrap(), vec!["ab".to_string(), "c".to_string()]);
    }

    #[test]
    fn zero_dim_string() {
        let mut b = Vec::new();
        for c in "adult".chars() {
            b.extend((c as u32).to_le_bytes());
        }
        let a = parse_npy(&npy("<U5", "()", &b)).unwrap();
        assert_eq!(a.into_strings().unwrap(), vec!["adult".to_string()]);
    }

    #[test]
    fn rejects_garbage() {
        assert!(parse_npy(b"nope").is_err());
        assert!(parse_npy(&npy("<f4", "(4,)", &[0; 4])).is_err());
    }
}
