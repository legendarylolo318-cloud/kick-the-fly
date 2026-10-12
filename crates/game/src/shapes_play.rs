//! Real neuron shapes in the game (`sim/realshapes.py` `ShapeLoader`, `game/shape_draw.py`): the inspected neuron's reconstructed skeleton from
//! the public MaleCNS v1.0 release, drawn over the big brain view and as a thumbnail on the inspector's card.
//!
//! Rules (all kept from Python): OFF until the player opts in (Settings > Brain > Download real neuron shapes); cached shapes load whether
//! or not it is on; `KICK_THE_FLY_OFFLINE=1` never fetches; a download is accepted only if the server's md5 (the `x-goog-hash` header)
//! matches the bytes; at most 5 MB and 400,000 nodes a file; one request at a time with a 0.25 s pause; one attempt a session per
//! neuron that fails; cached under the data folder's `shapes/`. Anything that goes wrong falls back to the estimated fiber. The simulation
//! never sees a shape: every neuron is still a single point.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use base64::Engine;
use bevy::prelude::*;
use ktf_data::shapes::{self, BASE_URL, MAX_BYTES, ShapeStore, Skeleton};
use md5::{Digest, Md5};

const RATE_LIMIT_S: f64 = 0.25;
const TIMEOUT_S: u64 = 8;

#[derive(Clone)]
pub enum ShapeState {
    Loading,
    Ready(Arc<Skeleton>, &'static str),
    /// Not allowed (not opted in, offline).
    Off(String),
    Failed(String),
}

#[derive(Resource, Clone, Default)]
pub struct Shapes(Arc<Mutex<HashMap<i64, ShapeState>>>);

fn last_request() -> &'static Mutex<Option<Instant>> {
    static L: OnceLock<Mutex<Option<Instant>>> = OnceLock::new();
    L.get_or_init(Default::default)
}

fn folder() -> std::path::PathBuf {
    ktf_data::paths::get().data_dir.join("shapes")
}

/// `realshapes.network_allowed`.
pub fn network_allowed(opted_in: bool) -> Result<(), String> {
    if !opted_in {
        return Err("real neuron shapes are off (Settings > Brain > Download real neuron shapes)".into());
    }
    if matches!(std::env::var("KICK_THE_FLY_OFFLINE").as_deref(), Ok("1" | "true" | "yes")) {
        return Err("KICK_THE_FLY_OFFLINE is set".into());
    }
    let (ok, why) = crate::selftest::network_allowed();
    if ok { Ok(()) } else { Err(why) }
}

/// Download, verify and cache one skeleton. The error says why in words.
fn fetch(store: &mut ShapeStore, body_id: i64, base_url: &str) -> Result<Skeleton, String> {
    {
        let mut last = last_request().lock().unwrap();
        if let Some(t) = *last {
            let wait = RATE_LIMIT_S - t.elapsed().as_secs_f64();
            if wait > 0.0 {
                std::thread::sleep(Duration::from_secs_f64(wait));
            }
        }
        *last = Some(Instant::now());
    }
    let url = format!("{}/{body_id}.swc", base_url.trim_end_matches('/'));
    let agent: ureq::Agent = ureq::Agent::config_builder().timeout_global(Some(Duration::from_secs(TIMEOUT_S))).build().into();
    let mut resp = match agent
        .get(&url)
        .header("User-Agent", format!("KickTheFly/{} (+https://github.com/legendarylolo318-cloud/kick-the-fly)", env!("CARGO_PKG_VERSION")))
        .call()
    {
        Ok(r) => r,
        Err(ureq::Error::StatusCode(404)) => return Err("no published skeleton (HTTP 404)".into()),
        Err(ureq::Error::StatusCode(c)) => return Err(format!("HTTP {c}")),
        Err(e) => return Err(format!("could not reach the server ({e})")),
    };
    // Google Cloud Storage sends the checksums as separate `x-goog-hash` headers (crc32c=..., md5=...); a single header with both separated
    // by commas is read too. (Python read only the first header, so against the real bucket it found no md5 and refused every file.)
    let want_md5 = resp
        .headers()
        .get_all("x-goog-hash")
        .iter()
        .filter_map(|h| h.to_str().ok())
        .flat_map(|raw| raw.split(',').map(|p| p.trim().to_string()).collect::<Vec<_>>())
        .find_map(|p| p.strip_prefix("md5=").map(str::to_string))
        .and_then(|b| base64::engine::general_purpose::STANDARD.decode(b).ok());
    let data =
        resp.body_mut().with_config().limit(MAX_BYTES as u64 + 1).read_to_vec().map_err(|e| format!("could not read the file ({e})"))?;
    if data.len() > MAX_BYTES {
        return Err(format!("over the {MAX_BYTES} byte limit"));
    }
    let Some(want) = want_md5 else { return Err("the server offered no checksum, so the file was refused".into()) };
    if Md5::digest(&data).as_slice() != want.as_slice() {
        return Err("the download did not match the server's md5 checksum, so it was thrown away".into());
    }
    let b64 = base64::engine::general_purpose::STANDARD.encode(&want);
    let stamp = ktf_data::diag::timestamp(ktf_data::diag::now_secs());
    store.put(body_id, &data, &b64, &url, &stamp).map_err(|e| format!("the file is not a usable skeleton ({e})"))
}

impl Shapes {
    /// Start getting a shape: the cache at once, else (opted in, allowed) a download on a thread of its own.
    pub fn request(&self, body_id: i64, opted_in: bool) {
        if body_id <= 0 {
            return;
        }
        {
            let mut m = self.0.lock().unwrap();
            if m.contains_key(&body_id) {
                return;
            }
            m.insert(body_id, ShapeState::Loading);
        }
        let me = self.0.clone();
        let base = std::env::var("KICK_THE_FLY_SHAPE_URL").unwrap_or_else(|_| BASE_URL.to_string());
        let set = move |s: ShapeState| {
            me.lock().unwrap().insert(body_id, s);
        };
        let _ = std::thread::Builder::new().name("real-shape".into()).spawn(move || {
            let mut store = ShapeStore::open(&folder());
            if let Some(s) = store.load(body_id) {
                return set(ShapeState::Ready(Arc::new(s), "cached"));
            }
            if let Err(why) = network_allowed(opted_in) {
                return set(ShapeState::Off(why));
            }
            match fetch(&mut store, body_id, &base) {
                Ok(s) => set(ShapeState::Ready(Arc::new(s), "downloaded")),
                Err(e) => set(ShapeState::Failed(e)),
            }
        });
    }

    pub fn state(&self, body_id: i64) -> Option<ShapeState> {
        self.0.lock().unwrap().get(&body_id).cloned()
    }

    /// After the player turns the download on: ask again for what was refused.
    pub fn reset_offline(&self) {
        self.0.lock().unwrap().retain(|_, v| !matches!(v, ShapeState::Off(_) | ShapeState::Failed(_)));
    }
}

/// The text of the inspector's line for this state (`Game._real_shape`).
pub fn status_line(state: Option<&ShapeState>, opted_in: bool) -> String {
    match state {
        None | Some(ShapeState::Loading) => "estimated fiber (downloading the real shape...)".into(),
        Some(ShapeState::Off(_)) if !opted_in => "estimated fiber (real shapes are off: Settings > Brain)".into(),
        Some(ShapeState::Off(m) | ShapeState::Failed(m)) => format!("estimated fiber ({})", m.chars().take(44).collect::<String>()),
        Some(ShapeState::Ready(s, _)) => format!(
            "real shape: {} nodes, {} um of cable",
            ktf_data::diag::group3(s.n()),
            ktf_data::diag::group3(s.cable_length_um().round() as usize)
        ),
    }
}

pub fn skeleton_of(state: Option<&ShapeState>) -> Option<Arc<Skeleton>> {
    match state {
        Some(ShapeState::Ready(s, _)) => Some(s.clone()),
        _ => None,
    }
}

/// The shape's own gizmo group: lines on the brain camera's layer, so they sit in the cloud and turn with it.
#[derive(Default, Reflect, GizmoConfigGroup)]
pub struct ShapeGizmos;

/// The inspected neuron's skeleton over the big view: its nodes in layout space (`layout()`'s affine of the EM coordinates), turned and
/// scaled exactly as the brain shader does, in the shape colour (1.0, 0.93, 0.55).
fn draw_overlay(
    insp: Res<crate::inspector_play::Inspector>,
    shapes: Res<Shapes>,
    view: Res<crate::brainview::BrainView>,
    mut gizmos: Gizmos<ShapeGizmos>,
) {
    if !view.big {
        return;
    }
    let Some(info) = insp.info.as_ref() else { return };
    let Some(skel) = skeleton_of(shapes.state(info.body_id).as_ref()) else { return };
    let to_view = |p: [f32; 3]| -> Vec3 {
        view.turn(Vec3::new((p[0] - 48_000.0) / 48_000.0, (35_000.0 - p[1]) / 48_000.0, (p[2] - 65_000.0) / 48_000.0))
    };
    let col = Color::srgb(1.0, 0.93, 0.55);
    for (a, b) in skel.segments() {
        gizmos.line(to_view(a), to_view(b), col);
    }
}

pub struct ShapesPlugin;

impl Plugin for ShapesPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Shapes>().init_gizmo_group::<ShapeGizmos>().add_systems(Startup, |mut store: ResMut<GizmoConfigStore>| {
            let (cfg, _) = store.config_mut::<ShapeGizmos>();
            cfg.render_layers = bevy::camera::visibility::RenderLayers::layer(crate::brainview::BRAIN_LAYER);
            cfg.line.width = 1.5;
        });
        app.add_systems(Update, draw_overlay);
        let _ = shapes::CREDIT;
    }
}

#[cfg(test)]
mod tests {
    use std::io::{Read, Write};
    use std::net::TcpListener;

    use super::*;

    const SWC: &str = "# t\n1 1 0 0 0 1 -1\n2 3 10 0 0 1 1\n3 3 10 10 0 1 2\n";

    /// A one-shot HTTP server for `n` requests: (status line, extra headers, body) per path.
    fn serve(routes: Vec<(String, u16, Vec<(String, String)>, Vec<u8>)>) -> String {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = l.local_addr().unwrap();
        std::thread::spawn(move || {
            for _ in 0..routes.len() {
                let Ok((mut s, _)) = l.accept() else { return };
                let mut buf = [0u8; 2048];
                let n = s.read(&mut buf).unwrap_or(0);
                let req = String::from_utf8_lossy(&buf[..n]).to_string();
                let path = req.split_whitespace().nth(1).unwrap_or("").to_string();
                let r = routes.iter().find(|r| r.0 == path);
                let (code, hdrs, body) = r.map_or((404, vec![], vec![]), |r| (r.1, r.2.clone(), r.3.clone()));
                let mut out = format!("HTTP/1.1 {code} X\r\nContent-Length: {}\r\nConnection: close\r\n", body.len());
                for (k, v) in hdrs {
                    out += &format!("{k}: {v}\r\n");
                }
                out += "\r\n";
                let _ = s.write_all(out.as_bytes());
                let _ = s.write_all(&body);
            }
        });
        format!("http://{addr}")
    }

    fn goog(data: &[u8]) -> (String, String) {
        ("x-goog-hash".into(), format!("md5={}", base64::engine::general_purpose::STANDARD.encode(Md5::digest(data))))
    }

    #[test]
    fn a_checked_download_is_cached_and_the_unchecked_are_refused() {
        let d = std::env::temp_dir().join(format!("ktf-fetch-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        let mut store = ShapeStore::open(&d);
        let base = serve(vec![
            ("/11.swc".into(), 200, vec![("x-goog-hash".into(), "crc32c=rsCDbw==".into()), goog(SWC.as_bytes())], SWC.as_bytes().to_vec()),
            ("/12.swc".into(), 200, vec![], SWC.as_bytes().to_vec()),
            ("/13.swc".into(), 200, vec![goog(b"something else")], SWC.as_bytes().to_vec()),
            ("/14.swc".into(), 404, vec![], vec![]),
        ]);
        let s = fetch(&mut store, 11, &base).unwrap();
        assert_eq!(s.n(), 3);
        assert_eq!(store.load(11).unwrap().n(), 3, "it is in the cache now");
        assert!(fetch(&mut store, 12, &base).unwrap_err().contains("no checksum"));
        assert!(fetch(&mut store, 13, &base).unwrap_err().contains("md5"));
        assert!(fetch(&mut store, 14, &base).unwrap_err().contains("404"));
        assert!(store.load(12).is_none() && store.load(13).is_none(), "nothing refused was kept");
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn the_download_is_off_until_opted_in() {
        assert!(network_allowed(false).unwrap_err().contains("Settings > Brain"));
        assert!(status_line(Some(&ShapeState::Off("x".into())), false).contains("real shapes are off"));
    }
}
