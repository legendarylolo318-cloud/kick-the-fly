//! The tool in your hand (Python: `Game3D.draw_viewmodel`, `game/kick3d.py`): a small model in camera space (x right, y up, -z forward)
//! that bobs as you walk and moves when the tool is used (the swatter's swing, the flicking finger, the thrown bomb / sugar / fruit that
//! is missing from the hand for a moment, the zapper's ring lighting up, the laser's LED colour).
//!
//! [`prims`] is the port of the Python drawing, shape for shape; the systems below only put those shapes on screen. Python drew them
//! with its own 60 degree lens squeezed into the front of the depth range. Here they hang on the player's camera: x and y are stretched
//! by `tan(fov / 2) / tan(30 deg)` (the factor `Game3D.tool_tip` uses), which gives the same picture under the game's field of view, and
//! the whole model is shrunk toward the eye (the picture does not change) so it never sinks into a wall.

use std::collections::HashMap;

use bevy::light::{NotShadowCaster, NotShadowReceiver};
use bevy::prelude::*;

use crate::config::Settings;
use crate::fly::Fly;
use crate::player::{CameraOverride, Player, PlayerCamera};
use crate::tools::{TOOLS, ToolState};

/// `VIEW_BASE`: where the hand sits in camera space.
pub const VIEW_BASE: Vec3 = Vec3::new(0.27, -0.26, -0.55);
/// `VIEWMODEL_FOV`, in degrees.
pub const VIEWMODEL_FOV: f32 = 60.0;
/// The model is drawn this much nearer to the eye (and smaller by the same factor, so it looks the same).
const SHRINK: f32 = 0.2;
const SKIN: [f32; 4] = [0.93, 0.74, 0.6, 1.0];

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Shape {
    /// Radius 1.
    Sphere,
    /// Radius 1, one unit long along y, centred.
    Cylinder,
    /// One unit a side (`mesh_cube`: corners at +-0.5).
    Cube,
    /// In the xz plane: ring radius 1, tube radius 0.18 (`mesh_torus`).
    Torus,
}

/// One shape of the model, in camera space.
#[derive(Clone, Copy, Debug)]
pub struct Prim {
    pub shape: Shape,
    pub at: Transform,
    pub color: [f32; 4],
    /// Python's glow factor (0: lit only).
    pub glow: f32,
}

/// What the drawing depends on besides the tool.
#[derive(Clone, Copy, Debug)]
pub struct ViewState {
    pub now: f32,
    /// `Player.walk_phase`.
    pub walk_phase: f32,
    pub grabbed: bool,
    pub flick_t: f32,
    pub swing_t: f32,
    pub throw_t: f32,
    pub zap_ready: f32,
    pub laser_activate: bool,
}

fn sphere(out: &mut Vec<Prim>, pos: Vec3, radii: Vec3, color: [f32; 4]) {
    out.push(Prim { shape: Shape::Sphere, at: Transform::from_translation(pos).with_scale(radii), color, glow: 0.0 });
}

/// `render3d.segment`: the cylinder from `a` to `b`.
fn segment(out: &mut Vec<Prim>, a: Vec3, b: Vec3, radius: f32, color: [f32; 4]) {
    let d = b - a;
    let len = d.length().max(1e-6);
    let rot = if d.length() < 1e-9 { Quat::IDENTITY } else { Quat::from_rotation_arc(Vec3::Y, d / len) };
    out.push(Prim {
        shape: Shape::Cylinder,
        at: Transform { translation: (a + b) / 2.0, rotation: rot, scale: Vec3::new(radius, len, radius) },
        color,
        glow: 0.0,
    });
}

/// The hand: a palm, four fingers (curled by `curl`; with `point`, the second one points and snaps forward by `flick`) and a thumb.
fn hand(out: &mut Vec<Prim>, pos: Vec3, curl: f32, point: bool, flick: f32) {
    sphere(out, pos, Vec3::new(0.045, 0.028, 0.06), SKIN);
    for (j, dx) in [-0.03f32, -0.01, 0.01, 0.03].into_iter().enumerate() {
        let tip = if point && j == 1 {
            let ang = -0.2 - 1.2 * flick;
            pos + Vec3::new(dx, 0.01 + 0.08 * (-ang).sin() * 0.4, -0.05 - 0.07 * ang.cos())
        } else {
            pos + Vec3::new(dx, -0.02 * curl, -0.05 - 0.05 * (1.0 - curl))
        };
        segment(out, pos + Vec3::new(dx, 0.0, -0.04), tip, 0.009, SKIN);
    }
    segment(out, pos + Vec3::new(-0.04, 0.0, -0.01), pos + Vec3::new(-0.07, 0.01, -0.05), 0.01, SKIN);
}

/// The swatter's angle `ph` seconds after a swing began: wound back, through in 0.12 s, then easing home by 0.42 s.
pub fn swing_angle(ph: f32) -> f32 {
    if ph < 0.12 {
        -0.9 * (1.0 - ph / 0.12) + 0.2
    } else if ph < 0.42 {
        0.2 - 0.2 * ((ph - 0.12) / 0.3).min(1.0)
    } else {
        0.0
    }
}

/// `draw_viewmodel`: the shapes of the held tool. Tools Python draws nothing for (alcohol, cVA, decoy) give an empty list.
pub fn prims(tool: &str, s: &ViewState) -> Vec<Prim> {
    let mut out = Vec::new();
    let o = &mut out;
    let base = VIEW_BASE + Vec3::new(0.0, 0.012 * (s.walk_phase * 2.0).sin(), 0.0);
    let v = Vec3::new;
    let rgb = |r: f32, g: f32, b: f32| [r, g, b, 1.0];
    match tool {
        "hand" => hand(o, base + v(0.0, 0.02, 0.0), if s.grabbed { 0.9 } else { 0.1 }, false, 0.0),
        "flick" => hand(o, base + v(0.0, 0.02, 0.0), 0.9, true, (1.0 - (s.now - s.flick_t) / 0.15).max(0.0)),
        "swatter" => {
            let r = Quat::from_rotation_x(-swing_angle(s.now - s.swing_t) - 0.3);
            let pivot = base + v(0.0, -0.05, 0.05);
            let handle_end = pivot + r * v(0.0, 0.32, -0.12);
            segment(o, pivot, handle_end, 0.012, rgb(0.5, 0.33, 0.18));
            let head = handle_end + r * v(0.0, 0.1, -0.03);
            o.push(Prim {
                shape: Shape::Cube,
                at: Transform { translation: head, rotation: r, scale: v(0.2, 0.22, 0.012) },
                color: [0.82, 0.16, 0.18, 0.9],
                glow: 0.0,
            });
            hand(o, pivot + v(0.0, 0.02, 0.02), 1.0, false, 0.0);
        }
        "bomb" => {
            if s.now - s.throw_t > 0.4 {
                sphere(o, base + v(0.0, 0.05, -0.05), Vec3::splat(0.07), rgb(0.12, 0.12, 0.14));
            }
            hand(o, base, 0.7, false, 0.0);
        }
        "torch" => {
            segment(o, base + v(0.0, 0.0, 0.05), base + v(0.0, 0.06, -0.2), 0.035, rgb(0.75, 0.2, 0.18));
            segment(o, base + v(0.0, 0.06, -0.2), base + v(0.0, 0.07, -0.26), 0.012, rgb(0.6, 0.6, 0.65));
            hand(o, base + v(0.0, -0.02, 0.04), 1.0, false, 0.0);
        }
        "cleaner" | "freeze" => {
            let body = if tool == "cleaner" { rgb(0.82, 0.14, 0.14) } else { rgb(0.25, 0.5, 0.85) };
            segment(o, base + v(0.0, -0.05, 0.0), base + v(0.0, 0.12, -0.02), 0.04, body);
            segment(o, base + v(0.0, 0.02, -0.01), base + v(0.0, 0.06, -0.012), 0.041, rgb(0.95, 0.95, 0.96));
            segment(o, base + v(0.0, 0.12, -0.02), base + v(0.0, 0.15, -0.06), 0.012, rgb(0.2, 0.2, 0.22));
            hand(o, base + v(0.0, -0.03, 0.03), 1.0, false, 0.0);
        }
        "zapper" => {
            segment(o, base, base + v(0.0, 0.12, -0.12), 0.014, rgb(0.2, 0.2, 0.25));
            let lit = s.now - s.zap_ready > -0.3 && s.now < s.zap_ready;
            o.push(Prim {
                shape: Shape::Torus,
                at: Transform { translation: base + v(0.0, 0.2, -0.2), rotation: Quat::from_rotation_x(-0.7), scale: Vec3::splat(0.09) },
                color: rgb(0.95, 0.85, 0.2),
                glow: if lit { 2.0 } else { 0.3 },
            });
            hand(o, base + v(0.0, 0.0, 0.02), 1.0, false, 0.0);
        }
        "spider" => {
            sphere(o, base + v(0.0, 0.06, -0.05), v(0.04, 0.035, 0.05), rgb(0.12, 0.11, 0.13));
            hand(o, base, 0.6, false, 0.0);
        }
        "frog" => {
            sphere(o, base + v(0.0, 0.06, -0.05), v(0.05, 0.035, 0.055), rgb(0.25, 0.5, 0.2));
            sphere(o, base + v(0.02, 0.095, -0.075), Vec3::splat(0.012), rgb(0.95, 0.85, 0.3));
            sphere(o, base + v(-0.02, 0.095, -0.075), Vec3::splat(0.012), rgb(0.95, 0.85, 0.3));
            hand(o, base, 0.6, false, 0.0);
        }
        "dragonfly" => {
            segment(o, base + v(0.0, 0.07, 0.0), base + v(0.0, 0.07, -0.12), 0.007, rgb(0.15, 0.45, 0.75));
            segment(o, base + v(-0.07, 0.085, -0.04), base + v(0.07, 0.085, -0.04), 0.004, rgb(0.85, 0.9, 0.95));
            hand(o, base, 0.6, false, 0.0);
        }
        "mantis" => {
            segment(o, base + v(0.0, 0.05, 0.0), base + v(0.0, 0.1, -0.1), 0.012, rgb(0.45, 0.62, 0.22));
            sphere(o, base + v(0.0, 0.105, -0.11), v(0.018, 0.016, 0.016), rgb(0.3, 0.45, 0.15));
            hand(o, base, 0.6, false, 0.0);
        }
        "sugar" => {
            if s.now - s.throw_t > 0.3 {
                o.push(Prim {
                    shape: Shape::Cube,
                    at: Transform {
                        translation: base + v(0.0, 0.05, -0.05),
                        rotation: Quat::from_rotation_y(0.5),
                        scale: Vec3::splat(0.055),
                    },
                    color: rgb(0.98, 0.98, 1.0),
                    glow: 0.0,
                });
            }
            hand(o, base, 0.6, false, 0.0);
        }
        "fruit" => {
            if s.now - s.throw_t > 0.3 {
                sphere(o, base + v(0.0, 0.05, -0.05), Vec3::splat(0.05), rgb(0.84, 0.25, 0.19));
            }
            hand(o, base, 0.6, false, 0.0);
        }
        "laser" => {
            segment(o, base + v(0.0, 0.0, 0.05), base + v(0.0, 0.05, -0.22), 0.025, rgb(0.2, 0.22, 0.26));
            segment(o, base + v(0.0, 0.05, -0.22), base + v(0.0, 0.06, -0.28), 0.015, rgb(0.7, 0.75, 0.8));
            let led = if s.laser_activate { rgb(1.0, 0.5, 0.1) } else { rgb(0.2, 0.7, 1.0) };
            sphere(o, base + v(0.0, 0.062, -0.285), Vec3::splat(0.008), led);
            hand(o, base + v(0.0, -0.02, 0.04), 1.0, false, 0.0);
        }
        _ => {}
    }
    out
}

#[derive(Component)]
struct ViewRoot;

/// The n-th shape of the model (a pool: shapes are re-pointed every frame, the unused ones hidden).
#[derive(Component)]
struct ViewPart(usize);

/// The most shapes a tool has (the swatter: handle, head and a six-piece hand).
const POOL: usize = 12;

#[derive(Resource)]
struct ViewAssets {
    meshes: HashMap<Shape, Handle<Mesh>>,
    materials: HashMap<[u8; 5], Handle<StandardMaterial>>,
}

fn setup(mut commands: Commands, mut meshes: ResMut<Assets<Mesh>>) {
    let mut map = HashMap::new();
    map.insert(Shape::Sphere, meshes.add(Sphere::new(1.0).mesh().uv(22, 14)));
    map.insert(Shape::Cylinder, meshes.add(Cylinder::new(1.0, 1.0).mesh().resolution(16)));
    map.insert(Shape::Cube, meshes.add(Cuboid::new(1.0, 1.0, 1.0)));
    map.insert(Shape::Torus, meshes.add(Torus::new(0.82, 1.18)));
    let sphere = map[&Shape::Sphere].clone();
    commands.insert_resource(ViewAssets { meshes: map, materials: HashMap::new() });
    commands.spawn((ViewRoot, Transform::default(), Visibility::Hidden)).with_children(|root| {
        for i in 0..POOL {
            root.spawn((
                ViewPart(i),
                Mesh3d(sphere.clone()),
                MeshMaterial3d::<StandardMaterial>(Handle::default()),
                Transform::default(),
                Visibility::Hidden,
                NotShadowCaster,
                NotShadowReceiver,
            ));
        }
    });
}

#[allow(clippy::too_many_arguments)]
fn draw(
    virt: Res<Time<Virtual>>,
    st: Res<ToolState>,
    settings: Res<Settings>,
    player: Single<&Player>,
    cam: Single<&Transform, (With<PlayerCamera>, Without<ViewRoot>, Without<ViewPart>)>,
    flies: Query<&Fly>,
    photo: Option<Res<crate::photo_play::Photo>>,
    eye: Option<Res<crate::flyeye::FlyEye>>,
    over: Option<Res<CameraOverride>>,
    mut root: Single<(&mut Transform, &mut Visibility), (With<ViewRoot>, Without<ViewPart>)>,
    mut parts: Query<(&ViewPart, &mut Transform, &mut Visibility, &mut Mesh3d, &mut MeshMaterial3d<StandardMaterial>), Without<ViewRoot>>,
    mut assets: ResMut<ViewAssets>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // not in photo mode, not through the fly's eyes (`Game3D.render`), and not while a scripted camera stands somewhere else
    let hidden = photo.is_some_and(|p| p.active) || eye.is_some_and(|e| e.on) || over.is_some();
    *root.1 = if hidden { Visibility::Hidden } else { Visibility::Visible };
    if hidden {
        return;
    }
    let k = (settings.fov_deg.to_radians() / 2.0).tan() / (VIEWMODEL_FOV.to_radians() / 2.0).tan();
    *root.0 = Transform { translation: cam.translation, rotation: cam.rotation, scale: Vec3::new(k * SHRINK, k * SHRINK, SHRINK) };
    let state = ViewState {
        now: virt.elapsed_secs(),
        walk_phase: player.walk_phase,
        grabbed: flies.iter().any(|f| f.grabbed.is_some()),
        flick_t: st.flick_t,
        swing_t: st.swing_t,
        throw_t: st.throw_t,
        zap_ready: st.zap_ready,
        laser_activate: st.laser.mode == "activate",
    };
    let list = prims(TOOLS[st.selected.min(TOOLS.len() - 1)].0, &state);
    for (part, mut tf, mut vis, mut mesh, mut mat) in &mut parts {
        let Some(p) = list.get(part.0) else {
            *vis = Visibility::Hidden;
            continue;
        };
        *vis = Visibility::Inherited;
        *tf = p.at;
        let want = &assets.meshes[&p.shape];
        if mesh.0 != *want {
            mesh.0 = want.clone();
        }
        let key = [
            (p.color[0] * 255.0) as u8,
            (p.color[1] * 255.0) as u8,
            (p.color[2] * 255.0) as u8,
            (p.color[3] * 255.0) as u8,
            (p.glow * 100.0) as u8,
        ];
        let handle = assets.materials.entry(key).or_insert_with(|| {
            let [r, g, b, a] = p.color;
            materials.add(StandardMaterial {
                base_color: Color::srgba(r, g, b, a),
                emissive: LinearRgba::from(Color::srgb(r, g, b)) * p.glow,
                perceptual_roughness: 0.6,
                alpha_mode: if a < 1.0 { AlphaMode::Blend } else { AlphaMode::Opaque },
                ..default()
            })
        });
        if mat.0 != *handle {
            mat.0 = handle.clone();
        }
    }
}

pub struct ViewModelPlugin;

impl Plugin for ViewModelPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, setup).add_systems(Update, draw.after(crate::player::update_camera));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state(now: f32) -> ViewState {
        ViewState {
            now,
            walk_phase: 0.0,
            grabbed: false,
            flick_t: -9.0,
            swing_t: -9.0,
            throw_t: -9.0,
            zap_ready: 0.0,
            laser_activate: true,
        }
    }

    #[test]
    fn every_python_tool_has_its_shapes() {
        // the number of `rd.add` calls of each branch of `draw_viewmodel` (the hand is 6)
        let want = [
            ("hand", 6),
            ("flick", 6),
            ("swatter", 8),
            ("bomb", 7),
            ("torch", 8),
            ("cleaner", 9),
            ("zapper", 8),
            ("freeze", 9),
            ("spider", 7),
            ("sugar", 7),
            ("alcohol", 0),
            ("laser", 9),
            ("cva", 0),
            ("decoy", 0),
            ("fruit", 7),
            ("frog", 9),
            ("dragonfly", 8),
            ("mantis", 8),
        ];
        assert_eq!(want.len(), TOOLS.len());
        for ((tool, n), t) in want.iter().zip(TOOLS.iter()) {
            assert_eq!(*tool, t.0);
            let list = prims(tool, &state(10.0));
            assert_eq!(list.len(), *n, "{tool}");
            assert!(list.len() <= POOL);
            assert!(list.iter().all(|p| p.at.translation.is_finite() && p.at.scale.min_element() > 0.0), "{tool}");
        }
    }

    #[test]
    fn the_swatter_swings_like_python() {
        // `ang` of `draw_viewmodel` at a few phases
        for (ph, want) in [(0.0, -0.7), (0.06, -0.25), (0.12, 0.2), (0.27, 0.1), (0.42, 0.0), (3.0, 0.0)] {
            assert!((swing_angle(ph) - want).abs() < 1e-5, "{ph}");
        }
        // at rest the head sits where Python puts it: pivot + R (0, 0.32, -0.12) + R (0, 0.1, -0.03), R = rot_x(-0.3)
        let list = prims("swatter", &state(10.0));
        let (s, c) = (-0.3f32).sin_cos();
        let rot = |y: f32, z: f32| Vec3::new(0.0, c * y - s * z, s * y + c * z);
        let head = VIEW_BASE + Vec3::new(0.0, -0.05, 0.05) + rot(0.32, -0.12) + rot(0.1, -0.03);
        assert!((list[1].at.translation - head).length() < 1e-5);
        assert_eq!(list[1].shape, Shape::Cube);
        // mid-swing it is somewhere else
        let mut st = state(10.0);
        st.swing_t = 9.95;
        assert!((prims("swatter", &st)[1].at.translation - head).length() > 0.05);
    }

    #[test]
    fn thrown_things_leave_the_hand_and_the_finger_flicks() {
        let mut st = state(10.0);
        st.throw_t = 9.8;
        assert_eq!(prims("bomb", &st).len(), 6);
        assert_eq!(prims("sugar", &st).len(), 6);
        assert_eq!(prims("fruit", &st).len(), 6);
        st.throw_t = 9.5;
        assert_eq!(prims("bomb", &st).len(), 7);
        // the pointing finger (the third shape: palm, finger 0, finger 1) reaches further forward while it flicks
        let rest = prims("flick", &st)[2].at;
        st.flick_t = 10.0;
        let snap = prims("flick", &st)[2].at;
        assert!(snap.translation.y > rest.translation.y + 0.005);
        // the zapper's ring glows for 0.3 s after a zap
        st.zap_ready = 10.2;
        assert_eq!(prims("zapper", &st)[1].glow, 2.0);
        st.zap_ready = 9.0;
        assert_eq!(prims("zapper", &st)[1].glow, 0.3);
    }
}
