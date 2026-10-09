//! Things in the world that are neither arena geometry nor tools: the cook, the orchard's fruit and the bowl's fruit, drawn each frame
//! from the state in `environment.rs` with the same primitive pool the predators use.

use bevy::prelude::*;

use crate::environment::{Fruit, Kitchen};
use crate::kitchen::{self as kit, CookState};
use crate::player::Player;
use crate::predator_play::{PaintBox, Prim, PrimPool, PrimTag, draw_prims};

#[derive(Resource, Default)]
struct PropPools {
    cook: PrimPool,
    fruit: PrimPool,
}

/// The cook: a torso, a head with a hat, one arm raised to a swatter that follows the engine's hand (so the swing the fly sees grow is the
/// swing you see).
pub fn layout_cook(c: &kit::Cook) -> Vec<Prim> {
    let (white, skin, dark) = ([0.92, 0.92, 0.9], [0.86, 0.66, 0.52], [0.2, 0.2, 0.22]);
    let base = Vec3::new(c.x, 0.0, kit::COOK_Z - 0.1);
    let mut out = vec![
        Prim::Ell { c: base + Vec3::new(0.0, 1.2, 0.0), fwd: Vec3::Y, size: Vec3::new(0.65, 0.4, 0.32), col: white },
        Prim::Ell { c: base + Vec3::new(0.0, 1.95, 0.0), fwd: Vec3::X, size: Vec3::new(0.2, 0.2, 0.2), col: skin },
        Prim::Ell { c: base + Vec3::new(0.0, 2.2, 0.0), fwd: Vec3::Y, size: Vec3::new(0.22, 0.19, 0.19), col: white },
        Prim::Limb { a: base + Vec3::new(-0.2, 0.2, 0.0), b: base + Vec3::new(-0.2, 0.9, 0.0), r: 0.11, col: dark },
        Prim::Limb { a: base + Vec3::new(0.2, 0.2, 0.0), b: base + Vec3::new(0.2, 0.9, 0.0), r: 0.11, col: dark },
        Prim::Bead { c: base + Vec3::new(0.07, 2.0, 0.17), r: 0.03, col: [0.05, 0.05, 0.05] },
        Prim::Bead { c: base + Vec3::new(-0.07, 2.0, 0.17), r: 0.03, col: [0.05, 0.05, 0.05] },
    ];
    let shoulder = base + Vec3::new(0.35, 1.55, 0.1);
    let hand = c.pos;
    let elbow = (shoulder + hand) / 2.0 + Vec3::new(0.15, 0.1, 0.1);
    out.push(Prim::Limb { a: shoulder, b: elbow, r: 0.09, col: white });
    out.push(Prim::Limb { a: elbow, b: hand, r: 0.07, col: skin });
    let raised = matches!(c.state, CookState::Windup | CookState::Swat | CookState::Recover);
    // the swatter: a handle and a flat head the size the fly's eyes measure (COOK_SWATTER_R)
    let up = if raised { Vec3::Y } else { Vec3::new(0.3, 1.0, 0.0).normalize() };
    out.push(Prim::Limb { a: hand, b: hand + up * 0.25, r: 0.015, col: [0.5, 0.35, 0.2] });
    out.push(Prim::Ell {
        c: hand + up * 0.42,
        fwd: Vec3::X,
        size: Vec3::new(kit::COOK_SWATTER_R * 0.5, 0.012, kit::COOK_SWATTER_R * 0.5),
        col: [0.9, 0.35, 0.25],
    });
    out
}

fn draw(
    mut commands: Commands,
    mut pools: Local<PropPools>,
    kitchen: Res<Kitchen>,
    fruit: Res<Fruit>,
    mut paint: ResMut<PaintBox>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut tfs: Query<(&mut Transform, &mut Visibility, &mut MeshMaterial3d<StandardMaterial>), With<PrimTag>>,
    player: Single<&Transform, (With<Player>, Without<PrimTag>)>,
) {
    let cook = kitchen.0.as_ref().map(|k| layout_cook(&k.cook)).unwrap_or_default();
    draw_prims(&mut commands, &mut pools.cook, &cook, &mut paint, &mut meshes, &mut materials, &mut tfs);
    let me = player.translation;
    let mut prims = Vec::new();
    if let Some(o) = fruit.0.as_ref() {
        for f in o.fruit.iter().filter(|f| f.ripe() && (f.pos - me).length() < 38.0) {
            let full = 0.4 + 0.6 * f.fullness();
            let col = if f.fermented { [0.55, 0.2, 0.45] } else { [0.85, 0.2, 0.15] };
            let r = 0.09 * full;
            prims.push(Prim::Ell { c: f.pos, fwd: Vec3::Y, size: Vec3::new(r, r * 0.95, r), col });
            prims.push(Prim::Limb { a: f.pos + Vec3::Y * r * 0.8, b: f.pos + Vec3::Y * (r * 1.7), r: 0.006, col: [0.25, 0.35, 0.12] });
        }
    }
    draw_prims(&mut commands, &mut pools.fruit, &prims, &mut paint, &mut meshes, &mut materials, &mut tfs);
}

pub struct PropsPlugin;

impl Plugin for PropsPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, draw);
    }
}
