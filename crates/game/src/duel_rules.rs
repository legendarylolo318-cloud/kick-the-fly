//! The 1v1 duel in play (X): the fly gets a blaster, aims it by turning (its LC10 / DNa02 neurons) and shoots when DNp35 fires.
//! Python: `Game3D.{toggle_duel, respawn_player, _duel_senses, _duel_motor, _fly_shoot, _pellets3d, _player_hit}` and the constants at
//! the top of `game/kick3d.py`. Plain logic, no engine and no brain thread: [`crate::duel_play`] feeds it and applies what it says.
//!
//! CONNECTOME: where you are is turned into pokes on the fly's real LC10 (target tracking, which drives the same-side DNa02 steering
//! neurons) and small-object detectors (which drive DNp35, the trigger), through the same `poke` the other senses use; the turn is the
//! right-minus-left firing of DNa01/DNa02; the shot is DNp35/DNpe052 above the `fire` threshold. How much it attends to you comes from
//! what its mushroom body learned about your smell. GAME RULE: the dead zone and gain from Hz to yaw, the cooldown, the pellet's speed,
//! spread and damage, the elevation of the shot toward your chest (a fly on the floor has no neurons for aiming up at a person), the
//! hit pulses of reward (PAM) and punishment (PPL1), and the respawn.

use bevy::math::{Vec2, Vec3};

pub const PLAYER_HP: f32 = 100.0;
pub const PLAYER_RADIUS: f32 = 0.32;
/// m/s, radians of scatter (per axis, normal), health per hit.
pub const PELLET_SPEED: f32 = 9.0;
pub const PELLET_SPREAD: f32 = 0.035;
pub const PELLET_DAMAGE: f32 = 9.0;
/// Seconds between shots, and how long a hit on you makes its reward neurons fire.
pub const FIRE_COOLDOWN: f32 = 0.45;
pub const REWARD_PULSE_S: f32 = 0.45;
/// DNa02 right-minus-left Hz below this is noise.
pub const STEER_DEADZONE_HZ: f32 = 1.5;
/// Radians a tick of turning per Hz above the dead zone, at most `STEER_MAX` (2 rad/s).
pub const STEER_GAIN: f32 = 0.004;
pub const STEER_MAX: f32 = 0.035;
/// The sideness that drives LC10 fully.
pub const TRACK_SCALE: f32 = 0.45;
/// Getting hurt while it smells you: PPL1 fires this long.
pub const PUNISH_PULSE_S: f32 = 0.6;
/// A pellet lives this long (s), falls by this much a second squared, and the fly sees you within this distance.
pub const PELLET_LIFE_S: f32 = 2.0;
pub const PELLET_FALL: f32 = 0.9;
pub const SIGHT_M: f32 = 7.5;
/// A fly this close (horizontally) to you smells you: `player_scent`.
pub const SCENT_M: f32 = 3.5;
/// The fly's mushroom body must fear you by this much more than it likes you before it runs (`k2.FEAR_ACT`).
pub const FEAR_ACT: f32 = 0.35;
/// The tick rate the Python physics ran at; pellet velocities are metres per tick.
pub const TICKS_PER_S: f32 = 60.0;

/// `duel_stats`: the card's last line.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Stats {
    pub shots: u32,
    pub hits: u32,
    pub deaths: u32,
}

/// A pellet in flight: position, velocity in metres per tick, and the time it was fired.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pellet {
    pub p: Vec3,
    pub v: Vec3,
    pub born: f32,
}

/// What the fly sees of you this tick (`_duel_senses`): the new valence and the pokes for LC10 and the small-object detectors.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Sense {
    pub valence: f32,
    /// `("track", side, strength 0.05, recruit)`.
    pub track: Option<(char, f32)>,
    /// `("small", side, strength 0.05, recruit)`, one or both sides.
    pub small: Vec<(char, f32)>,
}

/// `body_axes(yaw)`: forward and side on the floor plane.
pub fn axes(yaw: f32) -> (Vec3, Vec3) {
    (Vec3::new(yaw.cos(), 0.0, yaw.sin()), Vec3::new(-yaw.sin(), 0.0, yaw.cos()))
}

/// How strongly the fly attends to you: learned fear switches the pursuit off, liking keeps it on.
pub fn attend(valence: f32) -> f32 {
    (0.7 + 1.5 * valence).clamp(0.0, 1.0)
}

/// `_duel_senses` for a living, free fly. `head` is its head, `eye` yours; `fear` and `like` are what its mushroom body holds about
/// your smell. The valence follows `like - fear` slowly whatever the distance.
pub fn sense(head: Vec3, yaw: f32, eye: Vec3, valence: f32, fear: f32, like: f32) -> Sense {
    let valence = valence + ((like - fear) - valence) * 0.05;
    let mut out = Sense { valence, track: None, small: Vec::new() };
    let to = eye - head;
    let dist = Vec2::new(to.x, to.z).length();
    if dist > SIGHT_M {
        return out;
    }
    let to_h = Vec3::new(to.x, 0.0, to.z) / dist.max(1e-6);
    let (fwd, side) = axes(yaw);
    let (ahead, sideness) = (to_h.dot(fwd), to_h.dot(side));
    let att = attend(valence);
    if att <= 0.02 {
        return out;
    }
    let near = (1.3 - dist / 7.0).clamp(0.3, 1.0);
    let s_side = (sideness.abs() / TRACK_SCALE).clamp(0.0, 1.0) * att * near;
    let key = if sideness > 0.0 { 'R' } else { 'L' };
    if ahead < 0.995 && s_side > 0.03 {
        out.track = Some((key, 0.15 + 0.45 * s_side)); // short pulses renewed each tick: little lag
    }
    if ahead > 0.985 && dist < 7.0 {
        // you are dead ahead: small-object detectors
        let s = att * near * ((ahead - 0.985) / 0.012).clamp(0.0, 1.0);
        if s > 0.02 {
            let recruit = 0.2 + 0.4 * s;
            if sideness.abs() < 0.12 {
                out.small = vec![('L', recruit), ('R', recruit)];
            } else {
                out.small = vec![(key, recruit)];
            }
        }
    }
    out
}

/// `self.steer += (turn - steer) * 0.35`, `turn` being DNa02/DNa01 right minus left in Hz.
pub fn steer_step(steer: f32, turn_hz: f32) -> f32 {
    steer + (turn_hz - steer) * 0.35
}

/// The yaw it steers toward this tick, if the smoothed steering is out of the dead zone.
pub fn steer_yaw_target(yaw: f32, steer: f32) -> Option<f32> {
    let mag = (steer.abs() - STEER_DEADZONE_HZ).max(0.0);
    (mag > 0.0).then(|| yaw + (mag * STEER_GAIN).min(STEER_MAX).copysign(steer))
}

/// `_muzzle`: the barrel's mouth and the way it points (the fly's heading, not its pitch).
pub fn muzzle(head: Vec3, yaw: f32) -> (Vec3, Vec3) {
    let (fwd, _) = axes(yaw);
    (head + fwd * 0.22 + Vec3::Y * 0.03, fwd)
}

/// `_fly_shoot`: the pellet's velocity (metres per tick). `noise` is three standard-normal draws.
pub fn shot_velocity(muzzle: Vec3, fwd: Vec3, eye: Vec3, noise: Vec3) -> Vec3 {
    let target = eye - Vec3::new(0.0, 0.35, 0.0);
    let d = target - muzzle;
    let horiz = Vec2::new(d.x, d.z).length().max(0.4);
    let pitch = d.y.atan2(horiz);
    let v = fwd * pitch.cos() + Vec3::new(0.0, pitch.sin(), 0.0) + noise * PELLET_SPREAD;
    v.normalize_or_zero() * PELLET_SPEED / TICKS_PER_S
}

/// One tick of a pellet's flight (`_pellets3d`): gravity then movement.
pub fn step_pellet(b: &mut Pellet) {
    b.v.y -= PELLET_FALL / (TICKS_PER_S * TICKS_PER_S);
    b.p += b.v;
}

/// Does this pellet hit you? `pos` is your floor position (x, z), `eye_h` your eye's height.
pub fn hits_player(p: Vec3, pos: Vec2, eye_h: f32, alive: bool) -> bool {
    alive && Vec2::new(p.x - pos.x, p.z - pos.y).length() < PLAYER_RADIUS && p.y > 0.05 && p.y < eye_h + 0.15
}

/// Out of the arena, or too old: the pellet bursts in a puff of dust.
pub fn expired(p: Vec3, age: f32, half: Vec3) -> bool {
    p.x.abs() > half.x || p.z.abs() > half.z || p.y < 0.0 || p.y > half.y || age > PELLET_LIFE_S
}

/// `respawn_player`: the corner of the arena (a 6 m box around the fly outdoors) farthest from the fly, and the yaw that faces it.
/// `rx`, `rz` are the arena's half sizes.
pub fn respawn(fly_xz: Vec2, rx: f32, rz: f32) -> (Vec2, f32) {
    let (span_x, span_z) = ((rx - 0.8).min(6.0), (rz - 0.8).min(6.0));
    let around = if rx > 10.0 { fly_xz } else { Vec2::ZERO };
    let lo = Vec2::new(-rx + 0.8, -rz + 0.8);
    let hi = Vec2::new(rx - 0.8, rz - 0.8);
    let mut best = (Vec2::ZERO, -1.0f32);
    for x in [-span_x, span_x] {
        for z in [-span_z, span_z] {
            let c = (around + Vec2::new(x, z)).clamp(lo, hi);
            let d = c.distance(fly_xz);
            if d > best.1 {
                best = (c, d);
            }
        }
    }
    let v = fly_xz - best.0;
    (best.0, v.y.atan2(v.x))
}

/// The card's mood line: the word and its colour.
pub fn mood(valence: f32) -> (&'static str, [u8; 3]) {
    if valence > 0.15 {
        ("HUNTING YOU", [255, 120, 90])
    } else if valence < -0.2 {
        ("FEARS YOU", [120, 190, 255])
    } else {
        ("SIZING YOU UP", [205, 212, 224])
    }
}

/// How far the dead player's eye has slumped: it eases toward 0.3 m by 5 % a tick.
pub fn slump(eye_h: f32) -> f32 {
    eye_h + (0.3 - eye_h) * 0.05
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::FRAC_PI_2;

    /// A player eye straight ahead of a fly looking along +x, `d` metres away.
    fn ahead(d: f32) -> (Vec3, f32, Vec3) {
        (Vec3::new(0.0, 0.1, 0.0), 0.0, Vec3::new(d, 1.6, 0.0))
    }

    #[test]
    fn dead_ahead_drives_the_small_object_detectors_on_both_sides() {
        let (head, yaw, eye) = ahead(3.0);
        let s = sense(head, yaw, eye, 0.0, 0.0, 0.0);
        assert!(s.track.is_none(), "it is looking right at you: nothing to track");
        assert_eq!(s.small.len(), 2);
        assert!(s.small.iter().any(|x| x.0 == 'L') && s.small.iter().any(|x| x.0 == 'R'));
        // near = 1.3 - 3/7, attend 0.7 -> s = 0.7 * 0.871 = 0.61 -> recruit 0.2 + 0.4 s
        let near = 1.3 - 3.0 / 7.0;
        let want = 0.2 + 0.4 * (0.7 * near);
        assert!((s.small[0].1 - want).abs() < 1e-5, "{} vs {want}", s.small[0].1);
    }

    #[test]
    fn you_on_its_right_are_tracked_by_the_right_lc10() {
        // yaw 0 faces +x, so its side axis (-sin, 0, cos) is +z: you at +z are on its right
        let s = sense(Vec3::ZERO, 0.0, Vec3::new(0.2, 1.6, 3.0), 0.0, 0.0, 0.0);
        let (side, strength) = s.track.expect("tracked");
        assert_eq!(side, 'R');
        assert!(strength > 0.3 && strength <= 0.6, "{strength}");
        let s = sense(Vec3::ZERO, 0.0, Vec3::new(0.2, 1.6, -3.0), 0.0, 0.0, 0.0);
        assert_eq!(s.track.map(|t| t.0), Some('L'));
    }

    #[test]
    fn out_of_sight_and_learned_fear_switch_the_pursuit_off() {
        let far = sense(Vec3::ZERO, 0.0, Vec3::new(8.0, 1.6, 2.0), 0.0, 0.0, 0.0);
        assert!(far.track.is_none() && far.small.is_empty());
        // a valence of -0.5 gives attend 0 (0.7 - 0.75 < 0)
        let afraid = sense(Vec3::ZERO, 0.0, Vec3::new(2.0, 1.6, 2.0), -0.5, 1.0, 0.0);
        assert!(afraid.track.is_none() && afraid.small.is_empty());
        assert!(afraid.valence < -0.5, "the valence keeps falling toward like - fear");
    }

    #[test]
    fn valence_follows_liking_minus_fear_by_five_percent_a_tick() {
        let s = sense(Vec3::ZERO, 0.0, Vec3::new(20.0, 1.6, 0.0), 0.0, 0.0, 1.0);
        assert!((s.valence - 0.05).abs() < 1e-6);
    }

    #[test]
    fn steering_has_a_dead_zone_and_a_ceiling() {
        assert_eq!(steer_yaw_target(1.0, 1.2), None);
        let t = steer_yaw_target(1.0, 2.5).unwrap();
        assert!((t - (1.0 + 0.004)).abs() < 1e-6, "1 Hz over the dead zone is 0.004 rad");
        let t = steer_yaw_target(1.0, -50.0).unwrap();
        assert!((t - (1.0 - STEER_MAX)).abs() < 1e-6, "capped at 0.035 rad a tick, to the left");
        assert!((steer_step(0.0, 10.0) - 3.5).abs() < 1e-6);
    }

    #[test]
    fn a_shot_travels_at_nine_metres_a_second_toward_your_chest() {
        let (m, fwd) = muzzle(Vec3::new(0.0, 0.2, 0.0), 0.0);
        assert!((m - Vec3::new(0.22, 0.23, 0.0)).length() < 1e-6);
        assert_eq!(fwd, Vec3::X);
        let v = shot_velocity(m, fwd, Vec3::new(3.0, 1.6, 0.0), Vec3::ZERO);
        assert!((v.length() * TICKS_PER_S - PELLET_SPEED).abs() < 1e-4);
        assert!(v.y > 0.0 && v.x > 0.0, "it rises toward a person standing up");
        // the elevation is the chest (eye - 0.35) over the horizontal distance
        let want = (1.25f32 - 0.23).atan2(2.78);
        assert!((v.y.atan2(v.x) - want).abs() < 1e-4);
    }

    #[test]
    fn a_pellet_falls_and_hits_only_inside_your_cylinder() {
        let mut b = Pellet { p: Vec3::new(0.0, 1.0, 0.0), v: Vec3::new(0.15, 0.0, 0.0), born: 0.0 };
        step_pellet(&mut b);
        assert!(b.v.y < 0.0 && (b.p.x - 0.15).abs() < 1e-6);
        let pos = Vec2::new(2.0, 0.0);
        assert!(hits_player(Vec3::new(2.1, 1.0, 0.1), pos, 1.6, true));
        assert!(!hits_player(Vec3::new(2.5, 1.0, 0.0), pos, 1.6, true), "outside the 0.32 m radius");
        assert!(!hits_player(Vec3::new(2.0, 1.9, 0.0), pos, 1.6, true), "over your head");
        assert!(!hits_player(Vec3::new(2.0, 1.0, 0.0), pos, 1.6, false), "you are already dead");
    }

    #[test]
    fn pellets_burst_at_the_walls_and_after_two_seconds() {
        let half = Vec3::new(4.2, 3.0, 3.6);
        assert!(!expired(Vec3::new(1.0, 1.0, 1.0), 1.0, half));
        assert!(expired(Vec3::new(4.3, 1.0, 1.0), 0.1, half));
        assert!(expired(Vec3::new(1.0, -0.01, 1.0), 0.1, half));
        assert!(expired(Vec3::new(1.0, 1.0, 1.0), 2.01, half));
    }

    #[test]
    fn respawn_goes_to_the_corner_farthest_from_the_fly_and_faces_it() {
        // the room: corners at (+-3.4, +-2.8); a fly near (-3, -2.5) sends you to (+3.4, +2.8)
        let (pos, yaw) = respawn(Vec2::new(-3.0, -2.5), 4.2, 3.6);
        assert!((pos - Vec2::new(3.4, 2.8)).length() < 1e-5, "{pos:?}");
        let v = Vec2::new(-3.0, -2.5) - pos;
        assert!((yaw - v.y.atan2(v.x)).abs() < 1e-6);
        // outdoors the box (6 m) is around the fly, inside the 30 m field
        let (pos, _) = respawn(Vec2::new(10.0, 10.0), 30.0, 30.0);
        assert!(pos.distance(Vec2::new(10.0, 10.0)) > 8.0 && pos.abs().max_element() <= 29.2);
        let _ = FRAC_PI_2;
    }

    #[test]
    fn the_mood_line_and_the_vignette() {
        assert_eq!(mood(0.4).0, "HUNTING YOU");
        assert_eq!(mood(-0.3).0, "FEARS YOU");
        assert_eq!(mood(0.0).0, "SIZING YOU UP");
        assert!(slump(1.6) < 1.6 && slump(0.3) == 0.3);
    }
}
