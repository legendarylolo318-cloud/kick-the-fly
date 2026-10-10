//! The sun and the clock neurons' drive in `f64` (`game/outdoors.py`), for the Lab engines; the game's `world.rs` has the `f32` ones.

/// `(azimuth_deg, elevation_deg)` of the day/night cycle at `time_s`.
pub fn diurnal_cycle(time_s: f64, day_length_s: f64) -> (f64, f64) {
    let phase = (time_s % day_length_s) / day_length_s * 2.0 * std::f64::consts::PI;
    let az = (phase.to_degrees() + 180.0) % 360.0;
    let el = 45.0 * (phase - std::f64::consts::FRAC_PI_2).sin() + 15.0;
    (az, el)
}

/// `(left eye, right eye)` photoreceptor drive from the sun.
pub fn sun_light(yaw: f64, az_deg: f64, el_deg: f64) -> (f64, f64) {
    let el = el_deg.to_radians();
    let total = if el > 0.0 { (0.15 + 0.85 * el.sin()).clamp(0.0, 1.0) } else { 0.0 };
    let az = az_deg.to_radians();
    let side = (-yaw.sin(), yaw.cos());
    let sideness = (az.cos() * side.0 + az.sin() * side.1) * el.cos();
    (total * (1.0 - 0.35 * sideness), total * (1.0 + 0.35 * sideness))
}

/// `circadian_clock_drive(el)["morning_cells"]`.
pub fn morning_cells(el_deg: f64) -> f64 {
    if el_deg > 0.0 { (el_deg / 60.0).clamp(0.0, 1.0) } else { 0.0 }
}
