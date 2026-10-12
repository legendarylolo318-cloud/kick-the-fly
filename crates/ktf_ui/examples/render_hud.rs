//! `cargo run -p ktf_ui --example render_hud -- out.png [window_w window_h]`: the HUD with the values of the reference screenshot
//! (`tests/ui_reference/py_3d_hud_room.png`, a 930x906 window), drawn at HUD units and scaled to the window size over a flat background.
use ktf_ui::Canvas;
use ktf_ui::hud::*;
use ktf_ui::layout::compute_layout;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let out = a.get(1).cloned().unwrap_or("hud.png".into());
    let (wn, hn): (i32, i32) = (a.get(2).and_then(|s| s.parse().ok()).unwrap_or(930), a.get(3).and_then(|s| s.parse().ok()).unwrap_or(906));
    let lay = compute_layout(wn, hn, true, 0, 1.0);
    let f = HudFonts::new(false);
    let st = HudState {
        fly_state: "flying".into(),
        flies: 1,
        max_flies: 16,
        health: 100.0,
        max_health: 100.0,
        arena: "room".into(),
        pain: 1.0,
        pain_parts: [0.0, 0.02, 0.0, 0.0, 0.0],
        pain_trace: vec![0.5; 400],
        pain_peak: 3.0,
        pain_level_name: "normal".into(),
        pain_neurons: 8745,
        pain_connectome: false,
        reward: 3.0,
        reward_level: 1.0,
        tools: [
            ("hand", "HAND", "1"),
            ("swatter", "SWAT", "2"),
            ("torch", "TORCH", "3"),
            ("freeze", "FREEZE", "4"),
            ("sugar", "SUGAR", "5"),
        ]
        .iter()
        .map(|(n, l, k)| ToolSlot { name: n.to_string(), label: l.to_string(), key: k.to_string() })
        .collect(),
        tool_name: "hand".into(),
        tool_line: "1  HAND: drag the fly and throw it".into(),
        pages: 1,
        free_mouse_key: "tab".into(),
        panel_alpha: 255,
        panel: Some(PanelData {
            neurons: 166_700,
            touch: [3.0, 2.0, 1.0, 3.0],
            motor: [
                ("head-touch DNs  > JUMP", 1.3, Some(3.0)),
                ("body-touch DNs  > RUN", 0.9, Some(2.4)),
                ("leg-touch DNs   > KICK", 1.0, Some(2.0)),
                ("DNp09 (P9)      > WALK", 2.0, Some(3.0)),
                ("MDN moonwalker  > BACK UP", 0.1, Some(3.8)),
                ("DNa01/02 left   > TURN", 0.8, Some(2.1)),
                ("DNa01/02 right  > TURN", 1.1, Some(2.1)),
                ("DNg02 wing power> FLY", 0.6, Some(1.58)),
                ("DNp01 giant fiber> DODGE", 1.0, Some(4.0)),
                ("DNp35 object DNs> SHOOT", 0.8, Some(3.0)),
            ]
            .iter()
            .map(|(l, v, t)| MotorRow { label: l.to_string(), level: *v, thresh: *t })
            .collect(),
            whole_trace: (0..600).map(|i| if i < 30 { i as f32 / 6.0 } else { 5.2 }).collect(),
            log: vec![
                LogLine { age: 0.1, text: "NEURODEX  SNxx14 discovered x4.8 calm".into(), source: None },
                LogLine { age: 0.5, text: "NEURODEX  ORN_VM5v discovered x5.0 calm".into(), source: None },
                LogLine { age: 2.4, text: "TAKE OFF DNg02 x1.58".into(), source: None },
            ],
            steps_per_s: 200.0,
            step_ms: 1.9,
            ..Default::default()
        }),
        ..Default::default()
    };
    let mut cv = Canvas::new(lay.hud_w, lay.hud_h);
    Hud { cv: &mut cv, lay, f: &f }.draw(&st, 0.0);
    // over a flat background, then scaled to the window like the game does (bilinear)
    let rgb = cv.to_rgb_over([120, 100, 80]);
    let (sw, sh) = (wn as usize, hn as usize);
    let mut img = vec![0u8; sw * sh * 3];
    for y in 0..sh {
        for x in 0..sw {
            let (fx, fy) = ((x as f32 + 0.5) * lay.hud_w as f32 / sw as f32 - 0.5, (y as f32 + 0.5) * lay.hud_h as f32 / sh as f32 - 0.5);
            let (x0, y0) = (fx.floor().max(0.0) as usize, fy.floor().max(0.0) as usize);
            let (x1, y1) = ((x0 + 1).min(lay.hud_w as usize - 1), (y0 + 1).min(lay.hud_h as usize - 1));
            let (tx, ty) = ((fx - x0 as f32).clamp(0.0, 1.0), (fy - y0 as f32).clamp(0.0, 1.0));
            for c in 0..3 {
                let p = |xx: usize, yy: usize| rgb[(yy * lay.hud_w as usize + xx) * 3 + c] as f32;
                let v =
                    p(x0, y0) * (1.0 - tx) * (1.0 - ty) + p(x1, y0) * tx * (1.0 - ty) + p(x0, y1) * (1.0 - tx) * ty + p(x1, y1) * tx * ty;
                img[(y * sw + x) * 3 + c] = v as u8;
            }
        }
    }
    let file = std::fs::File::create(out).unwrap();
    let mut enc = png::Encoder::new(std::io::BufWriter::new(file), wn as u32, hn as u32);
    enc.set_color(png::ColorType::Rgb);
    enc.set_depth(png::BitDepth::Eight);
    enc.write_header().unwrap().write_image_data(&img).unwrap();
}
