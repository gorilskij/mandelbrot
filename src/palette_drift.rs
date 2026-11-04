//! Continuous palette scrolling with `[` / `]`: Alt drifts the hue phase,
//! Ctrl the lightness phase (both with both held), Shift inverts the
//! lightness direction, as with Alt/Ctrl-scroll. Without Alt or Ctrl the
//! brackets do nothing. Each phase's speed eases toward its target (smooth
//! onset and stop); both brackets held means a target of 0.

use std::collections::HashMap;
use std::time::Duration;

use winit::event::{ElementState, KeyEvent};
use winit::keyboard::{Key, ModifiersState, PhysicalKey};

use crate::rendering::PalettePhase;

/// Full speed, in turns per second (the same 1 : 4 ratio as a scroll notch's
/// HUE_TURNS_PER_NOTCH : LIGHT_TURNS_PER_NOTCH).
const HUE_TURNS_PER_SEC:   f64 = 1.0 / 8.0;
const LIGHT_TURNS_PER_SEC: f64 = 1.0 / 2.0;

/// Time constant of the speed's approach to its target.
const EASE_SECS: f64 = 0.15;

/// Below this speed (turns per second) with a target of 0, stop.
const REST_SPEED: f64 = 1e-4;

#[derive(Default)]
pub struct PaletteDrift {
    /// Held bracket keys, by physical key (so a release still matches when
    /// the modifiers changed meanwhile, which changes the character), and
    /// their direction: −1 for `[`, +1 for `]`.
    held:  HashMap<PhysicalKey, f64>,
    /// Current speeds, turns per second: hue, lightness.
    speed: (f64, f64),
}

impl PaletteDrift {
    /// Take a key event; true if it was a bracket (pressed or released).
    pub fn key(&mut self, e: &KeyEvent) -> bool {
        match e.state {
            ElementState::Pressed => {
                let Some(dir) = bracket(e) else { return false };
                self.held.insert(e.physical_key, dir);
                true
            }
            ElementState::Released => self.held.remove(&e.physical_key).is_some(),
        }
    }

    /// Focus lost: no release will come for the held keys.
    pub fn release_all(&mut self) {
        self.held.clear();
    }

    /// Whether the phases are moving or about to (needs ticks).
    pub fn active(&self) -> bool {
        !self.held.is_empty() || self.speed.0 != 0.0 || self.speed.1 != 0.0
    }

    /// Advance by `dt` with the current modifiers; true if `palette` changed.
    pub fn tick(&mut self, dt: Duration, mods: ModifiersState, palette: &mut PalettePhase) -> bool {
        let dt = dt.as_secs_f64();
        // Both brackets (or neither): no direction.
        let dir = self.held.values().sum::<f64>().clamp(-1.0, 1.0);
        let light_dir = if mods.shift_key() { -dir } else { dir };
        let target = (
            if mods.alt_key() { dir * HUE_TURNS_PER_SEC } else { 0.0 },
            if mods.control_key() { light_dir * LIGHT_TURNS_PER_SEC } else { 0.0 },
        );
        let k = 1.0 - (-dt / EASE_SECS).exp();
        let ease = |v: &mut f64, t: f64| {
            *v += (t - *v) * k;
            if t == 0.0 && v.abs() < REST_SPEED { *v = 0.0; }
        };
        ease(&mut self.speed.0, target.0);
        ease(&mut self.speed.1, target.1);
        if self.speed == (0.0, 0.0) { return false; }
        palette.hue   = (palette.hue   + self.speed.0 * dt).rem_euclid(1.0);
        palette.light = (palette.light + self.speed.1 * dt).rem_euclid(1.0);
        true
    }
}

/// −1 for `[`, +1 for `]`, by the layout's character without modifiers
/// (natively), or by the characters the browser reports with them (on the
/// web `event.key` includes them: macOS's Option layer types “ ” for `[`
/// and ‘ ’ for `]`, Shift `{` `}`).
fn bracket(e: &KeyEvent) -> Option<f64> {
    #[cfg(not(target_arch = "wasm32"))]
    let key = {
        use winit::platform::modifier_supplement::KeyEventExtModifierSupplement;
        e.key_without_modifiers()
    };
    #[cfg(target_arch = "wasm32")]
    let key = e.logical_key.clone();
    match &key {
        Key::Character(c) => match c.as_str() {
            "[" | "{" | "“" | "”" => Some(-1.0),
            "]" | "}" | "‘" | "’" => Some(1.0),
            _ => None,
        },
        _ => None,
    }
}
