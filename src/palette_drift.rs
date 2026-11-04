//! Palette control: which phases the colour controls move (toggled by
//! tapping Alt = hue, Ctrl = lightness, Shift = lightness inverted), a drift
//! speed set with `[` / `]` (×1.5 steps; `{` / `}` ×√1.5; `0` clears) that
//! moves them continuously, and the per-notch shift for Cmd-scroll. The
//! actual speed eases toward the set one (smooth onset and stop).

use std::time::Duration;

use winit::event::{ElementState, KeyEvent};
use winit::keyboard::{Key, ModifiersState, NamedKey};

use crate::rendering::PalettePhase;

/// The drift's base speed (level ±1), in turns per second (the same 1 : 4
/// ratio as Cmd-scroll's HUE_TURNS_PER_NOTCH : LIGHT_TURNS_PER_NOTCH).
const HUE_TURNS_PER_SEC:   f64 = 1.0 / 8.0;
const LIGHT_TURNS_PER_SEC: f64 = 1.0 / 2.0;

/// Highest speed, as a multiple of the base one: a step past it is refused
/// (so the top level is 1.5⁷ ≈ 17.1×).
const MAX_FACTOR: f64 = 20.0;

/// Time constant of the speed's approach to the set one.
const EASE_SECS: f64 = 0.15;

/// Below this speed (turns per second) with a target of 0, stop.
const REST_SPEED: f64 = 1e-4;

/// The modifier keys that toggle a phase when tapped.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Toggle { Hue, Light, Invert }

pub struct PaletteDrift {
    /// Phases moved by the drift and Cmd-scroll.
    pub hue:    bool,
    pub light:  bool,
    /// Lightness moves against the hue.
    pub invert: bool,
    /// Set speed: 0 stopped, else ±(1 + half-steps above the base speed);
    /// the factor is 1.5^((|level| − 1) / 2).
    level: i32,
    /// Current speeds, turns per second: hue, lightness.
    speed: (f64, f64),
    /// A modifier pressed with no other key since: toggles on release.
    tap: Option<Toggle>,
}

impl Default for PaletteDrift {
    fn default() -> Self {
        Self { hue: true, light: false, invert: false, level: 0, speed: (0.0, 0.0), tap: None }
    }
}

impl PaletteDrift {
    /// Take a key event (modifiers already updated); returns what it did,
    /// for the title, if it was one of ours.
    pub fn key(&mut self, e: &KeyEvent, mods: ModifiersState) -> Option<String> {
        let toggle = match &e.logical_key {
            Key::Named(NamedKey::Alt)     => Some(Toggle::Hue),
            Key::Named(NamedKey::Control) => Some(Toggle::Light),
            Key::Named(NamedKey::Shift)   => Some(Toggle::Invert),
            _ => None,
        };
        if e.repeat { return None; }
        match (e.state, toggle) {
            (ElementState::Pressed, Some(t)) => { self.tap = Some(t); return None; }
            (ElementState::Released, Some(t)) => {
                if self.tap.take() != Some(t) { return None; }
                let on = match t {
                    Toggle::Hue    => { self.hue = !self.hue; self.hue }
                    Toggle::Light  => { self.light = !self.light; self.light }
                    Toggle::Invert => { self.invert = !self.invert; self.invert }
                };
                return Some(format!("{t:?} {}", if on { "on" } else { "off" }));
            }
            (ElementState::Released, None) => return None,
            (ElementState::Pressed, None) => self.tap = None, // a chord, not a tap
        }
        if is_char(e, "0") {
            self.level = 0;
            return Some("drift stop".into());
        }
        let dir = bracket(e)?;
        // Shift (`{` / `}`): a half-step.
        let half = mods.shift_key() || matches!(base_char(e).as_deref(), Some("{" | "}"));
        let steps = if half { 1 } else { 2 };
        let refused = !self.step(dir, steps);
        Some(format!("drift {}{}", self.describe_level(), if refused { " (max)" } else { "" }))
    }

    /// Something else happened (a click, a scroll): a held modifier is a
    /// chord now, not a tap.
    pub fn cancel_tap(&mut self) {
        self.tap = None;
    }

    /// Move the set speed `steps` half-steps in direction `dir` (±1) along
    /// the ladder …, −1.5×, −1×, 0, +1×, +1.5×, … (below 1× is 0). False if
    /// that would pass MAX_FACTOR.
    fn step(&mut self, dir: i32, steps: i32) -> bool {
        let new = if self.level == 0 {
            dir
        } else if self.level.signum() == dir {
            self.level + dir * steps
        } else {
            let mag = self.level.abs() - steps;
            if mag < 1 { 0 } else { -dir * mag }
        };
        if factor(new) > MAX_FACTOR { return false; }
        self.level = new;
        true
    }

    fn describe_level(&self) -> String {
        if self.level == 0 { "0".into() } else { format!("{:+.2}×", self.level.signum() as f64 * factor(self.level)) }
    }

    /// For the title: the set speed and the toggled phases.
    pub fn describe(&self) -> String {
        let mut out = format!("drift {}", self.describe_level());
        for (on, name) in [(self.hue, " hue"), (self.light, " light"), (self.invert && self.light, "⇅")] {
            if on { out += name; }
        }
        out
    }

    /// Whether the phases are moving or about to (needs ticks).
    pub fn active(&self) -> bool {
        self.level != 0 || self.speed != (0.0, 0.0)
    }

    /// Advance by `dt`; true if `palette` changed.
    pub fn tick(&mut self, dt: Duration, palette: &mut PalettePhase) -> bool {
        let dt = dt.as_secs_f64();
        let v = if self.level == 0 { 0.0 } else { self.level.signum() as f64 * factor(self.level) };
        let target = (
            if self.hue { v * HUE_TURNS_PER_SEC } else { 0.0 },
            if self.light { self.light_dir() * v * LIGHT_TURNS_PER_SEC } else { 0.0 },
        );
        let k = 1.0 - (-dt / EASE_SECS).exp();
        let ease = |s: &mut f64, t: f64| {
            *s += (t - *s) * k;
            if t == 0.0 && s.abs() < REST_SPEED { *s = 0.0; }
        };
        ease(&mut self.speed.0, target.0);
        ease(&mut self.speed.1, target.1);
        if self.speed == (0.0, 0.0) { return false; }
        palette.hue   = (palette.hue   + self.speed.0 * dt).rem_euclid(1.0);
        palette.light = (palette.light + self.speed.1 * dt).rem_euclid(1.0);
        true
    }

    /// Cmd-scroll: shift the toggled phases by `hue` / `light` turns.
    pub fn scroll(&self, hue: f64, light: f64, palette: &mut PalettePhase) -> bool {
        if self.hue { palette.hue = (palette.hue + hue).rem_euclid(1.0); }
        if self.light { palette.light = (palette.light + self.light_dir() * light).rem_euclid(1.0); }
        self.hue || self.light
    }

    fn light_dir(&self) -> f64 {
        if self.invert { -1.0 } else { 1.0 }
    }
}

/// The speed factor of a level: 1.5^((|level| − 1) / 2).
fn factor(level: i32) -> f64 {
    1.5f64.powf((level.abs() - 1) as f64 / 2.0)
}

/// The layout's character without modifiers (natively), or as reported (on
/// the web `event.key` includes them).
fn base_char(e: &KeyEvent) -> Option<String> {
    #[cfg(not(target_arch = "wasm32"))]
    let key = {
        use winit::platform::modifier_supplement::KeyEventExtModifierSupplement;
        e.key_without_modifiers()
    };
    #[cfg(target_arch = "wasm32")]
    let key = e.logical_key.clone();
    match key {
        Key::Character(c) => Some(c.to_string()),
        _ => None,
    }
}

fn is_char(e: &KeyEvent, c: &str) -> bool {
    base_char(e).as_deref() == Some(c)
}

/// −1 for `[`, +1 for `]`; also their Shift characters `{` `}` (the web
/// reports those) and those of macOS's Option layer (“ ” and ‘ ’).
fn bracket(e: &KeyEvent) -> Option<i32> {
    match base_char(e)?.as_str() {
        "[" | "{" | "“" | "”" => Some(-1),
        "]" | "}" | "‘" | "’" => Some(1),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn speed(d: &PaletteDrift) -> f64 {
        if d.level == 0 { 0.0 } else { d.level.signum() as f64 * factor(d.level) }
    }

    #[test]
    fn full_steps_are_1_5x_through_zero() {
        let mut d = PaletteDrift::default();
        assert!(d.step(1, 2));
        assert_eq!(speed(&d), 1.0);
        assert!(d.step(1, 2));
        assert!((speed(&d) - 1.5).abs() < 1e-12);
        assert!(d.step(-1, 2));
        assert_eq!(speed(&d), 1.0);
        assert!(d.step(-1, 2));
        assert_eq!(speed(&d), 0.0);
        assert!(d.step(-1, 2));
        assert_eq!(speed(&d), -1.0);
        assert!(d.step(-1, 2));
        assert!((speed(&d) + 1.5).abs() < 1e-12);
    }

    #[test]
    fn half_steps_are_sqrt_1_5x_and_mix_with_full_ones() {
        let mut d = PaletteDrift::default();
        d.step(1, 1);
        assert_eq!(speed(&d), 1.0);       // the first step is always 1×
        d.step(1, 1);
        assert!((speed(&d) - 1.5f64.sqrt()).abs() < 1e-12);
        d.step(1, 2);
        assert!((speed(&d) - 1.5f64.powf(1.5)).abs() < 1e-12);
        d.step(-1, 2);                    // back to √1.5×
        d.step(-1, 2);                    // below 1×: 0
        assert_eq!(speed(&d), 0.0);
    }

    #[test]
    fn steps_past_20x_are_refused_not_rounded() {
        let mut d = PaletteDrift::default();
        d.step(1, 2);
        for _ in 0..7 { assert!(d.step(1, 2)); }
        let top = speed(&d);
        assert!((top - 1.5f64.powi(7)).abs() < 1e-9); // ≈ 17.09
        assert!(!d.step(1, 2));            // 25.6×
        assert!(!d.step(1, 1));            // 20.9×
        assert_eq!(speed(&d), top);
        let mut d = PaletteDrift::default();
        d.step(-1, 1);
        for _ in 0..14 { assert!(d.step(-1, 1)); }
        assert!(!d.step(-1, 1));
        assert!((speed(&d) + 1.5f64.powi(7)).abs() < 1e-9);
    }
}

