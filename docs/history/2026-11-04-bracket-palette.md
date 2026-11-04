# Palette drift with `[` and `]` (2026-11-04)

**What.** The owner asked for `[` and `]` to scroll the palette continuously, with smoothed
onset and offset, and no motion with both pressed. Agreed in the conversation: colours only,
never zoom; the brackets act only with the scroll modifiers (Alt hue, Ctrl lightness, both),
and Shift inverts the lightness direction as with Ctrl-scroll ("a core feature: how else would
you have chroma and lightness going opposite ways"). Speeds to tune by feel, starting at the
scroll notches' 1 : 4 ratio: hue 1/8 turn per second, lightness 1/2.

**How** (`palette_drift.rs`). Held brackets are kept by physical key, so a release still
matches when the modifiers changed in between (which changes the character). Each phase's
speed eases toward its target, the bracket direction (sum of the held ones, so both = 0)
times its modifier, with a 0.15 s time constant, and snaps to 0 below 1e-4 turns per second.
`about_to_wait` advances the phases by the time since the last tick (capped at 100 ms) and
ticks at ~60 Hz while a speed is non-zero; at rest the main thread is event-driven as before.
Focus lost releases the keys. Matching the bracket: natively the character without modifiers
(`key_without_modifiers`); the web backend has no such call and reports the browser's
`event.key`, so the characters macOS's Option layer types (“ ” for `[`, ‘ ’ for `]`) and
Shift's `{` `}` are accepted too.

**Checked** in Chrome with synthetic key events (`Input.dispatchKeyEvent`, with the modifier
keys pressed and released as keys): `]` alone nothing; Alt+`]` 2 s: hue +0.232 (0.25 minus
the ease-in lag), coasting to +0.251; Alt+`[` 2 s back to 0.000; Ctrl+`]` 1 s: lightness
+0.428 → 0.500; Ctrl+Shift+`]` the inverse; both brackets no motion. Two harness pitfalls on
the way: modifier flags on the bracket events alone left winit's web modifier state wrong, and
Chrome throttles timers in a window it considers in the background (motion then slows, by the
100 ms cap). Not checked: natively (no synthetic input there), and real Option-layer
characters in a browser (synthetic events carry whatever key is given).
