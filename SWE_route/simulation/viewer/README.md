# Three.js lamp viewer

The browser renders actual MuJoCo link poses supplied by the Rust controller. It uses the same 123 JSON gestures and original fixture geometry as the Python viewer. Assets and dependencies are served locally; no CDN or remote font is required.

From this directory, install the pinned packages and build:

```powershell
npm ci
npm test
npm run build
```

The Rust server serves `dist/` on its local port (8082 by default), `/api/scene`, `/assets/`, and `/ws`. During development, start the Rust server on port 8082 first, then run `npm run dev` and open `http://127.0.0.1:5173`. Vite stays on loopback port 5173 and exits if that port is occupied. It forwards scene, asset, and WebSocket requests to port 8082 after checking the original local Host and matching Origin; WebSockets require an Origin. The native server's origin restrictions remain enabled. Run the built viewer through the native server for measurements.

Choose a family or search, select a gesture, and press Play. Stop or Escape interrupts playback. Manual joint sliders send a new target when released. Angles are displayed in degrees and transmitted in radians. Pace, light controls, reset pose, and gesture reload are acknowledged by the controller. Reset view, Light view, and axes affect only the browser.

Each animation row has a native checkbox: checked means pass, unchecked means fail.
The review filter and Next button help you work through the list. Optional notes,
checks, and counts are separate from playback; changing a review never sends a
motion command. Keyboard users can toggle checkboxes with Space and move between
animation buttons with Up/Down or Home/End.

Reviews save in localStorage under `lamp-animation-review-v1` for the current
browser origin. Export review downloads a versioned JSON file with all current
animations, regardless of filters. Import review replaces the local review only
after validating the complete file; unknown names are reported, missing current
names default to fail, and malformed or unrelated files preserve the old review.
The file uses `type: "lamp-animation-review"`, `schema_version: 1`, and
`rule: "checked_pass_unchecked_fail"`; each animation includes its name, label,
group, status, notes, and last-edited timestamp. Reviews follow stable names,
so a changed motion should be unchecked and reviewed again. See the
[review workflow](../README.md#review-the-animations) for handing off failed clips.

The renderer interpolates actual world positions and quaternions with a 33.3 ms display buffer. It never predicts later motion. A disconnected or stale session disables commands and reconnects to obtain the complete current state; commands are never queued offline. Session details distinguish command acknowledgement from visible motion latency and rendering cadence from physics/control rates.

`?benchmark=1` selects the preserved baseline camera, stage dimensions, lighting, and fixed 940 × 805 canvas for browser comparisons. Ordinary mode uses a responsive layout and a closer presentation camera. Browser rendering cadence measures submitted frames, not confirmed display presentation.
