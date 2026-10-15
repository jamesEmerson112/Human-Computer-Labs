# Three.js lamp viewer

The browser renders actual MuJoCo link poses supplied by the Rust controller. It uses the shared JSON gesture library and original fixture geometry with the Python viewer. The selected wheel excludes the entire Bad family: 117 emotion animations plus nine utility gestures. Neutral / Reset pose is a separate control. Assets and dependencies are served locally; no CDN or remote font is required.

From this directory, install the pinned packages and build:

```powershell
npm ci
npm test
npm run build
```

The Rust server serves `dist/` on its local port (8082 by default), `/api/scene`, `/assets/`, and `/ws`. During development, start the Rust server on port 8082 first, then run `npm run dev` and open `http://127.0.0.1:5173`. Vite stays on loopback port 5173 and exits if that port is occupied. It forwards scene, asset, and WebSocket requests to port 8082 after checking the original local Host and matching Origin; WebSockets require an Origin. The native server's origin restrictions remain enabled. Run the built viewer through the native server for measurements.

Choose a family or search, select a gesture, and press Play. Stop or Escape interrupts playback. Manual joint sliders send a new target when released. Angles are displayed in degrees and transmitted in radians. Pace, light controls, reset pose, and gesture reload are acknowledged by the controller. Reset view, Light view, and axes affect only the browser.

Each animation row has a native checkbox: checked means pass, unchecked means fail.
The review-round filter separates six core, 37 middle, 74 outer emotions, and nine
utility gestures. The review filter and Next button help you work through the list. Optional notes,
checks, and counts are separate from playback; changing a review never sends a
motion command. Keyboard users can toggle checkboxes with Space and move between
animation buttons with Up/Down or Home/End.

Reviews save in localStorage under `lamp-animation-review-v2` for the current
browser origin. Export review downloads a versioned JSON file with all current
animations, regardless of filters. Import review replaces the local review only
after validating the complete file; unknown names are reported and retained in
exportable history, missing current names default to fail, and malformed or
wholly unrelated files preserve the old review. A report containing only known
retired baseline animations adds history without replacing active checks.
Earlier local reviews are archived before replacement. Downloaded and imported
reports share a 10 MiB UTF-8 size bound; oversized replacements are rejected
before changing active reviews.
The file uses `type: "lamp-animation-review"`, `schema_version: 2`, and
`rule: "checked_pass_unchecked_fail"`; each animation includes its name, label,
group, SHA256 revision, status, notes, and last-edited timestamp. A `history` array
retains prior statuses, notes, identities, and reasons for migration or reset.
Changed revisions or meaning clear approval on browser reload, live gesture
reload, reconnect, and import; notes and prior approval remain in history.
Cross-tab replacements save the combined history before accepting new checks,
so an immediate reload retains conflicting earlier notes. Tabs compare logical
review content to avoid repeated storage writes. Catalog changes are admitted
atomically: if validation or history size fails, existing checks keep their old
revision bindings and playback/review editing disable; Stop stays available.

Version 1 reports and browser storage remain supported. The viewer fetches
`/api/review-migration` before restoring legacy storage and leaves the original
`lamp-animation-review-v1` key untouched as a backup. Because v1 reports lack
motion hashes, migration explicitly assumes their complete name/label/group
identifies the recorded October 13 baseline. A pass carries only through an
explicit equivalent mapping whose target identity and hash match the current
catalog. Moved meaning, changed motion, missing mappings, and stale mappings
leave the animation unchecked. This is a baseline assumption, not proof of which
motion an older user saw. New v2 reviews carry actual file hashes. Human checks
remain the only source of approval; technical audits never mark an animation as
passed. See the
[review workflow](../README.md#review-the-animations) for handing off failed clips.

The renderer interpolates actual world positions and quaternions with a 33.3 ms display buffer. It never predicts later motion. A disconnected or stale session disables commands and reconnects to obtain the complete current state; commands are never queued offline. Session details distinguish command acknowledgement from visible motion latency and rendering cadence from physics/control rates.

`?benchmark=1` selects the preserved baseline camera, stage dimensions, lighting, and fixed 940 × 805 canvas for browser comparisons. Ordinary mode uses a responsive layout and a closer presentation camera. Browser rendering cadence measures submitted frames, not confirmed display presentation.
