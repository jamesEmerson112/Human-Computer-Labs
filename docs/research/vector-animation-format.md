# Vector's animation group and clip format

Researched 2026-10-01 from the community [Anki Vector technical reference manual](https://randym32.github.io/Vector-TRM.pdf), edition 2021.02.14. Page numbers below are the manual's printed page numbers. Everything under "From the manual" restates the manual. Everything under "Adapting it for the lamp" is our own proposal. A playable explainer is at https://claude.ai/artifact/ThMQwzA41joSu3GsFzhqG6.

## From the manual

### The layers

A behaviour asks for a reaction by firing a trigger name, such as a greeting. `AnimationTriggerMap.json` maps each trigger name to an animation group, which is a JSON file in `assets/animationGroups`. The group lists candidate clips. The engine screens the candidates by Vector's current mood, draws one at random by weight, and hands the clip name to vic-anim. vic-anim plays the clip's keyframes for the face, sound and lights, and vic-robot drives the head, lift and tread motors (chapter 22, pages 349 to 352).

Anki recommends firing triggers rather than clip names, because individual clip names can change between software updates (page 354). wire-pod does not follow that advice. Its `animationMap` in `chipper/pkg/wirepod/ttr/kgsim_cmds.go` maps its own words straight to clip names, for example `happy` to `anim_onboarding_reacttoface_happy_01`.

### Animation group file

A group file is one object with an `animations` array. Each entry has the fields below (table 494, page 352).

| Field | Type | Meaning |
|---|---|---|
| `Name` | string | The clip to play. |
| `Mood` | string | The simple mood this entry applies to, or `Default`. |
| `Weight` | float | Relative chance of this entry being drawn. Typically 1.0. |
| `CooldownTime_Sec` | float | Seconds after this clip finishes before it can be drawn again. Typically 0.0. |
| `UseHeadAngle` | bool | If true, also move the head to a random angle in the range below. Optional, default false. |
| `HeadAngleMin_Deg`, `HeadAngleMax_Deg` | float | The head angle range, within −22° to 45°. Used only when `UseHeadAngle` is true. |

The mood tag belongs to the entry, not to the clip. The clip file knows nothing about moods, so the same clip can appear in several groups with different tags and weights.

### How one entry is drawn

The engine first screens out entries that do not apply to the current mood. It then sums the weights of the remaining entries, divides each weight by that sum to get a probability, and draws one entry (page 352). Weights are therefore relative. A weight of 1.0 means 40% when the surviving weights are 1.0, 1.0 and 0.5, and 50% when they are 1.0 and 1.0. An entry that is cooling down is not available to be drawn.

The manual does not say what happens when no entry matches the current mood, or whether `Default` entries stay eligible under other moods. Any implementation of ours has to choose a rule for both cases.

Here is a made-up example that uses the real field names.

```json
{
  "animations": [
    { "Name": "anim_greet_01",         "Mood": "Default",  "Weight": 1.0, "CooldownTime_Sec": 0.0 },
    { "Name": "anim_greet_02",         "Mood": "Default",  "Weight": 1.0, "CooldownTime_Sec": 0.0 },
    { "Name": "anim_greet_bigwave_03", "Mood": "Default",  "Weight": 0.5, "CooldownTime_Sec": 8.0 },
    { "Name": "anim_greet_excited_01", "Mood": "HighStim", "Weight": 2.0, "CooldownTime_Sec": 0.0,
      "UseHeadAngle": true, "HeadAngleMin_Deg": 20.0, "HeadAngleMax_Deg": 40.0 },
    { "Name": "anim_greet_excited_02", "Mood": "HighStim", "Weight": 1.0, "CooldownTime_Sec": 4.0 }
  ]
}
```

Under `Default`, the three Default entries have probabilities of 40%, 40% and 20%. Once the big wave has played, it sits out for 8 seconds and the other two split 50/50. Under `HighStim`, the excited entries have probabilities of 67% and 33%, and the first one also sends the head to a random angle between 20° and 40°.

Variety comes from three mechanisms working together. Weights set how often each clip appears. Cooldowns stop a memorable clip from repeating back to back. Random ranges inside the group entry and inside the clip's keyframes mean even a repeated clip never plays identically.

### Where the mood comes from

Vector tracks five emotion values: Happy, Confident, Social, Stimulated and Trust (chapter 29, page 411). Events such as hearing its name or failing to lift its cube are mapped to changes, usually between −1 and 1, and each change decays back toward neutral along a configured curve (pages 412 to 413). A simple mood is a slower-changing name that maps to ranges of those values. The listed simple moods are `Default`, `Frustrated`, `HighStim`, `LowStim` and `MedStim` (pages 411 to 412). The manual says it is not known exactly how the simple mood interacts with the emotion values.

### Clip file

Clips are stored as FlatBuffers binary files under `assets/animations`, with several clips per file. By convention, a clip's name starts with its file's name. `anim_manifest.json` lists every clip name with its length in milliseconds (pages 353 and 391).

Each clip holds one keyframe array per track (table 533, page 397). The tracks are optional, can have different numbers of keyframes, and do not need to start together. Within one array, keyframe times must be in ascending order and no two can share a time. The tracks most relevant to us are below (tables 527 to 536, pages 394 to 399).

| Track | Key fields | Behaviour |
|---|---|---|
| `HeadAngleKeyFrame` | `triggerTime_ms`, `durationTime_ms`, `angle_deg`, `angleVariability_deg` | Reach the angle by the end of the duration, ramping speed smoothly. The angle can vary randomly by up to `angleVariability_deg`. |
| `LiftHeightKeyFrame` | `triggerTime_ms`, `durationTime_ms`, `height_mm`, `heightVariability_mm` | The same as the head, for the lift. |
| `BodyMotionKeyFrame` | `triggerTime_ms`, `durationTime_ms`, `radius_mm`, `speed` | Drive or turn the treads for the duration. |
| `BackpackLightsKeyFrame` | `triggerTime_ms`, `durationTime_ms`, `Front`, `Middle`, `Back` (RGBA, 0 to 1) | Hold the colours for the duration, then fade linearly to the next keyframe's colours. |
| `RobotAudioKeyFrame` | `triggerTime_ms`, `eventGroups` | Send a sound event. Each event group picks among several sounds by probability. |
| `ProceduralFaceKeyFrame` | `triggerTime_ms`, face scale and angle, per-eye lid and shape values | Squash, stretch and tilt the eyes. |
| `EventKeyFrame` | `triggerTime_ms`, `event_id` | Pause the clip until a named event arrives, such as `LISTENING_BEGIN`. |

While a clip plays, it locks every track it uses. A clip that needs a track that is already locked cannot play. A caller can mark the head, lift or body tracks as ignored so that something else can drive them (page 353).

## Adapting it for the lamp

This section is a proposal, not a decision. The character logic would fire triggers such as `greet`, `curious` or `celebrate`. Each trigger would have a small group of three to six entries using the same fields: a clip name, a state tag, a weight, a cooldown, and optional random ranges. Each clip would be a list of timed joint targets for the five URDF joints, plus light and sound cues, and could be keyframed in Blender. The body would clamp every target to the URDF's soft limits and velocity caps before the simulator sees it, which is the "play a timed joint trajectory" call in `docs/prompts/sim-prep.txt`.

Vector's tracks map onto the lamp as follows. `HeadAngle` maps to `head_pitch_joint`. `LiftHeight` maps loosely to `shoulder_pitch_joint` and `elbow_pitch_joint` together, used as posture. `BodyMotion` turning maps to `base_yaw_joint`. `BackpackLights` maps to the light at `light_emitter_link`, and `RobotAudio` maps to the laptop speaker. The lamp's `neck_yaw_joint` has no Vector counterpart. The lamp has no screen, so `ProceduralFace` has no counterpart either, and light pulses and small head movements would have to carry that expression.

On the ML route, the state tag is where a running emotional state built from per-utterance MELD labels would plug in, in the same way Vector's simple mood filters its groups.
