//! Prepared speech timelines and deterministic, bounded emotional motion cues.
//! Sampling uses monotonic wall time: delayed frames never replay missed accents.

use crate::motion::{JointLimit, Pose};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{collections::BTreeMap, path::Path};

pub const LEAD_S: f64 = 0.8;
pub const PHRASE_GAP_S: f64 = 0.18;
pub const SETTLE_S: f64 = 1.2;
pub const EMOTIONS: [&str; 7] = [
    "Happy",
    "Sad",
    "Angry",
    "Fearful",
    "Disgusted",
    "Surprised",
    "Neutral",
];
const JOINTS: [&str; 5] = [
    "base_yaw_joint",
    "shoulder_pitch_joint",
    "elbow_pitch_joint",
    "neck_yaw_joint",
    "head_pitch_joint",
];

#[derive(Clone, Debug, Deserialize)]
pub struct Phrase {
    pub text: String,
    pub audio: String,
    pub duration_s: f64,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Scene {
    pub id: String,
    pub label: String,
    pub text: String,
    pub phrases: Vec<Phrase>,
}

impl Scene {
    pub fn duration(&self) -> f64 {
        self.phrases.iter().map(|p| p.duration_s).sum::<f64>()
            + self.phrases.len().saturating_sub(1) as f64 * PHRASE_GAP_S
    }

    fn phrase_at(&self, elapsed: f64) -> Option<(usize, f64)> {
        let mut offset = 0.0;
        for (index, phrase) in self.phrases.iter().enumerate() {
            if elapsed >= offset && elapsed < offset + phrase.duration_s {
                return Some((index, (elapsed - offset) / phrase.duration_s));
            }
            offset += phrase.duration_s + PHRASE_GAP_S;
        }
        None
    }
}

#[derive(Deserialize)]
struct Catalog {
    schema_version: u32,
    scenes: Vec<Scene>,
}

/// Old fixture roots without performances remain usable for controller benchmarks.
pub fn load_scenes(root: &Path) -> Result<BTreeMap<String, Scene>, String> {
    let folder = root.join("performances");
    let path = folder.join("demo.json");
    if !path.exists() {
        return Ok(BTreeMap::new());
    }
    let bytes = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    if bytes.len() > 128 * 1024 {
        return Err("Performance catalog exceeds 128 KiB".into());
    }
    let text = std::str::from_utf8(&bytes).map_err(|e| e.to_string())?;
    let value = crate::gestures::parse_unique_json(text)?;
    let catalog: Catalog =
        serde_json::from_value(value).map_err(|e| format!("Invalid performance catalog: {e}"))?;
    if catalog.schema_version != 1 || catalog.scenes.is_empty() || catalog.scenes.len() > 3 {
        return Err("Expected performance schema 1 with one to three prepared scenes".into());
    }
    let canonical_folder = folder.canonicalize().map_err(|e| e.to_string())?;
    let mut scenes = BTreeMap::new();
    for scene in catalog.scenes {
        if !["greeting", "question", "reassurance"].contains(&scene.id.as_str())
            || scene.label.is_empty()
            || scene.label.len() > 128
            || scene.text.is_empty()
            || scene.text.len() > 4096
            || scene.phrases.is_empty()
            || scene.phrases.len() > 32
        {
            return Err("Invalid prepared performance scene".into());
        }
        if scene
            .phrases
            .iter()
            .map(|p| p.text.as_str())
            .collect::<Vec<_>>()
            .join(" ")
            != scene.text
        {
            return Err(format!(
                "Scene {} phrases must preserve its exact text",
                scene.id
            ));
        }
        for phrase in &scene.phrases {
            if phrase.text.is_empty()
                || !phrase.duration_s.is_finite()
                || !(0.0 < phrase.duration_s && phrase.duration_s <= 30.0)
                || !phrase.audio.starts_with("audio/")
                || !phrase.audio.ends_with(".wav")
                || phrase.audio.contains('\\')
                || phrase
                    .audio
                    .split('/')
                    .any(|s| s.is_empty() || s == "." || s == ".." || s.contains(':'))
            {
                return Err(format!("Invalid prepared phrase in {}", scene.id));
            }
            let audio = folder
                .join(&phrase.audio)
                .canonicalize()
                .map_err(|e| format!("{}: {e}", phrase.audio))?;
            if !audio.starts_with(&canonical_folder) {
                return Err("Speech asset escapes performance folder".into());
            }
            let metadata = std::fs::metadata(&audio).map_err(|e| e.to_string())?;
            if metadata.len() > 8 * 1024 * 1024 {
                return Err("Speech WAV exceeds 8 MiB".into());
            }
            let wav = std::fs::read(&audio).map_err(|e| e.to_string())?;
            let measured = wav_duration(&wav)?;
            if (measured - phrase.duration_s).abs() > 0.0001 {
                return Err(format!("Speech duration mismatch: {}", phrase.audio));
            }
        }
        if scene.duration() > 60.0 {
            return Err("Performance exceeds 60 seconds".into());
        }
        let id = scene.id.clone();
        if scenes.insert(id.clone(), scene).is_some() {
            return Err(format!("Duplicate performance scene: {id}"));
        }
    }
    Ok(scenes)
}

fn wav_duration(bytes: &[u8]) -> Result<f64, String> {
    let invalid = || "Prepared speech must be a complete PCM WAV".to_owned();
    if bytes.len() < 12 || &bytes[..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(invalid());
    }
    let u32_at =
        |offset| u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize;
    if u32_at(4) + 8 != bytes.len() {
        return Err(invalid());
    }
    let mut offset = 12;
    let mut byte_rate = None;
    let mut data_size = None;
    while offset + 8 <= bytes.len() {
        let size = u32_at(offset + 4);
        let start = offset + 8;
        if size > bytes.len() - start {
            return Err(invalid());
        }
        match &bytes[offset..offset + 4] {
            b"fmt " => {
                if size < 16 || byte_rate.is_some() {
                    return Err(invalid());
                }
                let u16_at = |i| u16::from_le_bytes(bytes[i..i + 2].try_into().unwrap()) as usize;
                let channels = u16_at(start + 2);
                let sample_rate = u32_at(start + 4);
                let rate = u32_at(start + 8);
                let align = u16_at(start + 12);
                let bits = u16_at(start + 14);
                if u16_at(start) != 1
                    || channels != 1
                    || bits != 16
                    || !(8000..=96000).contains(&sample_rate)
                    || align != channels * bits / 8
                    || rate != sample_rate * align
                {
                    return Err(invalid());
                }
                byte_rate = Some(rate);
            }
            b"data" => {
                if data_size.replace(size).is_some() || size == 0 || size % 2 != 0 {
                    return Err(invalid());
                }
            }
            _ => {}
        }
        offset = start + size + size % 2;
    }
    if offset != bytes.len() {
        return Err(invalid());
    }
    Ok(data_size.ok_or_else(invalid)? as f64 / byte_rate.ok_or_else(invalid)? as f64)
}

#[derive(Clone, Debug)]
pub struct Performance {
    pub id: String,
    pub owner: u64,
    pub scene: Scene,
    pub emotion: String,
    pub intensity: f64,
    pub start_at_s: f64,
    origin: Pose,
}

impl Performance {
    pub fn new(
        id: String,
        owner: u64,
        scene: Scene,
        emotion: String,
        intensity: f64,
        now: f64,
        origin: Pose,
    ) -> Result<Self, String> {
        if id.is_empty() || id.len() > 128 {
            return Err("Performance id must contain 1 to 128 UTF-8 bytes".into());
        }
        if !EMOTIONS.contains(&emotion.as_str()) {
            return Err("Unknown core performance emotion".into());
        }
        if !intensity.is_finite() || !(0.0..=1.0).contains(&intensity) {
            return Err("Performance intensity must be between zero and one".into());
        }
        if !now.is_finite() || now < 0.0 {
            return Err("Invalid performance clock".into());
        }
        Ok(Self {
            id,
            owner,
            scene,
            emotion,
            intensity,
            start_at_s: now + LEAD_S,
            origin,
        })
    }

    pub fn state(&self, now: f64) -> Value {
        let elapsed = now - self.start_at_s;
        let phase = if elapsed < 0.0 {
            "anticipating"
        } else if elapsed < self.scene.duration() {
            "speaking"
        } else if elapsed < self.scene.duration() + SETTLE_S {
            "settling"
        } else {
            "idle"
        };
        let phrase_index = if phase == "speaking" {
            self.scene.phrase_at(elapsed).map(|(i, _)| i)
        } else {
            None
        };
        json!({"id":self.id,"scene":self.scene.id,"emotion":self.emotion,"intensity":self.intensity,
            "text":self.scene.text,"start_at_s":self.start_at_s,"audio_duration_s":self.scene.duration(),
            "phase":phase,"phrase_index":phrase_index})
    }

    pub fn velocity_scale(&self) -> f64 {
        let energy = match self.emotion.as_str() {
            "Sad" => 0.27,
            "Fearful" => 0.5,
            "Disgusted" => 0.4,
            "Neutral" => 0.35,
            "Angry" => 0.7,
            "Surprised" => 0.8,
            _ => 0.6,
        };
        energy * (0.5 + self.intensity * 0.5)
    }

    pub fn sample(&self, now: f64, limits: &BTreeMap<String, JointLimit>) -> Pose {
        let elapsed = now - self.start_at_s;
        let posture = match self.emotion.as_str() {
            "Happy" => [0.0, -0.16, -0.14, 0.04, -0.18],
            "Sad" => [0.0, 0.14, -0.30, 0.0, 0.30],
            "Angry" => [0.0, -0.22, -0.12, 0.0, 0.18],
            "Fearful" => [-0.07, 0.17, -0.26, 0.10, -0.06],
            "Disgusted" => [-0.12, 0.06, -0.12, -0.18, 0.12],
            "Surprised" => [0.0, -0.23, -0.18, 0.0, -0.28],
            _ => [0.0, -0.025, -0.06, 0.0, -0.025],
        };
        let settle = ease((elapsed - self.scene.duration()) / SETTLE_S);
        let mut target = posture.map(|p| p * self.intensity * (1.0 - settle * 0.55));
        if let Some((index, fraction)) = self.scene.phrase_at(elapsed) {
            // A single smooth phrase accent, with no word-level timing claim.
            let accent =
                (std::f64::consts::PI * fraction).sin().powi(2) * (0.35 + 0.65 * self.intensity);
            match self.scene.id.as_str() {
                "greeting" => {
                    target[1] -= accent * if index == 1 { 0.13 } else { 0.045 };
                    target[4] -= accent * if index == 0 { 0.13 } else { 0.05 };
                }
                "question" => {
                    target[3] += 0.14 * accent;
                    target[4] -= 0.08 * accent;
                }
                "reassurance" => {
                    target[1] -= 0.07 * accent;
                    target[4] += 0.07 * accent;
                }
                _ => {}
            }
            if self.emotion == "Angry" {
                target[4] += accent * 0.12 * self.intensity;
            }
        }
        let idle_time = elapsed - self.scene.duration() - SETTLE_S;
        if idle_time > 0.0 {
            // Infrequent, sub-degree attentive movement with zero slope at each edge.
            let cycle = idle_time % 8.0;
            if cycle < 2.4 {
                let pulse = (std::f64::consts::PI * cycle / 2.4).sin().powi(2);
                target[4] -= 0.009 * pulse;
                target[3] += 0.006 * pulse;
            }
        }
        let arrival = ease((elapsed + LEAD_S) / LEAD_S);
        limits
            .iter()
            .map(|(name, limit)| {
                let goal = JOINTS
                    .iter()
                    .position(|joint| *joint == name)
                    .map(|i| target[i])
                    .unwrap_or(0.0);
                let origin = self.origin[name];
                (
                    name.clone(),
                    (origin + (goal - origin) * arrival).clamp(limit.lower, limit.upper),
                )
            })
            .collect()
    }
}

fn ease(value: f64) -> f64 {
    let t = value.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{motion::MotionController, server::read_limits};

    fn root() -> &'static Path {
        Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap()
    }
    fn scene() -> Scene {
        load_scenes(root()).unwrap()["greeting"].clone()
    }
    fn performance(emotion: &str, origin: Pose) -> Performance {
        Performance::new("test".into(), 1, scene(), emotion.into(), 0.8, 0.0, origin).unwrap()
    }

    #[test]
    fn real_fixture_catalog_has_exact_text_and_measured_audio_durations() {
        let scenes = load_scenes(root()).unwrap();
        assert_eq!(scenes.len(), 3);
        assert_eq!(scenes["greeting"].text, "Oh! You're back. I missed you.");
        assert!((scenes["greeting"].duration() - 4.408072562358276).abs() < 1e-9);
        assert!(
            load_scenes(&root().join("models/derived"))
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn phases_and_phrase_gaps_follow_wall_clock_and_skip_late_accents() {
        let limits = read_limits(root()).unwrap();
        let origin = MotionController::new(limits.clone()).unwrap().positions;
        let p = performance("Happy", origin.clone());
        assert_eq!(p.state(0.0)["phase"], "anticipating");
        assert_eq!(p.sample(0.0, &limits), origin);
        assert_eq!(p.state(LEAD_S)["phase"], "speaking");
        assert_eq!(p.state(LEAD_S)["phrase_index"], 0);
        let gap = LEAD_S + p.scene.phrases[0].duration_s + PHRASE_GAP_S / 2.0;
        assert!(p.state(gap)["phrase_index"].is_null());
        assert_eq!(p.state(gap + PHRASE_GAP_S)["phrase_index"], 1);
        assert_eq!(p.state(LEAD_S + p.scene.duration())["phase"], "settling");
        assert_eq!(p.state(100.0)["phase"], "idle");
        let late = p.sample(100.0, &limits);
        let repeated = p.sample(100.0, &limits);
        assert_eq!(
            late, repeated,
            "sampling cannot replay a queued phrase accent"
        );
        let mut c = MotionController::new(limits).unwrap();
        c.synchronize(&origin).unwrap();
        let before = c.positions.clone();
        c.stream_targets(&late, 1.0 / 30.0, p.velocity_scale())
            .unwrap();
        for (name, limit) in &c.limits {
            assert!((c.positions[name] - before[name]).abs() <= limit.velocity / 30.0 + 1e-12);
        }
    }

    #[test]
    fn all_emotions_scenes_and_limit_starts_remain_bounded_through_idle() {
        let limits = read_limits(root()).unwrap();
        let scenes = load_scenes(root()).unwrap();
        let dt = 1.0 / 30.0;
        for emotion in EMOTIONS {
            for scene in scenes.values() {
                for start in 0..3 {
                    let mut c = MotionController::new(limits.clone()).unwrap();
                    let origin: Pose = limits
                        .iter()
                        .map(|(name, l)| {
                            (
                                name.clone(),
                                match start {
                                    1 => l.lower,
                                    2 => l.upper,
                                    _ => 0.0,
                                },
                            )
                        })
                        .collect();
                    c.synchronize(&origin).unwrap();
                    let p = Performance::new(
                        "test".into(),
                        1,
                        scene.clone(),
                        emotion.into(),
                        1.0,
                        0.0,
                        origin,
                    )
                    .unwrap();
                    let mut velocities: Pose = limits.keys().map(|n| (n.clone(), 0.0)).collect();
                    for tick in 0..600 {
                        let before = c.positions.clone();
                        let targets = p.sample(tick as f64 * dt, &limits);
                        c.stream_targets(&targets, dt, p.velocity_scale()).unwrap();
                        for (name, limit) in &limits {
                            let value = c.positions[name];
                            let velocity = (value - before[name]) / dt;
                            assert!(
                                (limit.lower..=limit.upper).contains(&value),
                                "{emotion} {} {start} {name}",
                                scene.id
                            );
                            assert!(velocity.abs() <= limit.velocity * p.velocity_scale() + 1e-10);
                            assert!(
                                (velocity - velocities[name]).abs()
                                    <= limit.velocity * 3.0 * dt + 1e-9
                            );
                            velocities.insert(name.clone(), velocity);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn emotion_and_scene_both_change_motion_and_idle_stays_subtle() {
        let limits = read_limits(root()).unwrap();
        let origin = MotionController::new(limits.clone()).unwrap().positions;
        let happy = performance("Happy", origin.clone());
        for emotion in ["Sad", "Angry"] {
            let other = performance(emotion, origin.clone());
            assert!(
                happy
                    .sample(1.4, &limits)
                    .iter()
                    .any(|(n, v)| (v - other.sample(1.4, &limits)[n]).abs() > 0.05)
            );
            assert_ne!(happy.velocity_scale(), other.velocity_scale());
        }
        let scenes = load_scenes(root()).unwrap();
        for id in ["question", "reassurance"] {
            let other = Performance::new(
                "other".into(),
                1,
                scenes[id].clone(),
                "Happy".into(),
                0.8,
                0.0,
                origin.clone(),
            )
            .unwrap();
            assert!(
                happy
                    .sample(1.4, &limits)
                    .iter()
                    .any(|(n, v)| (v - other.sample(1.4, &limits)[n]).abs() > 0.025)
            );
        }
        let idle_start = happy.start_at_s + happy.scene.duration() + SETTLE_S;
        let baseline = happy.sample(idle_start, &limits);
        for tick in 0..300 {
            for (name, value) in happy.sample(idle_start + tick as f64 / 30.0, &limits) {
                assert!((value - baseline[&name]).abs() <= 0.009 + 1e-12);
            }
        }
    }

    #[test]
    fn invalid_streams_and_performance_inputs_are_rejected_without_partial_targets() {
        let limits = read_limits(root()).unwrap();
        let mut c = MotionController::new(limits).unwrap();
        c.set_targets(&[("head_pitch_joint".into(), 0.2)].into())
            .unwrap();
        let before = format!("{c:?}");
        for dt in [0.0, -0.1, f64::NAN] {
            assert!(c.stream_targets(&c.targets.clone(), dt, 0.5).is_err());
            assert_eq!(format!("{c:?}"), before);
        }
        let mut invalid = c.targets.clone();
        invalid.insert("head_pitch_joint".into(), f64::NAN);
        assert!(c.stream_targets(&invalid, 0.03, 0.5).is_err());
        assert_eq!(format!("{c:?}"), before);
        for (id, emotion, intensity) in [
            ("", "Happy", 1.0),
            ("a", "Bad", 1.0),
            ("a", "Sad", f64::NAN),
            ("a", "Happy", -0.1),
            ("a", "Happy", 1.1),
        ] {
            assert!(
                Performance::new(
                    id.into(),
                    1,
                    scene(),
                    emotion.into(),
                    intensity,
                    0.0,
                    c.positions.clone()
                )
                .is_err()
            );
        }
    }

    #[test]
    fn wav_validation_rejects_truncation_wrong_format_and_invalid_rates() {
        let phrase = scene().phrases[0].clone();
        let original = std::fs::read(root().join("performances").join(phrase.audio)).unwrap();
        assert!((wav_duration(&original).unwrap() - phrase.duration_s).abs() < 1e-10);
        assert!(wav_duration(&original[..original.len() - 2]).is_err());
        for offset in [0, 8, 20, 24, 28] {
            let mut invalid = original.clone();
            invalid[offset] ^= 0x40;
            assert!(
                wav_duration(&invalid).is_err(),
                "accepted corruption at {offset}"
            );
        }
    }
}
