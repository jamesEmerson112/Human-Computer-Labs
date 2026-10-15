use lamp_controller::{
    gestures::{Gesture, load_gestures, parse_gesture, parse_gesture_json},
    motion::{JointLimit, MotionController, Pose},
    physics::{Backend, discover_library},
    server::{Action, Config, Runtime},
};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::{Path, PathBuf},
    process::Command,
};

fn simulation() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .to_path_buf()
}

fn limits() -> BTreeMap<String, JointLimit> {
    [
        ("base_yaw_joint", -2.45, 2.45, 1.5),
        ("shoulder_pitch_joint", -0.65, 0.95, 0.95),
        ("elbow_pitch_joint", -1.7, 0.3, 1.15),
        ("neck_yaw_joint", -1.25, 1.25, 1.6),
        ("head_pitch_joint", -0.8, 0.6, 1.45),
    ]
    .into_iter()
    .map(|(name, lower, upper, velocity)| {
        (
            name.into(),
            JointLimit {
                lower,
                upper,
                velocity,
            },
        )
    })
    .collect()
}

fn names() -> BTreeSet<String> {
    limits().into_keys().collect()
}

fn definition(steps: Value) -> Value {
    json!({"schema_version":1,"name":"test","label":"Test","units":"radians","steps":steps})
}

fn gesture(steps: Value) -> Gesture {
    parse_gesture(&definition(steps), &names()).unwrap()
}

#[derive(Deserialize)]
struct Snapshot {
    positions: Vec<f64>,
    targets: Vec<f64>,
    moving: bool,
    activity: String,
}

fn check_snapshot(motion: &MotionController, expected: &Snapshot, context: &str) {
    for (index, (name, actual)) in motion.positions.iter().enumerate() {
        assert!(
            (actual - expected.positions[index]).abs() < 1e-12,
            "{context}: {name} position {actual} != {}",
            expected.positions[index]
        );
        assert!(
            (motion.targets[name] - expected.targets[index]).abs() < 1e-12,
            "{context}: {name} target differs"
        );
    }
    assert_eq!(motion.moving(), expected.moving, "{context}");
    assert_eq!(motion.activity, expected.activity, "{context}");
}

#[derive(Deserialize)]
struct ReferenceCase {
    name: String,
    start: String,
    origin: Pose,
    frames: Vec<Snapshot>,
    interrupted: Snapshot,
    stopped: Snapshot,
}

/// Python remains the reference for every authored clip, every frame, three
/// starting poses, and interruptions. LAMP_PYTHON supports CI virtualenv paths.
#[test]
fn all_gestures_match_live_python_traces_and_interruption() {
    let python = std::env::var_os("LAMP_PYTHON")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            simulation().join(if cfg!(windows) {
                ".venv/Scripts/python.exe"
            } else {
                ".venv/bin/python"
            })
        });
    let output = Command::new(&python)
        .arg(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/python_reference.py"))
        .current_dir(simulation())
        .output()
        .expect("Install the locked Python environment with uv sync --locked, or set LAMP_PYTHON");
    assert!(
        output.status.success(),
        "Python reference failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let output = String::from_utf8(output.stdout).unwrap();
    let mut lines = output.lines();
    let header: Value = serde_json::from_str(lines.next().unwrap()).unwrap();
    let limits: BTreeMap<String, JointLimit> =
        serde_json::from_value(header["limits"].clone()).unwrap();
    let gestures = load_gestures(
        &limits.keys().cloned().collect(),
        &simulation().join("gestures"),
    )
    .unwrap();
    assert_eq!(
        gestures.keys().cloned().collect::<Vec<_>>(),
        serde_json::from_value::<Vec<String>>(header["gesture_names"].clone()).unwrap(),
        "Rust and the independently loaded Python catalog must match"
    );
    let mut cases = 0;
    let mut total_frames = 0;
    let mut seen_cases = BTreeSet::new();
    let mut scripted = MotionController::new(limits.clone()).unwrap();
    for line in lines {
        let data: Value = serde_json::from_str(line).unwrap();
        if let Some(action) = data.get("action") {
            if let Some(value) = action.get("targets") {
                scripted
                    .set_targets(&serde_json::from_value(value.clone()).unwrap())
                    .unwrap();
            } else if let Some(value) = action.get("speed") {
                scripted.set_speed(value.as_f64().unwrap()).unwrap();
            } else if let Some(value) = action.get("play") {
                scripted.play(&gestures[value.as_str().unwrap()]).unwrap();
            } else if let Some(value) = action.get("dt") {
                scripted.step(value.as_f64().unwrap()).unwrap();
            } else if action.get("stop").is_some() {
                scripted.stop();
            } else if action.get("reset").is_some() {
                scripted.reset().unwrap();
            } else if let Some(value) = action.get("synchronize") {
                scripted
                    .synchronize(&serde_json::from_value(value.clone()).unwrap())
                    .unwrap();
            }
            check_snapshot(
                &scripted,
                &serde_json::from_value(data["expected"].clone()).unwrap(),
                &action.to_string(),
            );
            continue;
        }
        let case: ReferenceCase = serde_json::from_value(data).unwrap();
        assert!(
            seen_cases.insert((case.name.clone(), case.start.clone())),
            "Duplicate reference case"
        );
        let mut motion = MotionController::new(limits.clone()).unwrap();
        motion.synchronize(&case.origin).unwrap();
        motion.set_speed(0.5).unwrap();
        motion.play(&gestures[&case.name]).unwrap();
        for (index, frame) in case.frames.iter().enumerate() {
            let before = motion.positions.clone();
            motion.step(1.0 / 30.0).unwrap();
            let context = format!("{} {} frame {index}", case.name, case.start);
            check_snapshot(&motion, frame, &context);
            for (name, value) in &motion.positions {
                let limit = motion.limits[name];
                assert!(
                    *value >= limit.lower - 1e-12 && *value <= limit.upper + 1e-12,
                    "{context}: {name} exceeded limits"
                );
                assert!(
                    (value - before[name]).abs() <= limit.velocity * motion.speed() / 30.0 + 1e-12,
                    "{context}: {name} exceeded velocity"
                );
            }
        }
        assert!(!motion.moving());
        motion.synchronize(&case.origin).unwrap();
        motion.set_speed(1.0).unwrap();
        motion.play(&gestures[&case.name]).unwrap();
        motion.step(0.3).unwrap();
        check_snapshot(&motion, &case.interrupted, "interrupt");
        motion.stop();
        motion.step(60.0).unwrap();
        check_snapshot(&motion, &case.stopped, "stopped");
        total_frames += case.frames.len();
        cases += 1;
    }
    assert_eq!(cases, gestures.len() * 3);
    for name in gestures.keys() {
        for start in ["neutral", "lower", "upper"] {
            assert!(
                seen_cases.contains(&(name.clone(), start.into())),
                "Missing {name} / {start}"
            );
        }
    }
    println!(
        "Matched {cases} gesture/start cases, {total_frames} frames, {cases} interruptions, and mixed command/pace/dt sequence against Python"
    );
}

#[test]
fn invalid_steps_and_metadata_are_rejected() {
    for steps in [
        json!([]),
        json!([{"targets":{}}]),
        json!([{"targets":{"typo_joint":0.1}}]),
        json!([{"targets":{"head_pitch_joint":true}}]),
        json!([{"targets":{"head_pitch_joint":"0.3"}}]),
        json!([{"targets":{"head_pitch_joint":0.1},"hold_seconds":1}]),
        json!([{"hold_seconds":0}]),
        json!([{"hold_seconds":-1}]),
        json!([{"hold_seconds":61}]),
        json!([{"hold_seconds":true}]),
        json!([{"return_to_start":false}]),
        json!([{"return_to_start":1}]),
        json!([{"offsets":{"head_pitch_joint":0.3},"on_limit":"ignore"}]),
        json!([{"targets":{"head_pitch_joint":0.3},"on_limit":"reverse"}]),
        json!([{"target":{"head_pitch_joint":0.3}}]),
        json!(vec![json!({"hold_seconds":0.1}); 129]),
    ] {
        assert!(
            parse_gesture(&definition(steps.clone()), &names()).is_err(),
            "accepted {steps}"
        );
    }
    for (key, value) in [
        ("schema_version", json!(2)),
        ("schema_version", json!(true)),
        ("units", json!("degrees")),
        ("name", json!("../nod")),
        ("name", json!("a".repeat(65))),
        ("label", json!(" ")),
        ("unexpected", json!(1)),
        ("group", json!(" ")),
        ("group", json!(7)),
        ("group", json!("a".repeat(61))),
        ("description", json!("a".repeat(501))),
    ] {
        let mut data = definition(json!([{"hold_seconds":1}]));
        data[key] = value;
        assert!(parse_gesture(&data, &names()).is_err(), "accepted {data}");
    }
}

#[test]
fn duplicate_keys_and_nonfinite_json_are_rejected_at_every_depth() {
    for text in [
        r#"{"name":"one","name":"two"}"#,
        r#"{"schema_version":1,"name":"test","label":"Test","units":"radians","steps":[{"offsets":{"head_pitch_joint":0.1,"head_pitch_joint":0.2}}]}"#,
        r#"{"schema_version":1,"name":"test","label":"Test","units":"radians","steps":[{"hold_seconds":0.1,"hold_seconds":0.2}]}"#,
        r#"{"steps":[{"hold_seconds":NaN}]}"#,
        r#"{"steps":[{"hold_seconds":1e999}]}"#,
        r#"{"steps":[}"#,
    ] {
        assert!(
            parse_gesture_json(text, &names()).is_err(),
            "accepted {text}"
        );
    }
    assert!(
        parse_gesture_json(
            &format!("\u{feff}{}", definition(json!([{"hold_seconds":0.1}]))),
            &names()
        )
        .is_ok()
    );
}

#[test]
fn library_reload_rejects_duplicates_and_preserves_prior_library() {
    struct TempDirectory(PathBuf);
    impl Drop for TempDirectory {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let directory = TempDirectory(
        std::env::temp_dir().join(format!("lamp-gesture-test-{}-{unique}", std::process::id())),
    );
    std::fs::create_dir(&directory.0).unwrap();
    assert!(
        load_gestures(&names(), &directory.0)
            .unwrap_err()
            .contains("No gesture")
    );
    let mut data = definition(json!([{"targets":{"head_pitch_joint":0.1}}]));
    let first = directory.0.join("first.json");
    let second = directory.0.join("second.json");
    std::fs::write(&first, data.to_string()).unwrap();
    let original = load_gestures(&names(), &directory.0).unwrap();
    assert_eq!(original["test"].revision.len(), 64);
    // Re-reading unchanged bytes preserves the revision; even formatting changes
    // must invalidate an approval tied to exact authored bytes.
    assert_eq!(
        load_gestures(&names(), &directory.0).unwrap()["test"].revision,
        original["test"].revision
    );
    std::fs::write(&first, format!("{}\n", data)).unwrap();
    let reformatted = load_gestures(&names(), &directory.0).unwrap();
    assert_ne!(reformatted["test"].revision, original["test"].revision);
    assert_eq!(
        reformatted["test"].steps[0].angles,
        original["test"].steps[0].angles
    );
    std::fs::write(&second, data.to_string()).unwrap();
    assert!(
        load_gestures(&names(), &directory.0)
            .unwrap_err()
            .contains("duplicate gesture name")
    );
    data["name"] = json!("another");
    std::fs::write(&second, data.to_string()).unwrap();
    assert!(
        load_gestures(&names(), &directory.0)
            .unwrap_err()
            .contains("duplicate gesture label")
    );
    data["label"] = json!("Another");
    data["steps"][0]["targets"]["head_pitch_joint"] = json!(0.2);
    std::fs::write(&second, data.to_string()).unwrap();
    let updated = load_gestures(&names(), &directory.0).unwrap();
    assert_eq!(updated.len(), 2);
    assert_eq!(original["test"].steps[0].angles[0].1, 0.1);
    assert_eq!(updated["another"].steps[0].angles[0].1, 0.2);
    std::fs::write(&second, "{bad JSON").unwrap();
    assert!(
        load_gestures(&names(), &directory.0)
            .unwrap_err()
            .contains("second.json")
    );
    assert_eq!(updated.len(), 2);
}

#[test]
fn runtime_reload_publishes_changed_file_revision_and_rejects_invalid_replacement() {
    struct TempDirectory(PathBuf);
    impl Drop for TempDirectory {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let directory = TempDirectory(
        std::env::temp_dir().join(format!("lamp-runtime-test-{}-{unique}", std::process::id())),
    );
    let models = directory.0.join("models/derived");
    std::fs::create_dir_all(&models).unwrap();
    std::fs::create_dir(directory.0.join("gestures")).unwrap();
    for name in ["scene.json", "lamp.xml", "lamp_shade.stl"] {
        std::fs::copy(
            simulation().join("models/derived").join(name),
            models.join(name),
        )
        .unwrap();
    }
    let path = directory.0.join("gestures/test.json");
    let data = definition(json!([{"hold_seconds":0.1}])).to_string();
    std::fs::write(&path, &data).unwrap();
    let config = Config::new(
        directory.0.clone(),
        Some(discover_library(&simulation()).unwrap()),
        Backend::MujocoKinematic,
        0.5,
        8082,
    )
    .unwrap();
    let mut runtime = Runtime::new(config).unwrap();
    let original = runtime.hello()["gestures"][0]["revision"].clone();
    std::fs::write(&path, format!("{data}\n")).unwrap();
    runtime.command(Action::Reload).unwrap();
    let refreshed = runtime.hello()["gestures"][0]["revision"].clone();
    assert_ne!(original, refreshed);
    assert_eq!(runtime.revision, 1);
    std::fs::write(&path, "{invalid}").unwrap();
    assert!(runtime.command(Action::Reload).is_err());
    assert_eq!(runtime.hello()["gestures"][0]["revision"], refreshed);
    assert_eq!(runtime.revision, 1);
}

#[test]
fn manual_targets_clamp_and_respect_joint_velocity() {
    for target in [-100.0, 100.0] {
        let mut motion = MotionController::new(limits()).unwrap();
        motion
            .set_targets(&names().into_iter().map(|name| (name, target)).collect())
            .unwrap();
        for _ in 0..600 {
            let before = motion.positions.clone();
            motion.step(1.0 / 120.0).unwrap();
            for (name, value) in &motion.positions {
                let limit = motion.limits[name];
                assert!(*value >= limit.lower && *value <= limit.upper);
                assert!((value - before[name]).abs() <= limit.velocity / 120.0 + 1e-12);
            }
        }
        assert!(!motion.moving());
        for (name, value) in &motion.positions {
            assert_eq!(
                *value,
                if target < 0.0 {
                    motion.limits[name].lower
                } else {
                    motion.limits[name].upper
                }
            );
        }
    }
}

#[test]
fn invalid_commands_preserve_active_motion() {
    let mut motion = MotionController::new(limits()).unwrap();
    motion
        .play(&gesture(
            json!([{"offsets":{"head_pitch_joint":0.2}},{"return_to_start":true}]),
        ))
        .unwrap();
    motion.step(0.1).unwrap();
    let baseline = motion.clone();
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(
            motion
                .set_targets(&BTreeMap::from([("head_pitch_joint".into(), value)]))
                .is_err()
        );
        assert!(motion.set_speed(value).is_err());
        assert!(motion.step(value).is_err());
        let mut observed = motion.positions.clone();
        observed.insert("head_pitch_joint".into(), value);
        assert!(motion.synchronize(&observed).is_err());
    }
    assert!(motion.step(-0.1).is_err());
    for value in [0.0, -1.0, 1.1] {
        assert!(motion.set_speed(value).is_err());
    }
    assert!(
        motion
            .set_targets(&BTreeMap::from([("unknown".into(), 0.1)]))
            .is_err()
    );
    assert!(motion.synchronize(&Pose::new()).is_err());
    let mut bad = gesture(json!([{"hold_seconds":1}]));
    bad.steps.push(crate_step("unknown", 0.1));
    assert!(motion.play(&bad).is_err());
    bad.steps[1] = crate_step("hold_seconds", f64::NAN);
    assert!(motion.play(&bad).is_err());
    let mut baseline = baseline;
    for _ in 0..60 {
        assert_eq!(
            motion.step(1.0 / 30.0).unwrap(),
            baseline.step(1.0 / 30.0).unwrap()
        );
        assert_eq!(motion.activity, baseline.activity);
    }
}

fn crate_step(kind: &str, seconds: f64) -> lamp_controller::gestures::GestureStep {
    lamp_controller::gestures::GestureStep {
        kind: kind.into(),
        seconds,
        angles: vec![],
        on_limit: "clamp".into(),
    }
}

#[test]
fn offsets_hold_and_large_dt_keep_python_segment_semantics() {
    let mut motion = MotionController::new(limits()).unwrap();
    motion
        .set_targets(&BTreeMap::from([("head_pitch_joint".into(), 0.1)]))
        .unwrap();
    motion.step(10.0).unwrap();
    motion
        .play(&gesture(json!([
            {"offsets":{"head_pitch_joint":0.2}}, {"targets":{"base_yaw_joint":0.4}},
            {"offsets":{"head_pitch_joint":-0.2}}
        ])))
        .unwrap();
    motion.step(0.25).unwrap();
    assert!((motion.positions["head_pitch_joint"] - 0.3).abs() < 1e-12);
    motion.step(0.4).unwrap();
    assert!((motion.positions["base_yaw_joint"] - 0.4).abs() < 1e-12);
    assert!((motion.positions["head_pitch_joint"] - 0.3).abs() < 1e-12);
    motion.step(10.0).unwrap();
    assert!((motion.positions["head_pitch_joint"] + 0.1).abs() < 1e-12);
    motion.reset().unwrap();
    motion.step(10.0).unwrap();
    motion
        .play(&gesture(json!([
            {"targets":{"head_pitch_joint":0.1},"duration_seconds":0.5},
            {"hold_seconds":0.05}, {"return_to_start":true}
        ])))
        .unwrap();
    motion.step(0.5).unwrap();
    motion.step(0.04).unwrap();
    assert_eq!(motion.positions["head_pitch_joint"], 0.1);
    motion.step(0.02).unwrap();
    assert!(motion.positions["head_pitch_joint"] < 0.1);
    motion.step(10.0).unwrap();
    assert!(!motion.moving());
    assert_eq!(motion.positions["head_pitch_joint"], 0.0);
}
