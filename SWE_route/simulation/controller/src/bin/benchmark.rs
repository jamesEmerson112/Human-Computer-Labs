//! Matched headless workload: warmup, full catalog, three repeats, 31 settling frames.
use clap::Parser;
use cpu_time::ProcessTime;
use lamp_controller::{
    gestures::{Gesture, load_gestures},
    motion::{JointLimit, MotionController},
    physics::{Backend, CONTROL_DT, Physics, discover_library},
};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    fs::{self, File},
    io::{BufWriter, Write},
    path::PathBuf,
    time::Instant,
};

#[derive(Parser)]
struct Args {
    #[arg(long, default_value = "..")]
    root: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, value_enum, default_value = "mujoco")]
    backend: Backend,
    #[arg(long, default_value_t = 3)]
    repeats: usize,
    #[arg(long, default_value_t = 0.5)]
    speed: f64,
}

#[derive(Deserialize)]
struct Scene {
    limits: BTreeMap<String, JointLimit>,
}

struct Row {
    frame: usize,
    compute_ms: f64,
    contacts: i32,
    penetration: f64,
    force: Option<f64>,
    effort: Option<f64>,
    desired: Vec<f64>,
    actual: Vec<f64>,
    velocity: Vec<f64>,
}

fn percentile(values: &mut [f64], fraction: f64) -> f64 {
    values.sort_by(f64::total_cmp);
    let index = fraction * (values.len() - 1) as f64;
    let low = index.floor() as usize;
    let high = index.ceil() as usize;
    values[low] + (values[high] - values[low]) * (index - low as f64)
}

fn trial(
    physics: &mut Physics,
    limits: &BTreeMap<String, JointLimit>,
    gesture: &Gesture,
    speed: f64,
    record: bool,
) -> Result<(Value, Vec<Row>), String> {
    let mut motion = MotionController::new(limits.clone())?;
    physics.reset(&motion.positions)?;
    motion.set_speed(speed)?;
    motion.play(gesture)?;
    let mut rows = Vec::new();
    let mut timings = Vec::new();
    let mut error_sq = 0.0;
    let mut max_error: f64 = 0.0;
    let mut violation: f64 = 0.0;
    let mut velocity_ratio: f64 = 0.0;
    let mut max_contacts = 0;
    let mut contact_pairs = std::collections::BTreeSet::new();
    let mut contact_frames = 0;
    let mut penetration: f64 = 0.0;
    let mut force: f64 = 0.0;
    let mut effort: f64 = 0.0;
    let mut tail = 0;
    let start_cpu = ProcessTime::now();
    let start = Instant::now();
    for frame in 0..20000 {
        let tick = Instant::now();
        let desired = motion.step(CONTROL_DT)?;
        let actual = physics.advance(&desired, CONTROL_DT)?;
        let compute_ms = tick.elapsed().as_secs_f64() * 1000.0;
        let (_, velocities) = physics.positions();
        let diag = &physics.diagnostics;
        if record {
            timings.push(compute_ms);
            for (name, limit) in limits {
                let delta = actual[name] - desired[name];
                error_sq += delta * delta;
                max_error = max_error.max(delta.abs());
                violation = violation
                    .max(limit.lower - actual[name])
                    .max(actual[name] - limit.upper);
                velocity_ratio = velocity_ratio.max(velocities[name].abs() / limit.velocity);
            }
            max_contacts = max_contacts.max(diag.contacts);
            contact_pairs.extend(diag.pairs.iter().cloned());
            contact_frames += usize::from(diag.contacts > 0);
            penetration = penetration.max(diag.penetration_m);
            force = force.max(diag.contact_force_n.unwrap_or(0.0));
            effort = effort.max(diag.max_effort_nm.unwrap_or(0.0));
            rows.push(Row {
                frame,
                compute_ms,
                contacts: diag.contacts,
                penetration: diag.penetration_m,
                force: diag.contact_force_n,
                effort: diag.max_effort_nm,
                desired: limits.keys().map(|n| desired[n]).collect(),
                actual: limits.keys().map(|n| actual[n]).collect(),
                velocity: limits.keys().map(|n| velocities[n]).collect(),
            });
        }
        if !motion.moving() {
            if tail >= 30 {
                break;
            }
            tail += 1;
        }
        if frame == 19999 {
            return Err(format!("{} exceeded benchmark frame limit", gesture.name));
        }
    }
    let wall = start.elapsed().as_secs_f64();
    let cpu = start_cpu.elapsed().as_secs_f64();
    if !record {
        return Ok((Value::Null, rows));
    }
    let simulated = rows.len() as f64 * CONTROL_DT;
    let final_row = rows.last().unwrap();
    let final_error = final_row
        .actual
        .iter()
        .zip(&final_row.desired)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    let result = json!({
        "gesture": gesture.name, "frames": rows.len(), "simulated_s": simulated,
        "loop_wall_s": wall, "cpu_s": cpu, "cpu_ms_per_simulated_s": cpu * 1000.0 / simulated,
        "realtime_factor": simulated / wall, "compute_p50_ms": percentile(&mut timings, 0.5),
        "compute_p95_ms": percentile(&mut timings, 0.95),
        "compute_max_ms": timings.iter().copied().fold(0.0, f64::max),
        "over_budget_frames": timings.iter().filter(|t| **t > 1000.0 * CONTROL_DT).count(),
        "tracking_rmse_deg": (error_sq / (rows.len() * limits.len()) as f64).sqrt().to_degrees(),
        "tracking_max_deg": max_error.to_degrees(), "final_error_max_deg": final_error.to_degrees(),
        "max_velocity_ratio": velocity_ratio, "soft_limit_violation_max_deg": violation.to_degrees(),
        "contact_frames": contact_frames, "max_contacts": max_contacts,
        "contact_pairs": contact_pairs,
        "max_penetration_mm": penetration * 1000.0,
        "max_contact_force_n": physics.diagnostics.contact_force_n.map(|_| force),
        "max_effort_nm": physics.diagnostics.max_effort_nm.map(|_| effort),
        "physics_warning_count": physics.diagnostics.warnings,
    });
    Ok((result, rows))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    if args.repeats == 0 || !args.speed.is_finite() || !(0.0 < args.speed && args.speed <= 1.0) {
        return Err("repeats must be positive and speed must be in (0,1]".into());
    }
    fs::create_dir(&args.output)?; // Never overwrite historical evidence.
    let started = Instant::now();
    let root = args.root.canonicalize()?;
    let scene: Scene = serde_json::from_slice(&fs::read(root.join("models/derived/scene.json"))?)?;
    let library = load_gestures(
        &scene.limits.keys().cloned().collect(),
        &root.join("gestures"),
    )?;
    let mut physics = Physics::new(
        &discover_library(&root)?,
        &root.join("models/derived"),
        args.backend,
        &scene.limits,
    )?;
    let initialization_s = started.elapsed().as_secs_f64();
    let backend = serde_json::to_value(args.backend)?
        .as_str()
        .unwrap()
        .to_owned();
    let mut csv = BufWriter::new(File::create(args.output.join(format!("{backend}.csv")))?);
    write!(
        csv,
        "gesture,repeat,frame,simulation_s,compute_ms,contacts,penetration_m,contact_force_n,max_effort_nm"
    )?;
    for prefix in ["desired_rad", "actual_rad", "velocity_rad_s"] {
        for name in scene.limits.keys() {
            write!(csv, ",{prefix}.{name}")?;
        }
    }
    writeln!(csv)?;
    let mut trials = Vec::new();
    for gesture in library.values() {
        trial(&mut physics, &scene.limits, gesture, args.speed, false)?;
        for repeat in 1..=args.repeats {
            let (mut result, rows) = trial(&mut physics, &scene.limits, gesture, args.speed, true)?;
            result["backend"] = backend.clone().into();
            result["repeat"] = repeat.into();
            trials.push(result);
            for row in rows {
                write!(
                    csv,
                    "{},{},{},{},{},{},{},{},{}",
                    gesture.name,
                    repeat,
                    row.frame,
                    (row.frame + 1) as f64 * CONTROL_DT,
                    row.compute_ms,
                    row.contacts,
                    row.penetration,
                    row.force.map(|v| v.to_string()).unwrap_or_default(),
                    row.effort.map(|v| v.to_string()).unwrap_or_default()
                )?;
                for value in row.desired.iter().chain(&row.actual).chain(&row.velocity) {
                    write!(csv, ",{value}")?;
                }
                writeln!(csv)?;
            }
        }
    }
    csv.flush()?;
    fs::write(
        args.output.join(format!("{backend}.json")),
        serde_json::to_vec_pretty(&json!({
            "backend": backend, "initialization_s": initialization_s, "trials": trials,
            "repeats": args.repeats, "speed": args.speed,
            "scope": "headless control+physics; render/network excluded; CSV serialization outside trial timing",
        }))?,
    )?;
    println!("Recorded {} trials ({backend})", trials.len());
    Ok(())
}
