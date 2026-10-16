use clap::{Parser, Subcommand, ValueEnum};
use lamp_controller::{
    physics::{Backend, CONTROL_DT},
    server::{Action, Config, Runtime, serve},
};
use serde_json::json;
use std::{
    fs::File,
    io::{BufWriter, Write},
    path::PathBuf,
    time::Instant,
};

#[derive(Parser)]
#[command(about = "Local lamp controller with native MuJoCo 3.14.0 physics")]
struct Cli {
    #[arg(long, global = true, default_value = concat!(env!("CARGO_MANIFEST_DIR"), "/.."))]
    root: PathBuf,
    #[arg(long, global = true)]
    mujoco_library: Option<PathBuf>,
    #[arg(long, global = true, value_enum, default_value = "mujoco")]
    backend: Backend,
    #[arg(long, global = true, default_value_t = 1.0)]
    speed: f64,
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// Serve the built Three.js viewer and versioned WebSocket protocol.
    Serve {
        #[arg(long, default_value_t = 8082)]
        port: u16,
        #[arg(long)]
        timing_log: Option<PathBuf>,
    },
    /// Validate model/library loading and the complete gesture catalog.
    Check,
    /// Emit a deterministic 30 Hz requested/actual trace; no real-time sleeps.
    Trace {
        #[arg(long, default_value = "nod")]
        gesture: String,
        #[arg(long, value_enum, default_value = "neutral")]
        start: Start,
        #[arg(long)]
        frames: Option<usize>,
        /// Zero-based frame before which Stop is applied to the observed pose.
        #[arg(long)]
        stop_frame: Option<usize>,
        #[arg(long)]
        output: PathBuf,
    },
}

#[derive(Clone, Copy, ValueEnum)]
enum Start {
    Neutral,
    Lower,
    Upper,
}

#[tokio::main]
async fn main() {
    if let Err(error) = run(Cli::parse()).await {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

async fn run(cli: Cli) -> Result<(), String> {
    let command = cli.command.unwrap_or(Command::Serve {
        port: 8082,
        timing_log: None,
    });
    let port = if let Command::Serve { port, .. } = command {
        port
    } else {
        8082
    };
    let mut config = Config::new(cli.root, cli.mujoco_library, cli.backend, cli.speed, port)?;
    if let Command::Serve { timing_log, .. } = &command {
        config.timing_log = timing_log.clone();
    }
    match command {
        Command::Serve { .. } => serve(config).await,
        Command::Check => {
            let runtime = Runtime::new(config)?;
            println!(
                "MuJoCo 3.14.0 loaded; {} gestures, {} joints, {} body frames",
                runtime.gestures.len(),
                runtime.controller.limits.len(),
                runtime.physics.links().len()
            );
            Ok(())
        }
        Command::Trace {
            gesture,
            start,
            frames,
            stop_frame,
            output,
        } => {
            let mut runtime = Runtime::new(config)?;
            let pose = runtime
                .controller
                .limits
                .iter()
                .map(|(name, limit)| {
                    (
                        name.clone(),
                        match start {
                            Start::Neutral => 0.0,
                            Start::Lower => limit.lower,
                            Start::Upper => limit.upper,
                        },
                    )
                })
                .collect();
            runtime.physics.reset(&pose)?;
            runtime.controller.synchronize(&pose)?;
            runtime.command(Action::Play { name: gesture })?;
            if let Some(parent) = output.parent().filter(|p| !p.as_os_str().is_empty()) {
                std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
            }
            // Existing traces are evidence; use a new filename for another run.
            let file = File::options()
                .write(true)
                .create_new(true)
                .open(&output)
                .map_err(|e| format!("{}: {e}", output.display()))?;
            let mut writer = BufWriter::new(file);
            let mut tail = 0;
            for frame in 0..frames.unwrap_or(100_000) {
                if stop_frame == Some(frame) {
                    runtime.command(Action::Stop)?;
                }
                let started = Instant::now();
                runtime.step(CONTROL_DT)?;
                let compute_ms = started.elapsed().as_secs_f64() * 1000.0;
                let (actual, velocity) = runtime.physics.positions();
                let row = json!({"frame":frame,"simulation_time_s":runtime.physics.time(),"requested":runtime.controller.positions,
                    "actual":actual,"velocity":velocity,"compute_ms":compute_ms,"diagnostics":runtime.physics.diagnostics,"activity":runtime.controller.activity,"moving":runtime.controller.moving()});
                writeln!(writer, "{row}").map_err(|e| e.to_string())?;
                if frames.is_none() && !runtime.controller.moving() {
                    if tail >= 30 {
                        break;
                    }
                    tail += 1;
                }
            }
            writer.flush().map_err(|e| e.to_string())?;
            println!("Wrote {}", output.display());
            Ok(())
        }
    }
}
