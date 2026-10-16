use crate::{
    gestures::{Gesture, load_gestures},
    motion::{JointLimit, MotionController},
    performance::{Performance, Scene as PerformanceScene, load_scenes},
    physics::{Backend, CONTROL_DT, CONTROL_HZ, PHYSICS_HZ, Physics, discover_library},
};
use axum::{
    Json, Router,
    extract::{
        State, WebSocketUpgrade,
        ws::{Message, WebSocket},
    },
    http::{HeaderMap, StatusCode},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::get,
    serve::ListenerExt,
};
use futures_util::{SinkExt, StreamExt, stream::SplitSink};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};
use tokio::sync::{mpsc, oneshot, watch};
use tower_http::services::{ServeDir, ServeFile};

#[derive(Clone)]
pub struct Config {
    pub root: PathBuf,
    pub library: PathBuf,
    pub backend: Backend,
    pub speed: f64,
    pub port: u16,
    pub timing_log: Option<PathBuf>,
}

impl Config {
    pub fn new(
        root: PathBuf,
        library: Option<PathBuf>,
        backend: Backend,
        speed: f64,
        port: u16,
    ) -> Result<Self, String> {
        let library = library.map(Ok).unwrap_or_else(|| discover_library(&root))?;
        Ok(Self {
            root,
            library,
            backend,
            speed,
            port,
            timing_log: None,
        })
    }
}

#[derive(Deserialize)]
struct Scene {
    limits: BTreeMap<String, JointLimit>,
}

pub fn read_limits(root: &Path) -> Result<BTreeMap<String, JointLimit>, String> {
    let path = root.join("models/derived/scene.json");
    let scene: Scene = serde_json::from_slice(
        &std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?,
    )
    .map_err(|e| e.to_string())?;
    Ok(scene.limits)
}

pub struct Runtime {
    pub controller: MotionController,
    pub physics: Physics,
    pub gestures: BTreeMap<String, Gesture>,
    config: Config,
    light_enabled: bool,
    brightness: f64,
    pub revision: u64,
    pub performance: Option<Performance>,
    scenes: BTreeMap<String, PerformanceScene>,
    used_performance_ids: BTreeSet<String>,
    performance_connection: Option<Arc<AtomicBool>>,
}

impl Runtime {
    pub fn new(config: Config) -> Result<Self, String> {
        let limits = read_limits(&config.root)?;
        let gestures = load_gestures(
            &limits.keys().cloned().collect(),
            &config.root.join("gestures"),
        )?;
        let mut controller = MotionController::new(limits.clone())?;
        controller.set_speed(config.speed)?;
        let physics = Physics::new(
            &config.library,
            &config.root.join("models/derived"),
            config.backend,
            &limits,
        )?;
        let scenes = load_scenes(&config.root)?;
        Ok(Self {
            controller,
            physics,
            gestures,
            config,
            light_enabled: true,
            brightness: 5.0,
            revision: 0,
            performance: None,
            scenes,
            used_performance_ids: BTreeSet::new(),
            performance_connection: None,
        })
    }

    pub fn step(&mut self, dt: f64) -> Result<(), String> {
        self.step_at(dt, self.physics.time() + dt)
    }

    pub fn step_at(&mut self, dt: f64, now: f64) -> Result<(), String> {
        if !now.is_finite() || now < 0.0 {
            return Err("Invalid performance clock".into());
        }
        self.cancel_disconnected()?;
        let requested = if let Some(performance) = &self.performance {
            let pose = performance.sample(now, &self.controller.limits);
            self.controller.activity = format!(
                "{} · {}",
                performance.emotion,
                performance.state(now)["phase"].as_str().unwrap()
            );
            self.controller
                .stream_targets(&pose, dt, performance.velocity_scale())?
        } else {
            self.controller.step(dt)?
        };
        self.physics.advance(&requested, dt)?;
        Ok(())
    }

    pub fn performance_state(&self, now: f64) -> Value {
        self.performance
            .as_ref()
            .map(|p| p.state(now))
            .unwrap_or(Value::Null)
    }

    fn cancel_active(&mut self) -> Result<(), String> {
        let mut controller = self.controller.clone();
        controller.synchronize(&self.physics.positions().0)?;
        self.controller = controller;
        self.performance = None;
        self.performance_connection = None;
        Ok(())
    }

    fn cancel_disconnected(&mut self) -> Result<(), String> {
        if self
            .performance_connection
            .as_ref()
            .is_some_and(|alive| !alive.load(Ordering::Acquire))
        {
            self.cancel_active()?;
        }
        Ok(())
    }

    pub fn hello(&self) -> Value {
        let gestures: Vec<Value> = self.gestures.values().map(|g| json!({"name":g.name,"label":g.label,"description":g.description,"group":g.group,"revision":g.revision})).collect();
        json!({"v":1,"type":"hello","gestures":gestures,"limits":self.controller.limits,"physics_hz":PHYSICS_HZ,"control_hz":CONTROL_HZ,"mujoco_version":"3.14.0"})
    }

    pub fn state(&self, sequence: u64, server_time_s: f64, timing: Value) -> Value {
        let (actual, velocity) = self.physics.positions();
        json!({"v":1,"type":"state","sequence":sequence,"server_time_s":server_time_s,
            "simulation_time_s":self.physics.time(),"backend":self.physics.backend,"requested":self.controller.positions,
            "actual":actual,"velocity":velocity,"activity":self.controller.activity,"speed":self.controller.speed(),
            "light":{"enabled":self.light_enabled,"brightness":self.brightness},"links":self.physics.links(),
            "diagnostics":self.physics.diagnostics,"timing":timing,"performance":self.performance_state(server_time_s)})
    }

    pub fn command(&mut self, action: Action) -> Result<(), String> {
        self.command_from(action, 0, self.physics.time())
    }

    /// Only the websocket owner can cancel by id; global Stop and valid manual
    /// overrides remain available to any local client.
    pub fn command_from(&mut self, action: Action, owner: u64, now: f64) -> Result<(), String> {
        // Build every prospective controller/model/catalog before committing it.
        // Rejected commands must leave current motion and physics untouched.
        match action {
            Action::Perform {
                performance_id,
                scene,
                emotion,
                intensity,
            } => {
                if self.used_performance_ids.contains(&performance_id) {
                    return Err("Performance id was already used; choose a fresh id".into());
                }
                if self.used_performance_ids.len() >= 16384 {
                    return Err(
                        "Performance session limit reached; restart the viewer server".into(),
                    );
                }
                let scene = self
                    .scenes
                    .get(&scene)
                    .ok_or("Unknown or unavailable prepared scene")?
                    .clone();
                let mut controller = self.controller.clone();
                controller.synchronize(&self.physics.positions().0)?;
                let performance = Performance::new(
                    performance_id.clone(),
                    owner,
                    scene,
                    emotion,
                    intensity,
                    now,
                    controller.positions.clone(),
                )?;
                controller.activity = "Anticipating speech".into();
                self.controller = controller;
                self.performance = Some(performance);
                self.performance_connection = None;
                self.used_performance_ids.insert(performance_id);
            }
            Action::CancelPerformance { performance_id } => {
                if performance_id.is_empty() || performance_id.len() > 128 {
                    return Err("Performance id must contain 1 to 128 UTF-8 bytes".into());
                }
                if self
                    .performance
                    .as_ref()
                    .is_some_and(|p| p.id == performance_id && p.owner == owner)
                {
                    self.cancel_active()?;
                }
            }
            Action::Light {
                enabled,
                brightness,
            } => {
                if !brightness.is_finite() || !(0.0..=10.0).contains(&brightness) {
                    return Err("Brightness must be between zero and ten".into());
                }
                self.light_enabled = enabled;
                self.brightness = brightness;
            }
            Action::Reload => {
                let names: BTreeSet<String> = self.controller.limits.keys().cloned().collect();
                let gestures = load_gestures(&names, &self.config.root.join("gestures"))?;
                if self.performance.is_some() {
                    self.cancel_active()?;
                }
                self.gestures = gestures;
                self.revision += 1;
            }
            Action::Mode { mode } => {
                let mut controller = self.controller.clone();
                controller.synchronize(&self.physics.positions().0)?;
                let mut physics = Physics::new(
                    &self.config.library,
                    &self.config.root.join("models/derived"),
                    mode,
                    &controller.limits,
                )?;
                physics.reset(&controller.positions)?;
                self.controller = controller;
                self.physics = physics;
                self.performance = None;
                self.performance_connection = None;
            }
            action => {
                if matches!(&action, Action::Pace { .. }) && self.performance.is_some() {
                    return Err(
                        "Pace cannot change while a performance or its attentive idle is active"
                            .into(),
                    );
                }
                let cancels_performance = !matches!(&action, Action::Pace { .. });
                let mut controller = self.controller.clone();
                if !matches!(&action, Action::Pace { .. }) {
                    controller.synchronize(&self.physics.positions().0)?;
                }
                match action {
                    Action::Play { name } => controller.play(
                        self.gestures
                            .get(&name)
                            .ok_or_else(|| format!("Unknown gesture: {name}"))?,
                    )?,
                    Action::Targets { targets } => controller.set_targets(&targets)?,
                    Action::Pace { speed } => controller.set_speed(speed)?,
                    Action::Stop => controller.stop(),
                    Action::Reset => controller.reset()?,
                    _ => unreachable!(),
                }
                self.controller = controller;
                if cancels_performance {
                    self.performance = None;
                    self.performance_connection = None;
                }
            }
        }
        Ok(())
    }
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum Action {
    Perform {
        performance_id: String,
        scene: String,
        emotion: String,
        intensity: f64,
    },
    CancelPerformance {
        performance_id: String,
    },
    Play {
        name: String,
    },
    Targets {
        targets: BTreeMap<String, f64>,
    },
    Pace {
        speed: f64,
    },
    Stop,
    Reset,
    Reload,
    Light {
        enabled: bool,
        brightness: f64,
    },
    Mode {
        mode: Backend,
    },
}

#[derive(Debug, Deserialize)]
struct Command {
    v: u8,
    id: String,
    #[serde(flatten)]
    action: Action,
}

fn parse_command(text: &str) -> Result<Command, String> {
    let value = crate::gestures::parse_unique_json(text)?;
    let fields = value.as_object().ok_or("A command must be a JSON object")?;
    let allowed: &[&str] = match fields.get("type").and_then(Value::as_str) {
        Some("perform") => &[
            "v",
            "id",
            "type",
            "performance_id",
            "scene",
            "emotion",
            "intensity",
        ],
        Some("cancel_performance") => &["v", "id", "type", "performance_id"],
        Some("play") => &["v", "id", "type", "name"],
        Some("targets") => &["v", "id", "type", "targets"],
        Some("pace") => &["v", "id", "type", "speed"],
        Some("light") => &["v", "id", "type", "enabled", "brightness"],
        Some("mode") => &["v", "id", "type", "mode"],
        Some("stop" | "reset" | "reload") => &["v", "id", "type"],
        _ => return Err("Unknown or missing command type".into()),
    };
    if let Some(key) = fields.keys().find(|key| !allowed.contains(&key.as_str())) {
        return Err(format!("Unexpected command field: {key}"));
    }
    let command: Command =
        serde_json::from_value(value).map_err(|e| format!("Invalid command: {e}"))?;
    if command.v != 1 {
        return Err("Unsupported protocol version".into());
    }
    if command.id.is_empty() || command.id.len() > 128 {
        return Err("Command id must contain 1 to 128 UTF-8 bytes".into());
    }
    Ok(command)
}

struct Request {
    command: Command,
    response: oneshot::Sender<Value>,
    owner: u64,
    connected: Arc<AtomicBool>,
}

struct ConnectionLifetime(Arc<AtomicBool>);
impl Drop for ConnectionLifetime {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
    }
}
static NEXT_OWNER: AtomicU64 = AtomicU64::new(1);

struct OwnerExit(Option<oneshot::Sender<()>>);
impl Drop for OwnerExit {
    fn drop(&mut self) {
        if let Some(sender) = self.0.take() {
            let _ = sender.send(());
        }
    }
}

/// Fixed 30 Hz time with at most eight control ticks in one owner-loop pass.
/// Every elapsed second is either simulated, pending, or explicitly discarded.
#[derive(Default)]
struct FixedClock {
    pending: f64,
    dropped: f64,
}

struct TickBudget {
    steps: u32,
    lag_s: f64,
    dropped_time_s: f64,
}

impl FixedClock {
    fn advance(&mut self, elapsed: f64) -> Result<TickBudget, String> {
        if !elapsed.is_finite() || elapsed < 0.0 || !(self.pending + elapsed).is_finite() {
            return Err("Clock elapsed time must be finite and nonnegative".into());
        }
        let mut pending = self.pending + elapsed;
        let mut steps = 0;
        while pending + 1e-12 >= CONTROL_DT && steps < 8 {
            pending = (pending - CONTROL_DT).max(0.0);
            steps += 1;
        }
        let mut dropped = self.dropped;
        if pending >= CONTROL_DT {
            let remainder = pending % CONTROL_DT;
            dropped += pending - remainder;
            pending = remainder;
        }
        if !dropped.is_finite() {
            return Err("Cumulative discarded clock time overflowed".into());
        }
        self.pending = pending;
        self.dropped = dropped;
        Ok(TickBudget {
            steps,
            lag_s: pending,
            dropped_time_s: dropped,
        })
    }
}
#[derive(Clone)]
struct Snapshot {
    hello: String,
    state: String,
    revision: u64,
}
#[derive(Clone)]
struct Shared {
    commands: mpsc::Sender<Request>,
    snapshots: watch::Receiver<Arc<Snapshot>>,
    port: u16,
    migration_path: PathBuf,
}

fn validate_review_migration(value: &Value, hello: &Value) -> Result<(), String> {
    if value.get("schema_version").and_then(Value::as_u64) != Some(1) {
        return Err("Unsupported review migration schema".into());
    }
    for key in ["source_catalog_id", "target_catalog_id"] {
        if !value
            .get(key)
            .and_then(Value::as_str)
            .is_some_and(|text| !text.is_empty() && text.len() <= 200)
        {
            return Err(format!("Migration is missing a valid {key}"));
        }
    }
    if !value.get("retired").is_some_and(Value::is_array) {
        return Err("Migration is missing retired entries".into());
    }
    let entries = value
        .get("entries")
        .and_then(Value::as_array)
        .ok_or("Migration entries must be an array")?;
    let catalog = hello
        .get("gestures")
        .and_then(Value::as_array)
        .ok_or("Gesture catalog is unavailable")?;
    let mut sources = BTreeSet::new();
    let mut targets = BTreeSet::new();
    for entry in entries {
        if entry
            .get("preserve_review")
            .and_then(Value::as_bool)
            .is_none()
            || !entry
                .get("reason")
                .and_then(Value::as_str)
                .is_some_and(|text| !text.is_empty() && text.len() <= 2000)
        {
            return Err("Migration entry needs preserve_review and reason".into());
        }
        for side in ["source", "target"] {
            let item = entry
                .get(side)
                .ok_or("Migration entry needs source and target")?;
            for key in ["name", "label", "group", "revision"] {
                let text = item
                    .get(key)
                    .and_then(Value::as_str)
                    .ok_or_else(|| format!("Migration {side} needs {key}"))?;
                if text.len() > 256 || (key != "group" && text.is_empty()) {
                    return Err(format!("Invalid migration {side} {key}"));
                }
                if key == "revision"
                    && (text.len() != 64
                        || !text
                            .bytes()
                            .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c)))
                {
                    return Err("Migration revisions must be lowercase SHA-256".into());
                }
            }
            let name = item["name"].as_str().unwrap();
            let names = if side == "source" {
                &mut sources
            } else {
                &mut targets
            };
            if !names.insert(name) {
                return Err(format!("Duplicate migration {side}: {name}"));
            }
        }
        let target = &entry["target"];
        let current = catalog
            .iter()
            .find(|gesture| gesture["name"] == target["name"])
            .ok_or_else(|| format!("Migration target is absent: {}", target["name"]))?;
        for key in ["label", "group", "revision"] {
            if current[key] != target[key] {
                return Err(format!(
                    "Migration is stale for {} ({key}); regenerate it against the current catalog",
                    target["name"]
                ));
            }
        }
    }
    Ok(())
}

async fn review_migration(State(shared): State<Shared>) -> Response {
    let result = (|| {
        let metadata = std::fs::metadata(&shared.migration_path).map_err(|error| {
            (
                if error.kind() == std::io::ErrorKind::NotFound {
                    StatusCode::NOT_FOUND
                } else {
                    StatusCode::INTERNAL_SERVER_ERROR
                },
                "Review migration is unavailable".to_owned(),
            )
        })?;
        if metadata.len() > 4 * 1024 * 1024 {
            return Err((
                StatusCode::CONFLICT,
                "Review migration exceeds the size limit".into(),
            ));
        }
        let text = std::fs::read_to_string(&shared.migration_path).map_err(|error| {
            (
                StatusCode::CONFLICT,
                format!("Cannot read review migration: {error}"),
            )
        })?;
        let migration = crate::gestures::parse_unique_json(&text).map_err(|error| {
            (
                StatusCode::CONFLICT,
                format!("Invalid review migration: {error}"),
            )
        })?;
        let hello: Value = serde_json::from_str(&shared.snapshots.borrow().hello)
            .map_err(|error| (StatusCode::SERVICE_UNAVAILABLE, error.to_string()))?;
        validate_review_migration(&migration, &hello)
            .map_err(|error| (StatusCode::CONFLICT, error))?;
        Ok::<_, (StatusCode, String)>(migration)
    })();
    match result {
        Ok(migration) => ([("cache-control", "no-store")], Json(migration)).into_response(),
        Err((status, error)) => (
            status,
            [("cache-control", "no-store")],
            Json(json!({"error":error})),
        )
            .into_response(),
    }
}

async fn local_only(
    State(shared): State<Shared>,
    request: axum::extract::Request,
    next: Next,
) -> Response {
    if !allowed_headers(request.headers(), shared.port) {
        return (
            StatusCode::FORBIDDEN,
            "This server accepts only its local browser origin",
        )
            .into_response();
    }
    next.run(request).await
}

fn allowed_headers(headers: &HeaderMap, port: u16) -> bool {
    let hosts = [format!("127.0.0.1:{port}"), format!("localhost:{port}")];
    let Some(host) = headers.get("host").and_then(|v| v.to_str().ok()) else {
        return false;
    };
    if !hosts.iter().any(|candidate| candidate == host) {
        return false;
    }
    match headers.get("origin") {
        None => true, // Local command-line clients do not send browser Origin.
        Some(origin) => origin
            .to_str()
            .is_ok_and(|origin| origin == format!("http://{host}")),
    }
}

async fn upgrade(State(shared): State<Shared>, upgrade: WebSocketUpgrade) -> Response {
    upgrade
        .max_message_size(16 * 1024)
        .max_frame_size(16 * 1024)
        .on_upgrade(move |socket| connection(socket, shared))
}

async fn socket_send(sink: &mut SplitSink<WebSocket, Message>, message: Message) -> bool {
    // A browser that stops reading must not keep a session (or shutdown) stuck.
    matches!(
        tokio::time::timeout(Duration::from_secs(2), sink.send(message)).await,
        Ok(Ok(()))
    )
}

async fn connection(socket: WebSocket, mut shared: Shared) {
    let owner = NEXT_OWNER.fetch_add(1, Ordering::Relaxed);
    let connected = Arc::new(AtomicBool::new(true));
    // This flag bypasses the bounded command queue, including on send timeout.
    let _lifetime = ConnectionLifetime(connected.clone());
    let (mut sink, mut stream) = socket.split();
    let initial = shared.snapshots.borrow_and_update().clone();
    if !socket_send(&mut sink, Message::Text(initial.hello.clone().into())).await
        || !socket_send(&mut sink, Message::Text(initial.state.clone().into())).await
    {
        return;
    }
    let mut revision = initial.revision;
    loop {
        tokio::select! {
            changed = shared.snapshots.changed() => {
                if changed.is_err() { break; }
                let snapshot = shared.snapshots.borrow_and_update().clone();
                if snapshot.revision != revision {
                    if !socket_send(&mut sink, Message::Text(snapshot.hello.clone().into())).await { break; }
                    revision = snapshot.revision;
                }
                if !socket_send(&mut sink, Message::Text(snapshot.state.clone().into())).await { break; }
            }
            message = stream.next() => {
                match message {
                    Some(Ok(Message::Text(text))) => {
                        let id = serde_json::from_str::<Value>(&text).ok().and_then(|v| v.get("id").and_then(Value::as_str).map(str::to_owned)).unwrap_or_default();
                        let reply = match parse_command(&text) {
                            Ok(command) => {
                                let (response, receive) = oneshot::channel();
                                match shared.commands.try_send(Request { command, response, owner, connected: connected.clone() }) {
                                    Ok(()) => match tokio::time::timeout(Duration::from_secs(2), receive).await {
                                        Ok(Ok(reply)) => reply,
                                        _ => json!({"v":1,"type":"ack","id":id,"ok":false,"error":"Controller did not acknowledge the command"}),
                                    },
                                    Err(_) => json!({"v":1,"type":"ack","id":id,"ok":false,"error":"Controller command queue is full"}),
                                }
                            }
                            Err(error) => json!({"v":1,"type":"ack","id":id,"ok":false,"error":error}),
                        };
                        if !socket_send(&mut sink, Message::Text(reply.to_string().into())).await { break; }
                    }
                    Some(Ok(Message::Ping(bytes))) => { if !socket_send(&mut sink, Message::Pong(bytes)).await { break; } }
                    Some(Ok(Message::Pong(_))) => {},
                    _ => break,
                }
            }
        }
    }
}

pub async fn serve(config: Config) -> Result<(), String> {
    let dist = config.root.join("viewer/dist");
    if !dist.join("index.html").is_file() {
        return Err(format!(
            "Viewer build missing at {}. Run npm ci and npm run build in {} first.",
            dist.display(),
            config.root.join("viewer").display()
        ));
    }
    if config.port == 0 {
        return Err("Choose a local port between 1 and 65535".into());
    }
    let (commands, mut requests) = mpsc::channel::<Request>(64);
    let (snapshots, receiver) = watch::channel(Arc::new(Snapshot {
        hello: String::new(),
        state: String::new(),
        revision: 0,
    }));
    let (ready_tx, ready_rx) = oneshot::channel();
    let (owner_exit_tx, owner_exit_rx) = oneshot::channel();
    let stop = Arc::new(AtomicBool::new(false));
    let owner_stop = stop.clone();
    let owner_config = config.clone();
    let owner = std::thread::Builder::new().name("lamp-physics".into()).spawn(move || -> Result<(), String> {
        let _completed = OwnerExit(Some(owner_exit_tx));
        let mut timing_log = match &owner_config.timing_log {
            None => None,
            Some(path) => {
                let file = std::fs::File::options().write(true).create_new(true).open(path).map_err(|e| format!("{}: {e}",path.display()))?;
                let mut writer = BufWriter::new(file);
                writeln!(writer,"started_s,simulation_time_s,steps,compute_ms,lag_s,dropped_time_s").map_err(|e|e.to_string())?;
                Some(writer)
            }
        };
        let mut runtime = match Runtime::new(owner_config) {
            Ok(runtime) => runtime,
            Err(error) => { let _ = ready_tx.send(Err(error.clone())); return Err(error); }
        };
        let start = Instant::now(); let mut previous = start;
        let mut clock = FixedClock::default(); let mut sequence = 0;
        let mut hello = runtime.hello().to_string(); let mut hello_revision = runtime.revision;
        let initial = runtime.state(0, 0.0, json!({"tick_compute_ms":0.0,"loop_lag_s":0.0,"dropped_time_s":0.0,"catchup_steps":0}));
        snapshots.send_replace(Arc::new(Snapshot { hello: runtime.hello().to_string(), state: initial.to_string(), revision: runtime.revision }));
        let _ = ready_tx.send(Ok(()));
        while !owner_stop.load(Ordering::Relaxed) {
            let now = Instant::now(); let elapsed = (now-previous).as_secs_f64(); previous = now;
            runtime.cancel_disconnected()?;
            for _ in 0..64 {
                let Ok(request) = requests.try_recv() else { break; };
                // A disconnected client or timed-out receiver cancels an unprocessed command.
                if request.response.is_closed() || !request.connected.load(Ordering::Acquire) { continue; }
                let is_perform = matches!(&request.command.action, Action::Perform { .. });
                let accepted_at = start.elapsed().as_secs_f64();
                let result = runtime.command_from(request.command.action, request.owner, accepted_at);
                if result.is_ok() && is_perform { runtime.performance_connection = Some(request.connected); }
                let _ = request.response.send(match result {
                    Ok(()) => json!({"v":1,"type":"ack","id":request.command.id,"ok":true,"server_time_s":start.elapsed().as_secs_f64(),"performance":runtime.performance_state(start.elapsed().as_secs_f64())}),
                    Err(error) => json!({"v":1,"type":"ack","id":request.command.id,"ok":false,"error":error,"server_time_s":start.elapsed().as_secs_f64()}),
                });
            }
            // Advance fixed control/physics time. Bounded catch-up prevents an
            // overloaded process from freezing command handling indefinitely.
            let tick_start = Instant::now();
            let TickBudget { steps, lag_s: accumulator, dropped_time_s: dropped } = clock.advance(elapsed)?;
            let performance_now = start.elapsed().as_secs_f64();
            for _ in 0..steps {
                // Catch-up advances physics, but samples the same current cue;
                // missed wall-clock accents are never rushed through.
                runtime.step_at(CONTROL_DT, performance_now)?;
                sequence += 1;
            }
            if steps > 0 {
                if runtime.revision != hello_revision { hello = runtime.hello().to_string(); hello_revision = runtime.revision; }
                let timing = json!({"tick_compute_ms":tick_start.elapsed().as_secs_f64()*1000.0/steps as f64,
                    "loop_lag_s":accumulator,"dropped_time_s":dropped,"catchup_steps":steps});
                snapshots.send_replace(Arc::new(Snapshot { hello: hello.clone(), state: runtime.state(sequence,start.elapsed().as_secs_f64(),timing).to_string(), revision: runtime.revision }));
                if let Some(writer) = &mut timing_log {
                    writeln!(writer,"{:.9},{:.9},{steps},{:.9},{accumulator:.9},{dropped:.9}", tick_start.duration_since(start).as_secs_f64(),runtime.physics.time(),tick_start.elapsed().as_secs_f64()*1000.0).map_err(|e|e.to_string())?;
                    // Persist at most one second of buffered evidence on abrupt termination.
                    if sequence % CONTROL_HZ as u64 == 0 { writer.flush().map_err(|e|e.to_string())?; }
                }
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        if let Some(writer) = &mut timing_log { writer.flush().map_err(|e|e.to_string())?; }
        Ok(())
    }).map_err(|e| e.to_string())?;
    ready_rx.await.map_err(|e| e.to_string())??;
    let shared = Shared {
        commands,
        snapshots: receiver,
        port: config.port,
        migration_path: config.root.join("catalogs/emotion-wheel-migration.json"),
    };
    let app = Router::new()
        .route("/ws", get(upgrade))
        .route("/api/review-migration", get(review_migration))
        .nest_service(
            "/performances",
            ServeDir::new(config.root.join("performances")),
        )
        .nest_service("/assets", ServeDir::new(config.root.join("models/derived")))
        .route_service(
            "/api/scene",
            ServeFile::new(config.root.join("models/derived/scene.json")),
        )
        .fallback_service(
            ServeDir::new(&dist).not_found_service(ServeFile::new(dist.join("index.html"))),
        )
        .layer(middleware::from_fn_with_state(shared.clone(), local_only))
        .with_state(shared);
    let listener =
        match tokio::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, config.port)).await {
            Ok(listener) => listener,
            Err(error) => {
                stop.store(true, Ordering::Relaxed);
                let _ = owner.join();
                return Err(error.to_string());
            }
        };
    println!(
        "Lamp viewer: http://127.0.0.1:{} (Rust / MuJoCo 3.14.0)",
        config.port
    );
    let shutdown_stop = stop.clone();
    // Small state and acknowledgment messages should not wait for TCP batching.
    let listener = listener.tap_io(|stream| {
        if let Err(error) = stream.set_nodelay(true) {
            eprintln!("Could not enable TCP_NODELAY on a local connection: {error}");
        }
    });
    let serve_result = axum::serve(listener, app)
        .with_graceful_shutdown(async move {
            tokio::select! { _ = tokio::signal::ctrl_c() => {}, _ = owner_exit_rx => {} }
            shutdown_stop.store(true, Ordering::Relaxed);
        })
        .await;
    stop.store(true, Ordering::Relaxed);
    let owner_result = owner
        .join()
        .map_err(|_| "Physics thread panicked".to_owned())?;
    serve_result.map_err(|e| e.to_string())?;
    owner_result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn performance_runtime() -> Runtime {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .to_path_buf();
        Runtime::new(Config::new(root, None, Backend::MujocoKinematic, 0.5, 8082).unwrap()).unwrap()
    }

    fn perform(id: &str) -> Action {
        Action::Perform {
            performance_id: id.into(),
            scene: "greeting".into(),
            emotion: "Happy".into(),
            intensity: 0.8,
        }
    }

    #[test]
    fn performance_validation_is_atomic_and_light_is_independent() {
        let mut runtime = performance_runtime();
        runtime.command_from(perform("first"), 1, 10.0).unwrap();
        runtime.step_at(CONTROL_DT, 10.3).unwrap();
        let before = runtime.state(1, 10.3, Value::Null);
        for invalid in [
            perform("first"),
            Action::Perform {
                performance_id: "unknown".into(),
                scene: "missing".into(),
                emotion: "Happy".into(),
                intensity: 0.8,
            },
            Action::Perform {
                performance_id: "emotion".into(),
                scene: "greeting".into(),
                emotion: "Bad".into(),
                intensity: 0.8,
            },
            Action::Perform {
                performance_id: "intensity".into(),
                scene: "greeting".into(),
                emotion: "Happy".into(),
                intensity: f64::NAN,
            },
            Action::Play {
                name: "missing".into(),
            },
            Action::Targets {
                targets: [("head_pitch_joint".into(), f64::NAN)].into(),
            },
            Action::Pace { speed: 0.8 },
        ] {
            assert!(runtime.command_from(invalid, 1, 10.3).is_err());
            assert_eq!(runtime.state(1, 10.3, Value::Null), before);
        }
        runtime
            .command(Action::Light {
                enabled: false,
                brightness: 1.0,
            })
            .unwrap();
        assert_eq!(runtime.performance.as_ref().unwrap().id, "first");
        runtime.step_at(CONTROL_DT, 50.0).unwrap();
        assert_eq!(runtime.performance_state(50.0)["phase"], "idle");
        assert!(runtime.command(Action::Pace { speed: 0.2 }).is_err());
    }

    #[test]
    fn ownership_replacement_disconnect_and_stop_never_resurrect_old_performances() {
        let mut runtime = performance_runtime();
        runtime.command_from(perform("first"), 1, 0.0).unwrap();
        runtime
            .command_from(
                Action::CancelPerformance {
                    performance_id: "first".into(),
                },
                2,
                0.1,
            )
            .unwrap();
        assert_eq!(runtime.performance.as_ref().unwrap().id, "first");
        runtime.command_from(perform("second"), 2, 0.2).unwrap();
        runtime
            .command_from(
                Action::CancelPerformance {
                    performance_id: "first".into(),
                },
                1,
                0.3,
            )
            .unwrap();
        assert_eq!(runtime.performance.as_ref().unwrap().id, "second");
        runtime
            .command_from(
                Action::CancelPerformance {
                    performance_id: "second".into(),
                },
                2,
                0.4,
            )
            .unwrap();
        assert!(runtime.performance.is_none());
        assert!(runtime.command_from(perform("second"), 2, 0.5).is_err());

        runtime.command_from(perform("disconnect"), 3, 1.0).unwrap();
        let alive = Arc::new(AtomicBool::new(true));
        runtime.performance_connection = Some(alive.clone());
        let guard = ConnectionLifetime(alive);
        drop(guard);
        runtime.step_at(CONTROL_DT, 1.1).unwrap();
        assert!(runtime.performance.is_none());
        let held = runtime.controller.positions.clone();
        for time in [2.0, 20.0, 100.0] {
            runtime.step_at(CONTROL_DT, time).unwrap();
            assert_eq!(runtime.controller.positions, held);
            assert!(runtime.performance.is_none());
        }
        runtime.command_from(perform("stop"), 1, 101.0).unwrap();
        runtime.command_from(Action::Stop, 2, 101.1).unwrap();
        assert!(runtime.performance.is_none());
        runtime.step_at(CONTROL_DT, 200.0).unwrap();
        assert!(runtime.performance.is_none());
    }

    #[test]
    fn valid_manual_overrides_cancel_active_performance_after_validation() {
        let mut runtime = performance_runtime();
        for (index, action) in [
            Action::Targets {
                targets: [("head_pitch_joint".into(), 0.15)].into(),
            },
            Action::Play { name: "nod".into() },
            Action::Reset,
            Action::Reload,
            Action::Mode {
                mode: Backend::MujocoKinematic,
            },
        ]
        .into_iter()
        .enumerate()
        {
            runtime
                .command_from(perform(&format!("override-{index}")), 1, index as f64)
                .unwrap();
            runtime.command_from(action, 2, index as f64 + 0.1).unwrap();
            assert!(runtime.performance.is_none());
        }
    }

    fn migration_fixture() -> (Value, Value) {
        let item = json!({"name":"nod","label":"Nod","group":"","revision":"a".repeat(64)});
        let hello = json!({"gestures":[item.clone()]});
        let migration = json!({"schema_version":1,"source_catalog_id":"old","target_catalog_id":"new",
            "entries":[{"source":item.clone(),"target":item,"preserve_review":true,"reason":"Unchanged"}],"retired":[]});
        (migration, hello)
    }

    #[test]
    fn review_migration_rejects_stale_revisions_duplicates_and_invalid_metadata() {
        let (migration, hello) = migration_fixture();
        assert!(validate_review_migration(&migration, &hello).is_ok());
        for key in ["name", "label", "group", "revision"] {
            let mut changed = migration.clone();
            changed["entries"][0]["target"][key] = json!(if key == "revision" {
                "b".repeat(64)
            } else {
                "changed".into()
            });
            assert!(
                validate_review_migration(&changed, &hello).is_err(),
                "accepted stale {key}"
            );
        }
        let mut duplicate = migration.clone();
        duplicate["entries"]
            .as_array_mut()
            .unwrap()
            .push(migration["entries"][0].clone());
        assert!(validate_review_migration(&duplicate, &hello).is_err());
        let mut invalid = migration.clone();
        invalid["entries"][0]["source"]["revision"] = json!("A".repeat(64));
        assert!(validate_review_migration(&invalid, &hello).is_err());
        invalid = migration.clone();
        invalid["entries"][0]["preserve_review"] = json!("true");
        assert!(validate_review_migration(&invalid, &hello).is_err());
        invalid = migration;
        invalid["schema_version"] = json!(2);
        assert!(validate_review_migration(&invalid, &hello).is_err());
    }

    #[tokio::test]
    async fn review_migration_endpoint_returns_json_and_rejects_stale_or_missing_files() {
        let (migration, hello) = migration_fixture();
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "lamp-review-test-{}-{unique}.json",
            std::process::id()
        ));
        let (commands, _receiver) = mpsc::channel(1);
        let (_sender, snapshots) = watch::channel(Arc::new(Snapshot {
            hello: hello.to_string(),
            state: String::new(),
            revision: 0,
        }));
        let shared = Shared {
            commands,
            snapshots,
            port: 8082,
            migration_path: path.clone(),
        };
        assert_eq!(
            review_migration(State(shared.clone())).await.status(),
            StatusCode::NOT_FOUND
        );
        std::fs::write(&path, migration.to_string()).unwrap();
        let response = review_migration(State(shared.clone())).await;
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(response.headers()["content-type"], "application/json");
        assert_eq!(response.headers()["cache-control"], "no-store");
        let bytes = axum::body::to_bytes(response.into_body(), 4 * 1024 * 1024)
            .await
            .unwrap();
        assert_eq!(serde_json::from_slice::<Value>(&bytes).unwrap(), migration);
        let mut stale = migration;
        stale["entries"][0]["target"]["revision"] = json!("b".repeat(64));
        std::fs::write(&path, stale.to_string()).unwrap();
        assert_eq!(
            review_migration(State(shared.clone())).await.status(),
            StatusCode::CONFLICT
        );
        std::fs::write(&path, "{invalid}").unwrap();
        assert_eq!(
            review_migration(State(shared)).await.status(),
            StatusCode::CONFLICT
        );
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn clock_keeps_fractional_time_and_exact_control_rate() {
        let mut clock = FixedClock::default();
        let mut ticks = 0;
        for _ in 0..200 {
            let budget = clock.advance(0.005).unwrap();
            ticks += budget.steps;
            assert!(budget.lag_s >= 0.0 && budget.lag_s < CONTROL_DT);
            assert_eq!(budget.dropped_time_s, 0.0);
        }
        assert_eq!(ticks, CONTROL_HZ);
        assert!(clock.pending < 1e-12);
        assert_eq!(clock.advance(CONTROL_DT / 2.0).unwrap().steps, 0);
        assert_eq!(clock.advance(CONTROL_DT / 2.0).unwrap().steps, 1);
    }

    #[test]
    fn clock_catches_up_and_reports_every_discarded_second() {
        let mut clock = FixedClock::default();
        let budget = clock.advance(0.2).unwrap();
        assert_eq!(budget.steps, 6);
        assert_eq!(budget.dropped_time_s, 0.0);
        let budget = clock.advance(0.37).unwrap();
        assert_eq!(budget.steps, 8);
        assert!((budget.dropped_time_s - 0.1).abs() < 1e-12);
        assert!((14.0 * CONTROL_DT + budget.dropped_time_s + budget.lag_s - 0.57).abs() < 1e-12);
        let budget = clock.advance(CONTROL_DT - clock.pending).unwrap();
        assert_eq!(budget.steps, 1);
        assert!((budget.dropped_time_s - 0.1).abs() < 1e-12);
    }

    #[test]
    fn invalid_clock_values_leave_state_unchanged() {
        let mut clock = FixedClock::default();
        clock.advance(0.01).unwrap();
        for elapsed in [-1.0, f64::NAN, f64::INFINITY] {
            assert!(clock.advance(elapsed).is_err());
            assert_eq!(clock.pending, 0.01);
            assert_eq!(clock.dropped, 0.0);
        }
    }

    #[test]
    fn command_schema_is_strict() {
        for valid in [
            r#"{"v":1,"id":"a","type":"stop"}"#,
            r#"{"v":1,"id":"a","type":"play","name":"nod"}"#,
        ] {
            assert!(parse_command(valid).is_ok());
        }
        for invalid in [
            r#"{"v":2,"id":"a","type":"stop"}"#,
            r#"{"v":1,"id":"","type":"stop"}"#,
            r#"{"v":1,"id":"a","type":"stop","extra":1}"#,
            r#"{"v":1,"id":"a","type":"play","name":3}"#,
            r#"{"v":1,"id":"a","id":"b","type":"stop"}"#,
            r#"{"v":1,"id":"a","type":"targets","targets":{"head_pitch_joint":0.1,"head_pitch_joint":0.2}}"#,
        ] {
            assert!(parse_command(invalid).is_err(), "{invalid}");
        }
    }
    #[test]
    fn local_host_and_origin_are_checked() {
        let mut headers = HeaderMap::new();
        headers.insert("host", "127.0.0.1:8082".parse().unwrap());
        assert!(allowed_headers(&headers, 8082));
        headers.insert("origin", "https://example.com".parse().unwrap());
        assert!(!allowed_headers(&headers, 8082));
        headers.insert("origin", "http://127.0.0.1:8082".parse().unwrap());
        assert!(allowed_headers(&headers, 8082));
        headers.insert("host", "attacker.example:8082".parse().unwrap());
        assert!(!allowed_headers(&headers, 8082));
    }
}
