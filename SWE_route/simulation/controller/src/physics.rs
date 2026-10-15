//! Single-owner native MuJoCo state. The C bridge checks header/library versions
//! before touching any model/data struct; Rust never reproduces their layouts.
use crate::motion::{JointLimit, Pose};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    ffi::{CStr, CString, c_char, c_int, c_void},
    path::{Path, PathBuf},
    ptr::NonNull,
};

pub const PHYSICS_HZ: u32 = 240;
pub const CONTROL_HZ: u32 = 30;
pub const CONTROL_DT: f64 = 1.0 / CONTROL_HZ as f64;

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, clap::ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum Backend {
    #[default]
    Mujoco,
    MujocoSource,
    MujocoKinematic,
}

#[derive(Default)]
#[repr(C)]
struct NativeDiagnostics {
    contacts: c_int,
    warnings: c_int,
    penetration_m: f64,
    contact_force_n: f64,
    max_effort_nm: f64,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct Diagnostics {
    pub contacts: i32,
    pub penetration_m: f64,
    pub contact_force_n: Option<f64>,
    pub max_effort_nm: Option<f64>,
    pub warnings: i32,
    pub tracking_error_rad: f64,
    pub pairs: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct LinkPose {
    pub position: [f64; 3],
    pub quaternion: [f64; 4],
}

unsafe extern "C" {
    fn lamp_open(
        library: *const c_char,
        xml: *const c_char,
        names: *const *const c_char,
        lower: *const f64,
        upper: *const f64,
        error: *mut c_char,
        error_size: c_int,
    ) -> *mut c_void;
    fn lamp_close(handle: *mut c_void);
    fn lamp_reset(handle: *mut c_void, positions: *const f64);
    fn lamp_advance(
        handle: *mut c_void,
        targets: *const f64,
        dt: f64,
        kinematic: c_int,
        diagnostics: *mut NativeDiagnostics,
    ) -> c_int;
    fn lamp_positions(handle: *mut c_void, positions: *mut f64, velocity: *mut f64);
    fn lamp_time(handle: *mut c_void) -> f64;
    fn lamp_body_count(handle: *mut c_void) -> c_int;
    fn lamp_pair_count(handle: *mut c_void) -> c_int;
    fn lamp_pair_body(handle: *mut c_void, pair: c_int, side: c_int) -> c_int;
    fn lamp_body_name(handle: *mut c_void, id: c_int) -> *const c_char;
    fn lamp_body_pose(handle: *mut c_void, id: c_int, position: *mut f64, quaternion: *mut f64);
}

pub struct Physics {
    handle: NonNull<c_void>,
    names: Vec<String>,
    pub backend: Backend,
    pub diagnostics: Diagnostics,
}

fn c_path(path: &Path) -> Result<CString, String> {
    CString::new(path.to_str().ok_or("Native paths must be valid UTF-8")?)
        .map_err(|_| "Path contains NUL".into())
}

pub fn discover_library(simulation: &Path) -> Result<PathBuf, String> {
    if let Some(path) = std::env::var_os("MUJOCO_LIBRARY") {
        return Ok(PathBuf::from(path));
    }
    let mut roots = Vec::new();
    if let Some(path) = std::env::var_os("MUJOCO_DIR") {
        roots.push(PathBuf::from(path));
    }
    roots.push(simulation.join(".venv/Lib/site-packages/mujoco"));
    if let Ok(dirs) = std::fs::read_dir(simulation.join(".venv/lib")) {
        for dir in dirs.flatten() {
            roots.push(dir.path().join("site-packages/mujoco"));
        }
    }
    let files = [
        "mujoco.dll",
        "libmujoco.so.3.14.0",
        "libmujoco.so",
        "libmujoco.3.14.0.dylib",
        "libmujoco.dylib",
    ];
    for root in roots {
        for prefix in [root.clone(), root.join("lib"), root.join("bin")] {
            for file in files {
                let path = prefix.join(file);
                if path.is_file() {
                    return Ok(path);
                }
            }
        }
    }
    Err("MuJoCo 3.14.0 shared library missing; set MUJOCO_LIBRARY or MUJOCO_DIR, or run uv sync --locked".into())
}

impl Physics {
    pub fn new(
        library: &Path,
        models: &Path,
        backend: Backend,
        limits: &BTreeMap<String, JointLimit>,
    ) -> Result<Self, String> {
        if limits.len() != 5 {
            return Err("The lamp model requires exactly five joint limits".into());
        }
        let names: Vec<String> = limits.keys().cloned().collect();
        let c_names: Vec<CString> = names
            .iter()
            .map(|n| CString::new(n.as_str()).map_err(|_| "Joint name contains NUL"))
            .collect::<Result<_, _>>()?;
        let pointers: Vec<*const c_char> = c_names.iter().map(|n| n.as_ptr()).collect();
        let lower: Vec<f64> = names.iter().map(|n| limits[n].lower).collect();
        let upper: Vec<f64> = names.iter().map(|n| limits[n].upper).collect();
        let xml = models.join(if matches!(backend, Backend::MujocoSource) {
            "lamp-source.xml"
        } else {
            "lamp.xml"
        });
        let library = c_path(library)?;
        let xml = c_path(&xml)?;
        let mut error = [0 as c_char; 2048];
        // All arrays contain exactly five elements and remain alive for the call.
        let handle = unsafe {
            lamp_open(
                library.as_ptr(),
                xml.as_ptr(),
                pointers.as_ptr(),
                lower.as_ptr(),
                upper.as_ptr(),
                error.as_mut_ptr(),
                error.len() as c_int,
            )
        };
        let handle = NonNull::new(handle).ok_or_else(|| {
            unsafe { CStr::from_ptr(error.as_ptr()) }
                .to_string_lossy()
                .into_owned()
        })?;
        Ok(Self {
            handle,
            names,
            backend,
            diagnostics: Diagnostics::default(),
        })
    }

    fn values(&self, pose: &Pose) -> Result<[f64; 5], String> {
        let mut values = [0.0; 5];
        for (i, name) in self.names.iter().enumerate() {
            values[i] = *pose
                .get(name)
                .ok_or_else(|| format!("Missing joint {name}"))?;
            if !values[i].is_finite() {
                return Err("Joint positions must be finite".into());
            }
        }
        Ok(values)
    }

    pub fn reset(&mut self, pose: &Pose) -> Result<(), String> {
        let values = self.values(pose)?;
        unsafe {
            lamp_reset(self.handle.as_ptr(), values.as_ptr());
        }
        self.diagnostics = Diagnostics::default();
        Ok(())
    }

    pub fn advance(&mut self, targets: &Pose, dt: f64) -> Result<Pose, String> {
        if !dt.is_finite() || !(0.0..=1.0).contains(&dt) {
            return Err("Physics dt must be finite and between zero and one second".into());
        }
        let values = self.values(targets)?;
        let mut diag = NativeDiagnostics::default();
        let kinematic = matches!(self.backend, Backend::MujocoKinematic);
        if unsafe {
            lamp_advance(
                self.handle.as_ptr(),
                values.as_ptr(),
                dt,
                kinematic as c_int,
                &mut diag,
            )
        } == 0
        {
            return Err("MuJoCo produced a nonfinite state".into());
        }
        let (actual, _) = self.positions();
        let mut pairs = Vec::new();
        for pair in 0..unsafe { lamp_pair_count(self.handle.as_ptr()) } {
            let mut bodies = Vec::new();
            for side in 0..2 {
                let id = unsafe { lamp_pair_body(self.handle.as_ptr(), pair, side) };
                let name = unsafe { lamp_body_name(self.handle.as_ptr(), id) };
                bodies.push(if name.is_null() {
                    "world".to_owned()
                } else {
                    unsafe { CStr::from_ptr(name) }
                        .to_string_lossy()
                        .into_owned()
                });
            }
            bodies.sort();
            pairs.push(bodies.join(" / "));
        }
        pairs.sort();
        self.diagnostics = Diagnostics {
            contacts: diag.contacts,
            penetration_m: diag.penetration_m,
            contact_force_n: (!kinematic).then_some(diag.contact_force_n),
            max_effort_nm: (!kinematic).then_some(diag.max_effort_nm),
            warnings: diag.warnings,
            tracking_error_rad: actual
                .iter()
                .map(|(n, v)| (v - targets[n]).abs())
                .fold(0.0, f64::max),
            pairs,
        };
        Ok(actual)
    }

    pub fn positions(&self) -> (Pose, Pose) {
        let mut positions = [0.0; 5];
        let mut velocities = [0.0; 5];
        unsafe {
            lamp_positions(
                self.handle.as_ptr(),
                positions.as_mut_ptr(),
                velocities.as_mut_ptr(),
            );
        }
        (
            self.names.iter().cloned().zip(positions).collect(),
            self.names.iter().cloned().zip(velocities).collect(),
        )
    }
    pub fn time(&self) -> f64 {
        unsafe { lamp_time(self.handle.as_ptr()) }
    }
    pub fn links(&self) -> BTreeMap<String, LinkPose> {
        let mut links = BTreeMap::new();
        for id in 0..unsafe { lamp_body_count(self.handle.as_ptr()) } {
            let name = unsafe { lamp_body_name(self.handle.as_ptr(), id) };
            if name.is_null() {
                continue;
            }
            let name = unsafe { CStr::from_ptr(name) }
                .to_string_lossy()
                .into_owned();
            let mut pose = LinkPose {
                position: [0.0; 3],
                quaternion: [0.0; 4],
            };
            unsafe {
                lamp_body_pose(
                    self.handle.as_ptr(),
                    id,
                    pose.position.as_mut_ptr(),
                    pose.quaternion.as_mut_ptr(),
                );
            }
            links.insert(name, pose);
        }
        links
    }
}

impl Drop for Physics {
    fn drop(&mut self) {
        unsafe {
            lamp_close(self.handle.as_ptr());
        }
    }
}
