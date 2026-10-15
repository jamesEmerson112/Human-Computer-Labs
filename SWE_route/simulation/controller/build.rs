use std::{env, path::PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=native/bridge.c");
    println!("cargo:rerun-if-env-changed=MUJOCO_DIR");
    let simulation = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap())
        .parent()
        .unwrap()
        .to_owned();
    let mut roots = Vec::new();
    if let Some(path) = env::var_os("MUJOCO_DIR") {
        roots.push(PathBuf::from(path));
    }
    roots.push(simulation.join(".venv/Lib/site-packages/mujoco"));
    if let Ok(dirs) = std::fs::read_dir(simulation.join(".venv/lib")) {
        for dir in dirs.flatten() {
            roots.push(dir.path().join("site-packages/mujoco"));
        }
    }
    let include = roots.iter().flat_map(|root| [root.join("include"), root.join("include/mujoco")])
        .find(|p| p.join("mujoco/mujoco.h").is_file())
        .expect("MuJoCo 3.14.0 headers missing: run uv sync --locked, or set MUJOCO_DIR to its SDK directory");
    cc::Build::new()
        .file("native/bridge.c")
        .include(include)
        .warnings(true)
        .compile("lamp_mujoco_bridge");
    if env::var("CARGO_CFG_TARGET_OS").unwrap() == "linux" {
        println!("cargo:rustc-link-lib=dl");
    }
}
