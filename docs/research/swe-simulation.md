# SWE route: simulating or visualizing the lamp on a CPU-only laptop

Researched 2026-10-01. Figures marked "estimate" are my own judgement. Figures marked "measured" come from loading the supplied URDF on the development machine (Ryzen 5 2600X, Windows 11, Python 3.13), not on the 4-core Ubuntu target, so treat them as relative rather than absolute.

## Bottom line

None of the viable options needs a GPU or a Runpod machine. The robot is five joints and about 22 primitive shapes, and every tool below either runs its physics on the CPU or does no physics at all. The target says "no discrete GPU", which still leaves integrated graphics with Mesa's OpenGL driver. If even that is missing, llvmpipe (Mesa's CPU software rasterizer) is the fallback; a browser or MuJoCo scene this small should stay usable on it (estimate), while Gazebo and Webots fare worst.

The strongest choices are a browser viewer driven from Python (viser), MuJoCo with a lightly patched copy of the URDF, and, if ROS credibility matters, ROS 2 Jazzy with RViz2. Kinematic playback with limit and velocity clamping is enough for this challenge, and physics is an optional upgrade.

## What this URDF does to each tool

Four properties of the supplied file decide most of the comparison. The lamp shade is an ASCII STL (360 triangles, 74 KB) referenced by a bare relative path, `assets/lamp_shade.stl`. The base has no fixed world joint. The three semantic links (`light_emitter_link`, `camera_link`, `speaker_link`) have no `<inertial>` block. URDF has no light element, so the only hint of light is the `fixture_light` material (RGBA 1.0, 0.95, 0.76, 0.95) on a disc and a sphere.

MuJoCo 3.14.0 loads the unmodified file without any error, but it silently drops every visual geom and colour (measured: 6 collision cylinders, 0 meshes). That happens because `discardvisual` and `fusestatic` default to true for URDF ([XML reference](https://mujoco.readthedocs.io/en/stable/XMLreference.html#compiler)). Setting `discardvisual` through `MjSpec` (MuJoCo's programmatic model editor) after parsing did not bring the visuals back, though setting `fusestatic` there did restore the fixed-joint bodies (measured). A derived copy with a `<mujoco><compiler discardvisual="false" fusestatic="false"/></mujoco>` block ([URDF extensions](https://mujoco.readthedocs.io/en/stable/modeling.html#curdf)) then fails on the shade with "stl_decoder: number of faces should be between 1 and 200000", because MuJoCo reads only binary STL. After a 20-line conversion to binary STL, the model loads with 11 bodies, 22 geoms and the mesh (measured). The relative mesh path resolves because version 3.7.0 (April 2026) stopped forcing `strippath` on URDF ([changelog](https://mujoco.readthedocs.io/en/stable/changelog.html)). The root body is welded to the world, so the missing world joint costs nothing.

PyBullet also fails on the ASCII shade ("cannot extract anything useful from mesh"), and it gives each massless semantic link a mass of 1 kg (measured). That adds 2 kg at the lamp head under physics, which is harmless for kinematic playback. RViz passes any mesh path that is not `package://` straight to libcurl ([resource_retriever source](https://github.com/ros/resource_retriever/blob/jazzy/resource_retriever/src/retriever.cpp)), so the bare relative path fails and needs a rewritten copy. Gazebo's URDF converter merges fixed-joint children into their parent unless `preserveFixedJoint` is set ([SDFormat URDF extensions](https://sdformat.org/tutorials/specification/sdformat_urdf_extensions/1.6/)), which erases the semantic frames. The browser loaders read the file as supplied: yourdfpy parsed the ASCII shade (measured), and three.js's STLLoader "supports both binary and ASCII encoded files" ([source](https://github.com/mrdoob/three.js/blob/dev/examples/jsm/loaders/STLLoader.js)).

## Comparison

In the table, RSS is resident memory, the RAM a process actually holds. An sdist is a source package that pip must compile locally. A JIT compile is code generation that happens on first run rather than at install time.

| Option | Runs without a GPU on 24.04 | Loads this URDF as supplied | Light | Link to character software | Download size | CPU and RAM |
|---|---|---|---|---|---|---|
| viser 1.1.1 + yourdfpy | Yes; the browser renders through WebGL | Yes | `add_light_spot` with colour, intensity, cone angle | Python API, built-in WebSocket | 3.6 MB wheel; 340 MB venv with trimesh and scipy (measured) | 126 MB server RSS (measured); browser tab 200–400 MB (estimate) |
| urdf-loader 0.13.1 + three.js | Yes, through WebGL | Yes | three.js SpotLight | Your own WebSocket bridge | 0.3 MB + three.js 20 MB unpacked | Browser tab 200–400 MB (estimate) |
| MuJoCo 3.14.0 | Yes; physics on CPU, viewer needs OpenGL | No; needs a binary STL and the compiler block | Spot light attached to a body, changeable at runtime | In-process Python API | 28.7 MB wheel, 59 MB installed (measured) | 72,000 steps/s, 56 MB RSS headless (measured) |
| PyBullet 3.2.7 | Builds from source on Python 3.12 | No; the shade is lost | No scene lights; fake the beam with a cone | In-process Python API | 80.5 MB sdist, or 63 MB conda-forge | 24,500 steps/s, 53 MB RSS (measured) |
| ROS 2 Jazzy + RViz2 | Yes | No; needs a `package://` or `file://` path | Marker cone with RGBA | ROS 2 topics | Docker `jazzy-desktop` 1.26 GB compressed | 0.5 core, 300–500 MB (estimate) |
| Gazebo Harmonic + Jazzy | Needs OpenGL above 3.3 | No; needs a world joint, frame preservation and a control plugin | SDF spot light, `light_config` service | ROS 2 topics or gz-transport | `jazzy-desktop-full` 1.42 GB compressed | 1–2 cores, 1–2 GB (estimate) |
| Foxglove (SDK 0.28.0) | Yes | STL format yes; relative mesh path untested | Scene primitive cone | SDK WebSocket server, no ROS needed | 2.2 MB wheel plus the app | 300–600 MB for the app (estimate) |
| Webots R2025a | Officially discouraged on Intel graphics | No; needs the urdf2webots converter | LED device driving a SpotLight | Python controller | 163 MB .deb plus 693 MB offline assets | 2 GB RAM minimum per docs |
| Genesis 1.4.3 | Yes, through the `gs.cpu` backend | Yes, through `gs.morphs.URDF` (not tested) | Fixed at scene build (not verified) | In-process Python API | About 750 MB download with CPU torch | Heavy; JIT compile on first run (estimate) |

Sources for the table rows are [viser on PyPI](https://pypi.org/project/viser/) (15 September 2026) with its [scene API](https://viser.studio/main/api/core/scene_api/), [urdf-loaders](https://github.com/gkjohnson/urdf-loaders) (v0.13.1, July 2026), [MuJoCo on PyPI](https://pypi.org/project/mujoco/) (22 September 2026), [PyBullet files](https://pypi.org/project/pybullet/#files) (30 January 2025) and [conda-forge](https://anaconda.org/conda-forge/pybullet), [osrf/ros on Docker Hub](https://hub.docker.com/r/osrf/ros/tags) (10 September 2026), the [Foxglove 3D panel docs](https://docs.foxglove.dev/docs/visualization/panels/3d), the [Webots R2025a release](https://github.com/cyberbotics/webots/releases/tag/R2025a) (4 February 2025, still the latest stable), and [Genesis install docs](https://genesis-world.readthedocs.io/en/latest/user_guide/overview/installation.html).

## Notes on each option

**Browser viewers.** viser's `ViserUrdf.update_cfg` takes a joint vector and pushes it to every connected browser over its own WebSocket, so the character process needs no separate bridge. The server alone handled 276 updates per second with no browser attached (measured), which is far above a 30–60 Hz animation rate. urdf-loader is lighter, but you write the bridge yourself. Under the Wi-Fi constraint, vendor three.js locally rather than loading it from a CDN.

**MuJoCo.** The passive viewer (`mujoco.viewer.launch_passive`) runs alongside your own loop ([Python docs](https://mujoco.readthedocs.io/en/stable/python.html#passive-viewer)). A `body/light` of type `spot` moves with its body and takes `cutoff` and `diffuse` colour ([light reference](https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-light)). Adding one to `light_emitter_link` and changing `model.light_diffuse` at runtime both worked (measured). The URDF imports no actuators, so you add five position actuators through `MjSpec` if you want physics.

**PyBullet.** It has no Python 3.12 wheel, and Ubuntu 24.04 ships 3.12, so `pip install pybullet` compiles the 80.5 MB source package. That takes several minutes and needs `build-essential` and `python3-dev` (estimate). The upstream bullet3 repository's last tagged release is 3.25 from April 2022.

**ROS 2 Jazzy.** Jazzy is the supported LTS for 24.04, with end of life in May 2029, and Gazebo Harmonic is its recommended pairing ([Gazebo docs](https://gazebosim.org/docs/latest/ros_installation/)). Gazebo Jetty is marked incompatible with Jazzy. Gazebo's Ogre 2 renderer needs OpenGL "higher than 3.3, preferably 4.3+" ([troubleshooting](https://gazebosim.org/docs/latest/troubleshooting/)). Joint control needs a `<ros2_control>` block and a plugin ([gz_ros2_control](https://control.ros.org/jazzy/doc/gz_ros2_control/doc/index.html)), which this URDF deliberately lacks. Lights change at runtime through `/world/<name>/light_config` ([light config](https://gazebosim.org/api/sim/10/light_config.html)). Setting up Gazebo in a few hours is achievable but eats a large share of the timebox (estimate).

**Foxglove.** The 3D panel loads a URDF from a URL, a file or a topic. It drives the URDF from transforms or from a joint-states topic in which "each joint states message must list every joint". The desktop app asks for sign-in on first launch and then works offline ([Foxglove 2.0 announcement](https://discourse.openrobotics.org/t/foxglove-2-0-integrated-ui-new-pricing-and-open-source-changes/36583), March 2024). [Lichtblick](https://github.com/lichtblick-suite/lichtblick) is the open-source fork under MPL-2.0 (Mozilla Public License).

**Webots.** Its LED device can drive a child SpotLight's colour ([LED reference](https://cyberbotics.com/doc/reference/led)), which suits a lamp. However, the [system requirements](https://cyberbotics.com/doc/guide/system-requirements) say "we do not recommend ... Intel graphics adapters" and "for Linux systems, we recommend only NVIDIA graphics cards", which describes the target laptop.

**Genesis.** PyTorch is a separate prerequisite. A plain `pip install torch` on Linux pulls the CUDA build, which is a 555 MB wheel plus CUDA packages, so you must use the [CPU index](https://download.pytorch.org/whl/cpu/torch/), whose wheel is 196 MB. The dependency list includes vtk (140 MB), pymeshlab (107 MB) and opencv (71 MB). That is far more than a 5-joint lamp needs.

## Physics or kinematic playback

The demonstration needs believable motion and light, not contact or force. Kinematic playback means the body sets joint angles directly after two checks. First it clamps each target to the URDF's soft limits from `safety_controller` (shoulder −0.65 to 0.95 rad). Then it caps the change per tick at the velocity limit times the tick period (shoulder 0.95 rad/s, or 0.032 rad per tick at 30 Hz), with an optional acceleration cap for smooth starts. This is deterministic, cheap (MuJoCo's kinematics alone ran at 110,000 calls per second, measured), and easy to test. It also earns the "physical reasoning" marks if the note explains the limits.

Full physics adds gravity sag and gain tuning, which the timebox cannot easily absorb. With position actuators at a stiffness of 40 N·m/rad in MuJoCo, the elbow settled 0.04 rad past its target under gravity, and head pitch did not move at all (measured). The cause is that the lamp-head and upper-arm collision cylinders already overlap by about 1 mm at the zero pose, and MuJoCo checks self-collision by default, so the contact pins the head. Disabling contacts let it reach 0.22 rad against a 0.2 rad target. This particular failure is MuJoCo-specific, because PyBullet ignores self-collision unless the `URDF_USE_SELF_COLLISION` flag is passed. Physics is therefore worth adding only after the character works, and MuJoCo is the one option where the same model serves both modes.

## Link latency

An in-process Python API costs nothing in transport. A localhost WebSocket message and an intra-host ROS 2 message of this size both take well under a millisecond (estimate). Perceived latency is dominated instead by the render frame (17–33 ms) and by the joint velocity limits themselves; base yaw at 1.5 rad/s needs about one second to turn 90 degrees.

## Install and setup time

viser installs in a minute or two and shows the lamp in under an hour (estimate). MuJoCo adds about an hour for the binary STL, the compiler block, actuators and the light (estimate). ROS 2 with RViz2 takes 30–60 minutes to install and wire up for someone who already knows ROS (estimate). Gazebo, Webots and Genesis each cost several hours of the six-to-eight-hour budget for little visible gain.
