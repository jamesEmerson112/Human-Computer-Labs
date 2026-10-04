"""Browser controls and rendering for the character lamp."""

import math
from contextlib import nullcontext
from queue import Empty, SimpleQueue
import time

import numpy as np
from trimesh.visual.material import PBRMaterial
from trimesh.visual.texture import TextureVisuals
import viser
from viser.extras import ViserUrdf
from viser.transforms import SO3
import yourdfpy

from .motion import MotionController, ROBOT_PATH, load_limits
from .gestures import load_gestures
from .backends import make_backend


CAMERA_POSITION = (-0.72, -1.15, 0.85)
CAMERA_TARGET = (-0.08, 0.0, 0.38)
JOINT_LABELS = {
    "base_yaw_joint": "Base turn",
    "shoulder_pitch_joint": "Shoulder tilt",
    "elbow_pitch_joint": "Elbow bend",
    "neck_yaw_joint": "Neck turn",
    "head_pitch_joint": "Head tilt",
}
JOINT_HINTS = {
    "base_yaw_joint": "Turns the whole arm around the fixed base.",
    "shoulder_pitch_joint": "Leans the entire arm forward or backward.",
    "elbow_pitch_joint": "Folds the upper arm relative to the lower arm.",
    "neck_yaw_joint": "Swivels the head without turning the lower arm.",
    "head_pitch_joint": "Tilts the shade. Positive angles aim its light downward.",
}
LESSONS = {
    "1. Explore the view": "**Orbit first.** Drag the scene, scroll to zoom, then click **Reset view**. Moving your view does not move the robot.",
    "2. Turn one joint": "Set **Base turn** to **20°**, then try **Neck turn** at **20°**. The base turns the whole arm; the neck turns only the head. Click **Reset pose** afterward.",
    "3. Fold the arm": "From the reset pose, set **Shoulder tilt** to **20°**, then **Elbow bend** to **−30°**. Notice that the elbow moves with the shoulder. Click **Reset pose** afterward.",
    "4. Aim the light": "Click **Light view**, then set **Head tilt** to **25°** from the reset pose. Watch the spot move down the wall. Toggle **Light on**, or enable **Show light direction** to see the red forward axis.",
    "5. Try expression": "Click **Reset pose**, choose **Slow** pace, then compare **Nod**, **Shake head**, and **Curious lean**. Each gesture returns to its starting pose. Try **Stop motion** halfway through.",
}
PLAYBACK_SPEEDS = {"Slow": 0.25, "Normal": 0.5, "Full speed": 1.0}
BACKEND_LABELS = {
    "MuJoCo physics": "mujoco",
    "Direct playback (original)": "kinematic",
    "MuJoCo pose check": "mujoco-kinematic",
    "MuJoCo source contacts": "mujoco-source",
}


class LampViewer:
    def __init__(self, port: int = 8080, backend: str = "mujoco", native_viewer: bool = False):
        self.motion = MotionController(load_limits())
        self.backend = make_backend(backend)
        self.native_viewer = native_viewer
        self.actual_positions = self.backend.positions.copy()
        self.motion.set_speed(PLAYBACK_SPEEDS["Normal"])
        self.gestures = load_gestures(self.motion.limits)
        self.commands: SimpleQueue[tuple[str, object]] = SimpleQueue()
        self.model = yourdfpy.URDF.load(str(ROBOT_PATH), mesh_dir=str(ROBOT_PATH.parent))
        # URDF face colors export without a glTF material. glTF's default metal
        # then looks black without an environment map. Supply a diffuse material
        # using each part's original color; do not edit the source URDF or mesh.
        for mesh in self.model.scene.geometry.values():
            mesh.visual = TextureVisuals(material=PBRMaterial(
                baseColorFactor=mesh.visual.main_color,
                metallicFactor=0.0, roughnessFactor=0.55,
            ))
        if set(self.model.actuated_joint_names) != set(self.motion.limits):
            raise ValueError("URDF joint names do not match the motion controller")
        self.server = viser.ViserServer(host="127.0.0.1", port=port, label="Lamp playground")
        self.server.gui.configure_theme(
            control_layout="fixed", control_width="medium", dark_mode=True,
            show_logo=False, show_share_button=False, brand_color=(233, 179, 78),
        )
        self.server.scene.set_up_direction("+z")
        self.server.initial_camera.position = CAMERA_POSITION
        self.server.initial_camera.look_at = CAMERA_TARGET
        self.server.initial_camera.up_direction = (0.0, 0.0, 1.0)
        self._build_scene()
        self._build_controls()
        self._last_status_time = 0.0
        self._last_slider_targets: dict[str, float] = {}
        self.tick(0.0)

    def _build_scene(self) -> None:
        scene = self.server.scene
        scene.configure_environment_map(hdri=None)
        scene.configure_default_lights(enabled=False)
        scene.add_light_ambient("/room/ambient", intensity=0.65)
        scene.add_light_directional(
            "/room/fill", intensity=1.4, wxyz=SO3.from_rpy_radians(0.5, -0.5, 0.0).wxyz,
        )
        scene.add_box(
            "/room/table", dimensions=(2.6, 2.0, 0.04),
            position=(-0.20, 0.0, -0.02), color=(63, 73, 89),
        )
        scene.add_grid(
            "/room/grid", width=2.6, height=2.0, cell_size=0.10,
            section_size=0.5, position=(-0.20, 0.0, 0.001),
            cell_color=(85, 95, 109), section_color=(110, 122, 139),
        )
        scene.add_box(
            "/room/screen", dimensions=(0.025, 2.0, 1.5),
            position=(-1.30, 0.0, 0.75), color=(140, 147, 160),
        )
        scene.add_frame("/lamp", show_axes=False)
        self.robot = ViserUrdf(self.server, self.model, root_node_name="/lamp")
        self.light = scene.add_light_spot(
            "/lamp_light", color=(255, 226, 163), direction=(1.0, 0.0, 0.0),
            intensity=5.0, angle=math.radians(24), penumbra=0.5,
            distance=4.0, decay=2.0, cast_shadow=False,
        )
        self.emitter_axes = scene.add_frame(
            "/emitter_axes", axes_length=0.14, axes_radius=0.003, visible=False,
        )

    def _build_controls(self) -> None:
        gui = self.server.gui
        gui.add_markdown(
            "# Lamp playground\n"
            "Explore a five-joint robot, one movement at a time.\n\n"
            "**Drag** to orbit · **Scroll** to zoom · **Right-drag** to pan."
        )
        self.status = gui.add_markdown("**Ready**")
        with gui.add_folder("Simulation"):
            self.backend_choice = gui.add_dropdown(
                "Mode", options=tuple(BACKEND_LABELS), disabled=self.native_viewer,
                initial_value=next(label for label, name in BACKEND_LABELS.items() if name == self.backend.name),
            )

            @self.backend_choice.on_update
            def on_backend(event):
                if event.client is not None:
                    self.commands.put(("backend", BACKEND_LABELS[event.target.value]))

            self.physics_status = gui.add_markdown("")
        with gui.add_folder("Learn the controls"):
            lesson = gui.add_dropdown("Practice step", options=tuple(LESSONS))
            instructions = gui.add_markdown(LESSONS[lesson.value])

            @lesson.on_update
            def on_lesson(event):
                instructions.content = LESSONS[event.target.value]

        self.sliders = {}
        with gui.add_folder("Joint targets · degrees"):
            for name, limit in self.motion.limits.items():
                # Round inward so the slider endpoints remain within the soft limits.
                slider = gui.add_slider(
                    JOINT_LABELS[name], min=math.ceil(math.degrees(limit.lower) * 10) / 10,
                    max=math.floor(math.degrees(limit.upper) * 10) / 10,
                    step=0.1, initial_value=0.0,
                    hint=JOINT_HINTS[name] + " The slider sets a target, not the current angle.",
                )
                self.sliders[name] = slider

                @slider.on_update
                def on_joint(event, joint_name=name):
                    if event.client is not None:
                        self.commands.put(("target", (joint_name, math.radians(event.target.value))))

        with gui.add_folder("Try a movement"):
            pace = gui.add_dropdown(
                "Pace", options=tuple(PLAYBACK_SPEEDS), initial_value="Normal",
                hint="Slow: 25%, Normal: 50%, Full speed: 100% of the allowed playback rate.",
            )

            @pace.on_update
            def on_pace(event):
                if event.client is not None:
                    self.commands.put(("speed", PLAYBACK_SPEEDS[event.target.value]))

            self.gesture_folder = gui.add_folder("Gestures")
            self.gesture_buttons = []
            self._build_gesture_buttons()
            reload_gestures = gui.add_button("Reload gestures")
            self.gesture_notice = gui.add_markdown(f"Loaded {len(self.gestures)} gestures.")

            @reload_gestures.on_click
            def on_reload(event):
                self.commands.put(("reload_gestures", None))

            for label, command in (("Reset pose", "reset"), ("Stop motion", "stop")):
                button = gui.add_button(label)

                @button.on_click
                def on_action(event, action=command):
                    self.commands.put((action, None))

            reset_view = gui.add_button("Reset view")

            @reset_view.on_click
            def on_reset_view(event):
                if event.client is not None:
                    event.client.camera.position = CAMERA_POSITION
                    event.client.camera.look_at = CAMERA_TARGET
                    event.client.camera.up_direction = (0.0, 0.0, 1.0)

            light_view = gui.add_button("Light view")

            @light_view.on_click
            def on_light_view(event):
                if event.client is not None:
                    event.client.camera.position = (-0.95, -1.8, 1.15)
                    event.client.camera.look_at = (-0.20, 0.0, 0.35)
                    event.client.camera.up_direction = (0.0, 0.0, 1.0)

        with gui.add_folder("Lamp light"):
            self.light_enabled = gui.add_checkbox("Light on", initial_value=True)
            self.brightness = gui.add_slider("Brightness", min=0.0, max=10.0, step=0.1, initial_value=5.0)
            self.show_axes = gui.add_checkbox("Show light direction", initial_value=False)
            gui.add_markdown("The red axis points out of the shade. Try tilting the head to move the light across the wall or table.")

        with gui.add_folder("Current pose"):
            self.pose = gui.add_markdown("")
        gui.add_markdown(
            "**Compare movement**\n\n"
            "Physics includes gravity and simulated motors. Direct playback assigns "
            "the requested pose exactly. Model parameters are assumptions for this fictional lamp."
        )

    def _build_gesture_buttons(self) -> None:
        for button in self.gesture_buttons:
            button.remove()
        self.gesture_buttons.clear()
        with self.gesture_folder:
            for gesture in self.gestures.values():
                button = self.server.gui.add_button(gesture.label, hint=gesture.description)
                self.gesture_buttons.append(button)

                @button.on_click
                def on_gesture(event, selected=gesture):
                    self.commands.put(("gesture", selected))

    def reload_gestures(self) -> None:
        try:
            gestures = load_gestures(self.motion.limits)
        except ValueError as error:
            self.gesture_notice.content = f"**Reload failed. Previous gestures kept.**\n\n{error}"
            return
        self.gestures = gestures
        self._build_gesture_buttons()
        self.gesture_notice.content = f"Loaded {len(gestures)} gestures."

    def tick(self, dt: float) -> None:
        # Viser callbacks run in worker threads. Only this loop mutates motion.
        while True:
            try:
                command, payload = self.commands.get_nowait()
            except Empty:
                break
            if self.backend.name in ("mujoco", "mujoco-source") and command in ("target", "gesture", "reset", "stop"):
                self.motion.synchronize(self.actual_positions)
            if command == "target":
                name, value = payload
                self.motion.set_targets({name: value})
            elif command == "gesture":
                self.motion.play(payload)
            elif command == "reload_gestures":
                self.reload_gestures()
            elif command == "speed":
                self.motion.set_speed(payload)
            elif command == "reset":
                self.motion.reset()
            elif command == "stop":
                self.motion.stop()
            elif command == "backend":
                replacement = make_backend(payload)
                replacement.reset(self.actual_positions)
                self.backend = replacement
                self.motion.synchronize(replacement.positions)
                self.actual_positions = replacement.positions.copy()
        desired = self.motion.step(dt)
        self.actual_positions = self.backend.advance(desired, dt)
        positions = self.actual_positions
        with self.server.atomic():
            self.robot.update_cfg(np.array([positions[n] for n in self.model.actuated_joint_names]))
            # FK includes the URDF head's 180-degree yaw; never hand-flip angles.
            transform = self.backend.emitter_transform()
            rotation = SO3.from_matrix(transform[:3, :3]).wxyz
            self.light.position = transform[:3, 3]
            self.light.wxyz = rotation
            self.light.intensity = self.brightness.value if self.light_enabled.value else 0.0
            self.emitter_axes.position = transform[:3, 3]
            self.emitter_axes.wxyz = rotation
            self.emitter_axes.visible = self.show_axes.value
            if self.motion.targets != self._last_slider_targets:
                for name, value in self.motion.targets.items():
                    slider = self.sliders[name]
                    slider.value = min(slider.max, max(slider.min, round(math.degrees(value), 1)))
                self._last_slider_targets = self.motion.targets.copy()
            now = time.monotonic()
            if now - self._last_status_time > 0.1:
                self.status.content = f"**{self.motion.activity}** · {('Light on' if self.light_enabled.value else 'Light off')}"
                self.pose.content = "\n".join(
                    f"- {JOINT_LABELS[n]}: **{math.degrees(v):.1f}°** (requested {math.degrees(desired[n]):.1f}°)" for n, v in positions.items()
                )
                diag = self.backend.diagnostics
                error = max(abs(positions[n] - desired[n]) for n in positions)
                if diag.contacts is None:
                    self.physics_status.content = "**Direct playback** · Contacts and forces are not measured."
                else:
                    policy = "Source collision shapes" if self.backend.name == "mujoco-source" else "Known overlapping arm/head pair excluded"
                    self.physics_status.content = (
                        f"Tracking error: **{math.degrees(error):.2f}°** max\n\n"
                        f"Contacts: **{diag.contacts}** · Penetration: **{diag.penetration_m * 1000:.2f} mm**\n\n"
                        f"{policy}. "
                        + ("Poses assigned directly; no motor dynamics." if self.backend.name == "mujoco-kinematic" else "Stop requests a hold; motors may settle.")
                    )
                self._last_status_time = now


def run(port: int = 8080, backend: str = "mujoco", native_viewer: bool = False) -> None:
    viewer = LampViewer(port, backend, native_viewer)
    native = None
    if native_viewer:
        import mujoco
        import mujoco.viewer
        native = mujoco.viewer.launch_passive(viewer.backend.model, viewer.backend.data)
        with native.lock():
            native.cam.lookat[:] = CAMERA_TARGET
            native.cam.distance = 1.4
            native.cam.azimuth = -120
            native.cam.elevation = -20
            native.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
            native.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
            native.opt.geomgroup[0] = False  # Hide collision proxies; contacts stay visible.
    print(f"\nOpen http://127.0.0.1:{viewer.server.get_port()} in your browser.", flush=True)
    print("Stop the server with Ctrl+C.\n", flush=True)
    previous = time.monotonic()
    try:
        while native is None or native.is_running():
            started = time.monotonic()
            # A stalled render loop slows simulated time instead of jumping a joint.
            with native.lock() if native else nullcontext():
                viewer.tick(min(started - previous, 1.0 / 30.0))
            if native:
                native.sync()
            previous = started
            time.sleep(max(0.0, 1.0 / 30.0 - (time.monotonic() - started)))
    except KeyboardInterrupt:
        print("\nLamp playground stopped.")
    finally:
        if native:
            native.close()
        viewer.server.stop()
