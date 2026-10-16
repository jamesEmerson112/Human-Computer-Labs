/**
 * Mounts the kinematic lamp in any page as the global LampEmbed, with no server or fetch.
 *
 * APIs:
 * - LampData: the inlined URDF, scene, shade mesh and gesture library
 * - MountOptions: optional callbacks for mount
 * - Clip: the clip chosen for a MELD label (from ./emotions)
 * - GestureSummary: one gesture as gestures() lists it
 * - Lamp: the handle mount returns
 * - mount: draw the lamp in an element and return its handle
 */
import type { Pose, SceneDescription } from '../protocol';
import { Stage } from '../stage';
import { LampDriver } from './driver';
import { Chooser, type Clip } from './emotions';
import { parseLibrary } from './gestures';
import { parseUrdf, type UrdfModel } from './urdf';

export type { Clip } from './emotions';

/**
 * Description: Everything the lamp needs, inlined by the page that embeds it.
 * Fields:
 * - urdf: the URDF file text.
 * - scene: models/derived/scene.json, parsed.
 * - mesh_base64: models/derived/lamp_shade.stl as base64.
 * - gestures: each gestures/*.json, parsed and sorted by file name, archive excluded.
 */
export interface LampData {
  urdf: string;
  scene: SceneDescription;
  mesh_base64: string;
  gestures: unknown[];
}

/**
 * Description: Optional callbacks for mount.
 * Fields:
 * - onActivity: receives each activity change, such as 'Playing Nod', and each display failure as plain text.
 *   After 'Lamp failed to load: ...' it receives nothing more, so that message stays shown.
 */
export interface MountOptions {
  onActivity?(text: string): void;
}

/**
 * Description: One gesture as gestures() lists it.
 * Fields:
 * - name: the gesture name.
 * - label: the gesture label.
 * - group: the emotion family, or null for a utility clip.
 */
export interface GestureSummary {
  name: string;
  label: string;
  group: string | null;
}

/**
 * Description: The handle for a mounted lamp.
 * Methods:
 * - choose: picks a clip for a MELD label without playing it; null when the family has no clip.
 * - react: picks and plays a clip, interrupting from the current pose; null when nothing played.
 * - play: plays a gesture by name.
 * - reset: returns the lamp home.
 * - setPace: sets the playback speed, above 0 and at most 7.5.
 * - setRunning: pauses or resumes the lamp.
 * - dispose: removes the lamp and frees its resources.
 * - gestures: lists the loaded gestures in library order.
 * Constraints: choose and react throw on an unknown MELD label, play on an unknown name and setPace on a
 * bad speed. After dispose, react returns null and play, reset, setPace and setRunning do nothing. After
 * the lamp visuals fail to load, react returns null and play and reset do nothing.
 */
export interface Lamp {
  choose(meldLabel: string): Clip | null;
  react(meldLabel: string): Clip | null;
  play(name: string): void;
  reset(): void;
  setPace(speed: number): void;
  setRunning(running: boolean): void;
  dispose(): void;
  gestures(): GestureSummary[];
}

/**
 * Description: Formats a caught value as message text.
 * Inputs:
 * - error: the caught value.
 * Outputs: its message.
 */
function message(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

/**
 * Description: Decodes base64 text to bytes.
 * Inputs:
 * - text: the base64 text.
 * Outputs: the decoded bytes.
 * Constraints: throws a readable error on invalid base64.
 */
function decodeBase64(text: string): ArrayBuffer {
  let binary: string;
  try {
    binary = atob(text);
  } catch {
    throw new Error('Lamp data mesh_base64 is not valid base64.');
  }
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index++) bytes[index] = binary.charCodeAt(index);
  return bytes.buffer;
}

/**
 * Description: Checks the shape of the inlined lamp data.
 * Inputs:
 * - data: the value passed to mount.
 * Outputs: none.
 * Constraints: throws a readable error naming the first bad field.
 */
function checkData(data: LampData): void {
  if (!data || typeof data !== 'object') throw new Error('Lamp data is missing.');
  if (typeof data.urdf !== 'string' || !data.urdf.includes('<robot')) throw new Error('Lamp data urdf is not URDF text.');
  const scene = data.scene;
  if (!scene || typeof scene !== 'object' || !Array.isArray(scene.visuals)) throw new Error('Lamp data scene has no visuals.');
  if (scene.schema_version !== 1 || scene.coordinate_system !== 'z-up') throw new Error('Lamp data scene has an unsupported schema.');
  if (!scene.limits || typeof scene.limits !== 'object') throw new Error('Lamp data scene has no limits.');
  if (typeof data.mesh_base64 !== 'string' || data.mesh_base64.length === 0) throw new Error('Lamp data mesh_base64 is empty.');
  if (!Array.isArray(data.gestures)) throw new Error('Lamp data gestures is not a list.');
}

/**
 * Description: Checks that the URDF limits equal the limits scene.json was exported with.
 * Inputs:
 * - model: the parsed URDF.
 * - scene: the scene description.
 * Outputs: none.
 * Constraints: throws naming the first joint that is missing or differs by more than 1e-12.
 */
function checkLimits(model: UrdfModel, scene: SceneDescription): void {
  const names = new Set([...Object.keys(model.limits), ...Object.keys(scene.limits)]);
  for (const name of names) {
    const urdf = model.limits[name];
    const exported = Object.hasOwn(scene.limits, name) ? scene.limits[name] : undefined;
    if (!urdf || !exported) throw new Error(`Lamp data joint ${name} is in only one of the URDF and scene.json.`);
    for (const key of ['lower', 'upper', 'velocity'] as const) {
      if (!(Math.abs(urdf[key] - exported[key]) <= 1e-12)) {
        throw new Error(`Lamp data ${name} ${key} limit is ${urdf[key]} in the URDF but ${exported[key]} in scene.json.`);
      }
    }
  }
}

/**
 * Description: Draws the lamp in an element and returns its handle.
 * Inputs:
 * - element: the container, which must have a size.
 * - data: the inlined lamp data.
 * - options: optional callbacks.
 * Outputs: the Lamp handle.
 * Constraints: throws a readable error on bad data or when WebGL is unavailable. A lost WebGL context
 * is reported through onActivity. A visual that fails to load is reported once and stops the lamp
 * reacting. The stage stops drawing while the lamp rests at home and the camera is still, and wakes on
 * any command, camera drag or WebGL restore, so an idle lamp costs no frames.
 */
export function mount(element: HTMLElement, data: LampData, options: MountOptions = {}): Lamp {
  if (!(element instanceof HTMLElement)) throw new Error('Lamp mount needs an HTML element.');
  checkData(data);
  const model = parseUrdf(data.urdf);
  checkLimits(model, data.scene);
  const library = parseLibrary(data.gestures, Object.keys(model.limits));
  const chooser = new Chooser(library.values());
  const mesh = decodeBase64(data.mesh_base64);
  const meshes = new Map<string, ArrayBuffer>();
  for (const visual of data.scene.visuals) if (visual.type === 'mesh' && visual.mesh) meshes.set(visual.mesh, mesh);
  // Set when the visuals fail to load; the lamp then stops reacting and keeps this message shown.
  let loadError: string | null = null;
  const report = (text: string): void => {
    if (loadError !== null && text !== loadError) return;
    try {
      options.onActivity?.(text);
    } catch {
      // A failing page callback must not stop the lamp.
    }
  };
  // The Stage draws its first frame in its constructor, so the driver must exist first.
  const driver = new LampDriver(model, library, { onActivity: report });
  // The page's setRunning sets wanted; asleep is set while the lamp rests at home with a still camera.
  // The driver and stage run only when wanted and awake. The driver's resume rule makes the first frame
  // after a wake advance nothing, so a clip played from sleep never skips its opening.
  let wanted = true;
  let asleep = false;
  let loaded = false;
  let disposed = false;
  let dragging = false;
  let cameraMovedMs = -Infinity;
  let stage: Stage | undefined;
  /**
   * Description: Starts or stops the driver and stage to match wanted and asleep.
   * Inputs: none.
   * Outputs: none.
   */
  const apply = (): void => {
    const running = wanted && !asleep;
    driver.setRunning(running);
    stage?.setRunning(running);
  };
  /**
   * Description: Starts drawing again after an idle stop.
   * Inputs: none.
   * Outputs: none.
   */
  const wake = (): void => {
    if (disposed || !asleep) return;
    asleep = false;
    apply();
  };
  /**
   * Description: Steps the driver for one frame, then stops drawing once nothing would change.
   * Inputs:
   * - now: the frame time in ms.
   * Outputs: the link poses.
   * Constraints: it never stops before the visuals have loaded or failed, while the camera is dragged, or
   * within 300 ms of a camera change, so the OrbitControls damping glide can finish.
   */
  const frame = (now: number): Record<string, Pose> => {
    const poses = driver.frame(now);
    if (loaded && !dragging && now - cameraMovedMs > 300 && driver.idle) {
      asleep = true;
      apply();
    }
    return poses;
  };
  try {
    stage = new Stage(element, () => {}, { benchmark: false, zoom: false, poseSource: frame });
  } catch (error) {
    throw new Error(`WebGL is unavailable: ${message(error)}`);
  }
  const shown = stage;
  const canvas = shown.renderer.domElement;
  const lost = (): void => report('Lamp display lost its WebGL context.');
  const restored = (): void => {
    report('Lamp display restored.');
    wake();
  };
  const dragStart = (): void => {
    dragging = true;
    wake();
  };
  const dragEnd = (): void => {
    dragging = false;
    cameraMovedMs = performance.now();
  };
  const cameraChange = (): void => {
    cameraMovedMs = performance.now();
    wake();
  };
  canvas.addEventListener('webglcontextlost', lost);
  canvas.addEventListener('webglcontextrestored', restored);
  shown.controls.addEventListener('start', dragStart);
  shown.controls.addEventListener('end', dragEnd);
  shown.controls.addEventListener('change', cameraChange);
  shown.load(data.scene, meshes).then(
    () => {
      loaded = true;
      wake();
    },
    (error: unknown) => {
      loadError = `Lamp failed to load: ${message(error)}`;
      report(loadError);
      loaded = true;
    },
  );
  const inactive = (): boolean => disposed || loadError !== null;
  return {
    choose: (meldLabel) => chooser.choose(meldLabel),
    react: (meldLabel) => {
      const clip = chooser.choose(meldLabel);
      if (inactive() || !clip) return null;
      driver.play(clip.name);
      wake();
      return clip;
    },
    play: (name) => {
      if (inactive()) return;
      driver.play(name);
      wake();
    },
    reset: () => {
      if (inactive()) return;
      driver.reset();
      wake();
    },
    setPace: (speed) => {
      if (!disposed) driver.setPace(speed);
    },
    setRunning: (running) => {
      if (disposed) return;
      wanted = running;
      asleep = false;
      apply();
    },
    dispose: () => {
      if (disposed) return;
      disposed = true;
      canvas.removeEventListener('webglcontextlost', lost);
      canvas.removeEventListener('webglcontextrestored', restored);
      shown.controls.removeEventListener('start', dragStart);
      shown.controls.removeEventListener('end', dragEnd);
      shown.controls.removeEventListener('change', cameraChange);
      shown.dispose();
    },
    gestures: () => [...library.values()].map((gesture) => ({ name: gesture.name, label: gesture.label, group: gesture.group || null })),
  };
}
