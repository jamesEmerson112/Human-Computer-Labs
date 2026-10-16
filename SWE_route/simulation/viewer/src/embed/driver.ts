/**
 * Steps the kinematic lamp by wall-clock frames and turns its joint positions into link poses.
 *
 * APIs:
 * - DriverOptions: timing settings and the activity callback
 * - LampDriver: plays, interrupts, paces, pauses and idles the lamp, one frame at a time
 */
import type { Pose } from '../protocol';
import type { Gesture } from './gestures';
import { forwardKinematics } from './kinematics';
import { clampToLimit, MotionController } from './motion';
import type { UrdfModel } from './urdf';

/**
 * Description: Optional driver settings.
 * Fields:
 * - maxStepSeconds: the most lamp time one frame may advance, 0.25 by default.
 * - idleResetSeconds: the lamp time to wait, idle away from home, before returning home; 0.4 by default.
 * - onActivity: receives the controller's activity text whenever it changes and after every play or reset.
 */
export interface DriverOptions {
  maxStepSeconds?: number;
  idleResetSeconds?: number;
  onActivity?(text: string): void;
}

/**
 * Description: Owns one MotionController and advances it from frame timestamps.
 * Constraints: a frame advances min(elapsed, maxStepSeconds) of wall time, and nothing while paused or on
 * the first frame after a resume. Idle time accrues only on frames that begin with the controller at
 * rest away from home; once it reaches idleResetSeconds the driver calls reset(), so repeated
 * interruptions cannot leave the lamp drifting from home.
 */
export class LampDriver {
  readonly motion: MotionController;
  private lastMs: number | undefined;
  private running = true;
  private idleSeconds = 0;
  private reported: string;
  private home: Record<string, number>;
  private maxStep: number;
  private idleReset: number;

  /**
   * Description: Creates a driver at the home pose.
   * Inputs:
   * - model: the parsed URDF.
   * - gestures: the validated library by name.
   * - options: timing settings and the activity callback.
   * Outputs: the driver.
   * Constraints: throws on invalid limits or timing settings.
   */
  constructor(private model: UrdfModel, private gestures: ReadonlyMap<string, Gesture>, private options: DriverOptions = {}) {
    this.motion = new MotionController(model.limits);
    this.maxStep = options.maxStepSeconds ?? 0.25;
    this.idleReset = options.idleResetSeconds ?? 0.4;
    if (!(this.maxStep > 0) || !(this.idleReset >= 0) || !Number.isFinite(this.maxStep) || !Number.isFinite(this.idleReset)) {
      throw new Error('Driver timing settings must be finite and positive.');
    }
    this.home = Object.fromEntries(Object.entries(model.limits).map(([name, limit]) => [name, clampToLimit(limit, 0)]));
    this.reported = this.motion.activity;
  }

  /**
   * Description: The controller's activity text.
   * Inputs: none.
   * Outputs: such as 'Ready' or 'Playing Nod'.
   */
  get activity(): string {
    return this.motion.activity;
  }

  /**
   * Description: Whether the lamp is moving.
   * Inputs: none.
   * Outputs: true while a segment is under way or queued.
   */
  get moving(): boolean {
    return this.motion.moving;
  }

  /**
   * Description: Whether the lamp is at rest at home, so a frame would draw nothing new.
   * Inputs: none.
   * Outputs: true when nothing is moving or queued and every joint is at its home angle.
   */
  get idle(): boolean {
    return !this.motion.moving && this.atHome();
  }

  /**
   * Description: Plays a gesture from the current pose, interrupting whatever is playing.
   * Inputs:
   * - name: the gesture name.
   * Outputs: none.
   * Constraints: throws on an unknown name, leaving the current motion untouched.
   */
  play(name: string): void {
    const gesture = this.gestures.get(name);
    if (!gesture) throw new Error(`Unknown gesture: ${name}`);
    this.motion.play(gesture);
    this.idleSeconds = 0;
    this.report(true);
  }

  /**
   * Description: Returns the lamp home from its current pose.
   * Inputs: none.
   * Outputs: none.
   */
  reset(): void {
    this.motion.reset();
    this.idleSeconds = 0;
    this.report(true);
  }

  /**
   * Description: Sets the playback pace.
   * Inputs:
   * - speed: above 0 and at most 7.5.
   * Outputs: none.
   * Constraints: throws on any other value.
   */
  setPace(speed: number): void {
    this.motion.setSpeed(speed);
  }

  /**
   * Description: Pauses or resumes the lamp's clock.
   * Inputs:
   * - running: false to pause.
   * Outputs: none.
   * Constraints: the first frame after a resume advances nothing, so a pause never causes a jump.
   */
  setRunning(running: boolean): void {
    if (running && !this.running) this.lastMs = undefined;
    this.running = running;
  }

  /**
   * Description: Advances the lamp to a frame time and returns its link poses.
   * Inputs:
   * - nowMs: the frame timestamp in milliseconds, such as a requestAnimationFrame time.
   * Outputs: the pose of every link in the base link frame.
   * Constraints: see the class constraints for the step cap, pausing and the idle reset.
   */
  frame(nowMs: number): Record<string, Pose> {
    if (this.running && Number.isFinite(nowMs)) {
      const dt = this.lastMs === undefined ? 0 : Math.min(this.maxStep, Math.max(0, (nowMs - this.lastMs) / 1000));
      this.lastMs = nowMs;
      const idleAway = !this.motion.moving && !this.atHome();
      this.motion.step(dt);
      if (idleAway) {
        this.idleSeconds += dt;
        // The tolerance absorbs rounding in the sum of many frame times.
        if (this.idleSeconds >= this.idleReset - 1e-9) {
          this.motion.reset();
          this.idleSeconds = 0;
        }
      } else {
        this.idleSeconds = 0;
      }
      this.report();
    }
    return forwardKinematics(this.model, this.motion.positions);
  }

  /**
   * Description: Whether every joint is at its home angle.
   * Inputs: none.
   * Outputs: true at home, within 1e-9 radians.
   */
  private atHome(): boolean {
    return Object.entries(this.home).every(([name, value]) => Math.abs(this.motion.positions[name] - value) <= 1e-9);
  }

  /**
   * Description: Sends the activity text to the callback when it has changed, or always after a command.
   * Inputs:
   * - always: true after play or reset, so replaying a clip is reported too.
   * Outputs: none.
   */
  private report(always = false): void {
    const activity = this.motion.activity;
    if (!always && activity === this.reported) return;
    this.reported = activity;
    this.options.onActivity?.(activity);
  }
}
