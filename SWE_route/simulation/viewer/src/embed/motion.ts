/**
 * Joint-limited kinematic motion, a line-for-line port of lamp_sim/motion.py MotionController.
 *
 * APIs:
 * - JointPositions: radians by joint name
 * - clampToLimit: one angle clamped to a joint limit
 * - MotionController: plays gestures and targets with the Python timing, easing and activity text
 */
import type { Limit } from '../protocol';
import type { Gesture } from './gestures';

/**
 * Description: Radians by joint name.
 */
export type JointPositions = Record<string, number>;

/**
 * Description: Clamps an angle to a joint limit, as JointLimit.clamp in lamp_sim/motion.py does.
 * Inputs:
 * - limit: the joint limit.
 * - value: the angle in radians.
 * Outputs: the angle inside [lower, upper].
 * Constraints: throws on a value that is not finite.
 */
export function clampToLimit(limit: Limit, value: number): number {
  if (!Number.isFinite(value)) throw new Error('Joint targets must be finite radians');
  return Math.min(limit.upper, Math.max(limit.lower, value));
}

/**
 * Description: One compiled move or hold, as _Segment in lamp_sim/motion.py.
 * Fields:
 * - targets: the pose at the end of the segment.
 * - seconds: the authored duration.
 * - hold: true for a hold, whose duration is used as written.
 */
interface Segment {
  targets: JointPositions;
  seconds: number;
  hold: boolean;
}

/**
 * Description: Single-owner kinematic controller. Cubic easing starts and ends each move at rest; its
 * maximum slope of 1.5 is part of each duration, so no joint exceeds its URDF velocity.
 * Fields:
 * - limits: the joint limits, in URDF order.
 * - positions: the current requested pose.
 * - targets: the goal of the current segment.
 * - activity: 'Ready', 'Moving', 'Resetting', 'Stopped' or 'Playing <label>'.
 * Constraints: all angles are radians and all times seconds. It does not simulate forces or contact.
 */
export class MotionController {
  readonly limits: Record<string, Limit>;
  positions: JointPositions;
  targets: JointPositions;
  activity = 'Ready';
  private start: JointPositions;
  private elapsed = 0;
  private duration = 0;
  private paceLimit = Infinity;
  private waypoints: Segment[] = [];
  private currentSpeed = 1;

  /**
   * Description: Starts at zero clamped into every limit, at rest.
   * Inputs:
   * - limits: the joint limits by name.
   * Outputs: the controller.
   * Constraints: throws on an empty or invalid limit set.
   */
  constructor(limits: Record<string, Limit>) {
    if (Object.keys(limits).length === 0) throw new Error('The model has no revolute joints');
    for (const [name, limit] of Object.entries(limits)) {
      if (![limit.lower, limit.upper, limit.velocity].every(Number.isFinite) || limit.lower > limit.upper || limit.velocity <= 0) {
        throw new Error(`Invalid limits for ${name}`);
      }
    }
    this.limits = { ...limits };
    this.positions = Object.fromEntries(Object.entries(limits).map(([name, limit]) => [name, clampToLimit(limit, 0)]));
    this.targets = { ...this.positions };
    this.start = { ...this.positions };
  }

  /**
   * Description: The playback speed.
   * Inputs: none.
   * Outputs: the speed set by setSpeed, 1 by default.
   */
  get speed(): number {
    return this.currentSpeed;
  }

  /**
   * Description: Scales playback time, never above the URDF velocity envelope.
   * Inputs:
   * - speed: the pace, above 0 and at most 7.5.
   * Outputs: none.
   * Constraints: throws on any other value and leaves the speed unchanged. The SWE controller and
   * lamp_sim stop at 3; the page lamp goes to 7.5, and moves still respect each joint's velocity limit.
   */
  setSpeed(speed: number): void {
    if (!Number.isFinite(speed) || !(speed > 0 && speed <= 7.5)) {
      throw new Error('Playback speed must be greater than zero and at most 7.5');
    }
    this.currentSpeed = speed;
  }

  /**
   * Description: Whether a segment is under way or queued.
   * Inputs: none.
   * Outputs: true while moving.
   */
  get moving(): boolean {
    return this.elapsed < this.duration || this.waypoints.length > 0;
  }

  /**
   * Description: Starts a segment from the current pose and computes its duration and pace limit.
   * Inputs:
   * - segment: the segment to start.
   * Outputs: none.
   */
  private begin(segment: Segment): void {
    this.start = { ...this.positions };
    this.targets = { ...segment.targets };
    this.elapsed = 0;
    let minimumDuration = -Infinity;
    for (const [name, limit] of Object.entries(this.limits)) {
      minimumDuration = Math.max(minimumDuration, 1.5 * Math.abs(this.targets[name] - this.positions[name]) / limit.velocity);
    }
    this.duration = segment.hold ? segment.seconds : Math.max(0.25, segment.seconds, minimumDuration);
    this.paceLimit = minimumDuration > 0 ? this.duration / minimumDuration : Infinity;
  }

  /**
   * Description: Replaces any gesture with a move to targets; unspecified joints keep their goals.
   * Inputs:
   * - targets: radians by joint name.
   * Outputs: none.
   * Constraints: throws on an unknown joint or a non-finite angle before changing anything.
   */
  setTargets(targets: JointPositions): void {
    const unknown = Object.keys(targets).filter((name) => !Object.hasOwn(this.limits, name)).sort();
    if (unknown.length > 0) throw new Error(`Unknown joints: ${unknown.join(', ')}`);
    const bounded = { ...this.targets };
    for (const [name, value] of Object.entries(targets)) bounded[name] = clampToLimit(this.limits[name], value);
    this.waypoints = [];
    this.begin({ targets: bounded, seconds: 0.25, hold: false });
    this.activity = 'Moving';
  }

  /**
   * Description: Moves every joint back to zero.
   * Inputs: none.
   * Outputs: none.
   */
  reset(): void {
    this.setTargets(Object.fromEntries(Object.keys(this.limits).map((name) => [name, 0])));
    this.activity = 'Resetting';
  }

  /**
   * Description: Starts from an observed pose, clamped into the limits, and clears queued motion.
   * Inputs:
   * - positions: radians for every joint.
   * Outputs: none.
   * Constraints: throws on a missing joint or a non-finite angle before changing anything.
   */
  synchronize(positions: JointPositions): void {
    const bounded: JointPositions = {};
    for (const [name, limit] of Object.entries(this.limits)) {
      if (!Object.hasOwn(positions, name)) throw new Error(`Missing observed joint: ${name}`);
      bounded[name] = clampToLimit(limit, positions[name]);
    }
    this.positions = bounded;
    this.stop();
  }

  /**
   * Description: Compiles a gesture against the current pose, then plays it, interrupting any motion.
   * Offsets refer to this starting pose, and unspecified joints keep the preceding waypoint's angle.
   * Inputs:
   * - gesture: a validated gesture.
   * Outputs: none.
   * Constraints: compilation finishes before playback changes, so a bad gesture cannot partly play.
   */
  play(gesture: Gesture): void {
    const origin = { ...this.positions };
    let pose = { ...origin };
    const segments: Segment[] = [];
    for (const step of gesture.steps) {
      if (step.kind === 'return_to_start') {
        pose = { ...origin };
      } else if (step.kind === 'targets' || step.kind === 'offsets') {
        for (const [name, angle] of step.angles) {
          if (!Object.hasOwn(this.limits, name)) throw new Error(`Unknown joint: ${name}`);
          const limit = this.limits[name];
          let target = angle;
          if (step.kind === 'offsets') {
            target = origin[name] + angle;
            if (step.onLimit === 'reverse' && !(limit.lower <= target && target <= limit.upper)) target = origin[name] - angle;
          }
          pose[name] = clampToLimit(limit, target);
        }
      } else if (step.kind !== 'hold_seconds') {
        throw new Error(`Unknown gesture step: ${String(step.kind)}`);
      }
      if (!Number.isFinite(step.seconds) || !(step.seconds > 0 && step.seconds <= 60)) {
        throw new Error('Step duration must be finite and between 0 and 60 seconds');
      }
      segments.push({ targets: { ...pose }, seconds: step.seconds, hold: step.kind === 'hold_seconds' });
    }
    if (segments.length === 0) throw new Error('A gesture needs at least one step');
    this.waypoints = segments.slice(1);
    this.begin(segments[0]);
    this.activity = `Playing ${gesture.label}`;
  }

  /**
   * Description: Freezes at the current pose and discards all queued motion.
   * Inputs: none.
   * Outputs: none.
   */
  stop(): void {
    this.waypoints = [];
    this.targets = { ...this.positions };
    this.start = { ...this.positions };
    this.elapsed = this.duration = 0;
    this.activity = 'Stopped';
  }

  /**
   * Description: Advances playback.
   * Inputs:
   * - dt: elapsed wall time in seconds.
   * Outputs: a copy of the new pose.
   * Constraints: throws on a negative or non-finite dt. At or below 1x the timeline scales with speed;
   * above 1x each segment runs at most as fast as its velocity headroom allows, while holds use the full pace.
   */
  step(dt: number): JointPositions {
    if (!Number.isFinite(dt) || dt < 0) throw new Error('dt must be finite and nonnegative');
    dt *= Math.min(this.currentSpeed, 1);
    while (dt > 0 && this.moving) {
      const scale = this.currentSpeed > 1 ? Math.min(this.currentSpeed, this.paceLimit) : 1;
      const remaining = this.duration - this.elapsed;
      const used = Math.min(dt * scale, remaining);
      this.elapsed += used;
      dt -= used / scale;
      const fraction = Math.min(1, this.elapsed / this.duration);
      const eased = fraction * fraction * (3 - 2 * fraction);
      const positions: JointPositions = {};
      for (const [name, value] of Object.entries(this.targets)) positions[name] = this.start[name] + (value - this.start[name]) * eased;
      this.positions = positions;
      if (this.elapsed >= this.duration) {
        this.positions = { ...this.targets };
        const next = this.waypoints.shift();
        if (next) this.begin(next);
        else this.activity = 'Ready';
      }
    }
    return { ...this.positions };
  }
}
