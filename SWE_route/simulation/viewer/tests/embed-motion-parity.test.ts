/**
 * Replays controller/tests/python_reference.py against the TypeScript MotionController, as
 * controller/tests/motion_parity.rs does for Rust, and ports that file's other motion checks.
 *
 * APIs: none; node:test cases only.
 */
import assert from 'node:assert/strict';
import test from 'node:test';
import { parseGesture, parseLibrary, type Gesture, type GestureStep } from '../src/embed/gestures.ts';
import { MotionController, type JointPositions } from '../src/embed/motion.ts';
import { parseUrdf } from '../src/embed/urdf.ts';
import type { Limit } from '../src/protocol.ts';
import { gestureFiles, URDF_TEXT } from './library.ts';
import { streamJsonLines } from './python-reference.ts';

/**
 * Description: The same limits motion_parity.rs hard-codes for its unit checks.
 */
const LIMITS: Record<string, Limit> = {
  base_yaw_joint: { lower: -2.45, upper: 2.45, velocity: 1.5 },
  shoulder_pitch_joint: { lower: -0.65, upper: 0.95, velocity: 0.95 },
  elbow_pitch_joint: { lower: -1.7, upper: 0.3, velocity: 1.15 },
  neck_yaw_joint: { lower: -1.25, upper: 1.25, velocity: 1.6 },
  head_pitch_joint: { lower: -0.8, upper: 0.6, velocity: 1.45 },
};

/**
 * Description: A snapshot line as python_reference.py prints it, with arrays in sorted joint order.
 */
interface Snapshot { positions: number[]; targets: number[]; moving: boolean; activity: string }

/**
 * Description: The largest position and target differences seen against Python.
 */
const worst = { position: 0, target: 0 };

/**
 * Description: Compares the controller with a Python snapshot, as check_snapshot in motion_parity.rs does.
 * Inputs:
 * - motion: the TypeScript controller.
 * - names: the joint names in sorted order.
 * - expected: the Python snapshot.
 * - context: a label for failures.
 * Outputs: none.
 * Constraints: positions and targets within 1e-12; moving and activity exactly equal.
 */
function checkSnapshot(motion: MotionController, names: string[], expected: Snapshot, context: string): void {
  names.forEach((name, index) => {
    const position = Math.abs(motion.positions[name] - expected.positions[index]);
    const target = Math.abs(motion.targets[name] - expected.targets[index]);
    worst.position = Math.max(worst.position, position);
    worst.target = Math.max(worst.target, target);
    if (!(position < 1e-12)) assert.fail(`${context}: ${name} position ${motion.positions[name]} != ${expected.positions[index]}`);
    if (!(target < 1e-12)) assert.fail(`${context}: ${name} target ${motion.targets[name]} != ${expected.targets[index]}`);
  });
  if (motion.moving !== expected.moving) assert.fail(`${context}: moving ${motion.moving} != ${expected.moving}`);
  if (motion.activity !== expected.activity) assert.fail(`${context}: activity ${motion.activity} != ${expected.activity}`);
}

/**
 * Description: Builds a one-off gesture for the unit checks, as gesture() in motion_parity.rs does.
 * Inputs:
 * - steps: the raw step list.
 * Outputs: the validated gesture.
 */
function gesture(steps: unknown[]): Gesture {
  return parseGesture({ schema_version: 1, name: 'test', label: 'Test', units: 'radians', steps }, Object.keys(LIMITS));
}

test('every gesture matches live Python traces at three starts and three paces, with interruption and scripted actions', async (t) => {
  let header: { limits: Record<string, Limit>; gesture_names: string[] } | undefined;
  let gestures = new Map<string, Gesture>();
  let names: string[] = [];
  let scripted: MotionController | undefined;
  let cases = 0;
  let frames = 0;
  let actions = 0;
  const seen = new Set<string>();
  await streamJsonLines('controller/tests/python_reference.py', (data) => {
    if (!header) {
      header = data;
      const limits = data.limits as Record<string, Limit>;
      assert.deepEqual(parseUrdf(URDF_TEXT).limits, limits, 'parseUrdf limits must equal Python load_limits');
      names = Object.keys(limits).sort();
      gestures = parseLibrary(gestureFiles(), Object.keys(limits));
      assert.deepEqual([...gestures.keys()].sort(), data.gesture_names, 'the TypeScript and Python libraries must match');
      scripted = new MotionController(limits);
      return;
    }
    const limits = header.limits;
    if (data.action) {
      const action = data.action;
      const motion = scripted!;
      if ('targets' in action) motion.setTargets(action.targets);
      else if ('speed' in action) motion.setSpeed(action.speed);
      else if ('play' in action) motion.play(gestures.get(action.play)!);
      else if ('authored' in action) motion.play(parseGesture(action.authored, Object.keys(limits)));
      else if ('dt' in action) motion.step(action.dt);
      else if ('stop' in action) motion.stop();
      else if ('reset' in action) motion.reset();
      else if ('synchronize' in action) motion.synchronize(action.synchronize);
      else assert.fail(`unknown action ${JSON.stringify(action)}`);
      checkSnapshot(motion, names, data.expected, JSON.stringify(action));
      actions += 1;
      return;
    }
    const key = `${data.name}|${data.start}|${data.speed}`;
    assert.ok(!seen.has(key), `duplicate reference case ${key}`);
    seen.add(key);
    const clip = gestures.get(data.name);
    assert.ok(clip, `no TypeScript gesture ${data.name}`);
    const motion = new MotionController(limits);
    motion.synchronize(data.origin);
    motion.setSpeed(data.speed);
    motion.play(clip);
    (data.frames as Snapshot[]).forEach((frame, index) => {
      const before = { ...motion.positions };
      motion.step(1 / 30);
      const context = `${data.name} ${data.start} ${data.speed}x frame ${index}`;
      checkSnapshot(motion, names, frame, context);
      for (const [name, value] of Object.entries(motion.positions)) {
        const limit = limits[name];
        if (!(value >= limit.lower - 1e-12 && value <= limit.upper + 1e-12)) assert.fail(`${context}: ${name} exceeded limits`);
        if (!(Math.abs(value - before[name]) <= limit.velocity * Math.min(motion.speed, 1) / 30 + 1e-12)) {
          assert.fail(`${context}: ${name} exceeded velocity`);
        }
      }
    });
    assert.equal(motion.moving, false, `${data.name} ${data.start} ${data.speed}x still moving`);
    motion.synchronize(data.origin);
    motion.setSpeed(Math.max(data.speed, 1));
    motion.play(clip);
    motion.step(0.3);
    checkSnapshot(motion, names, data.interrupted, `${key} interrupt`);
    motion.stop();
    motion.step(60);
    checkSnapshot(motion, names, data.stopped, `${key} stopped`);
    frames += data.frames.length;
    cases += 1;
  });
  assert.ok(header, 'python_reference.py printed nothing');
  assert.equal(gestures.size, 126);
  assert.equal(cases, gestures.size * 9);
  for (const name of gestures.keys()) {
    for (const start of ['neutral', 'lower', 'upper']) {
      for (const speed of [0.5, 2, 3]) assert.ok(seen.has(`${name}|${start}|${speed}`), `missing ${name} / ${start} / ${speed}x`);
    }
  }
  assert.ok(actions > 0, 'the scripted actions were replayed');
  t.diagnostic(`matched ${cases} cases, ${frames} frames, ${cases} interruptions and ${actions} scripted actions`);
  t.diagnostic(`largest difference: position ${worst.position}, target ${worst.target}`);
});

test('manual targets clamp and respect joint velocity', () => {
  for (const target of [-100, 100]) {
    const motion = new MotionController(LIMITS);
    motion.setTargets(Object.fromEntries(Object.keys(LIMITS).map((name) => [name, target])));
    for (let index = 0; index < 600; index++) {
      const before = { ...motion.positions };
      motion.step(1 / 120);
      for (const [name, value] of Object.entries(motion.positions)) {
        const limit = motion.limits[name];
        assert.ok(value >= limit.lower && value <= limit.upper);
        assert.ok(Math.abs(value - before[name]) <= limit.velocity / 120 + 1e-12);
      }
    }
    assert.equal(motion.moving, false);
    for (const [name, value] of Object.entries(motion.positions)) {
      assert.equal(value, target < 0 ? motion.limits[name].lower : motion.limits[name].upper);
    }
  }
});

test('invalid commands preserve active motion', () => {
  const motion = new MotionController(LIMITS);
  const clip = gesture([{ offsets: { head_pitch_joint: 0.2 } }, { return_to_start: true }]);
  motion.play(clip);
  motion.step(0.1);
  const baseline = new MotionController(LIMITS);
  baseline.play(clip);
  baseline.step(0.1);
  for (const value of [NaN, Infinity, -Infinity]) {
    assert.throws(() => motion.setTargets({ head_pitch_joint: value }));
    assert.throws(() => motion.setSpeed(value));
    assert.throws(() => motion.step(value));
    assert.throws(() => motion.synchronize({ ...motion.positions, head_pitch_joint: value }));
  }
  assert.throws(() => motion.step(-0.1));
  for (const value of [0, -1, 3.01]) assert.throws(() => motion.setSpeed(value));
  assert.throws(() => motion.setTargets({ unknown: 0.1 }));
  assert.throws(() => motion.synchronize({}));
  const bad = gesture([{ hold_seconds: 1 }]);
  const unknownStep = { kind: 'unknown', angles: [], seconds: 0.1, onLimit: 'clamp' } as unknown as GestureStep;
  assert.throws(() => motion.play({ ...bad, steps: [...bad.steps, unknownStep] }));
  assert.throws(() => motion.play({ ...bad, steps: [...bad.steps, { kind: 'hold_seconds', angles: [], seconds: NaN, onLimit: 'clamp' }] }));
  assert.throws(() => motion.play({ ...bad, steps: [{ kind: 'targets', angles: [['unknown', 0.1]], seconds: 0.25, onLimit: 'clamp' }] }));
  assert.throws(() => motion.play({ ...bad, steps: [] }));
  for (let index = 0; index < 60; index++) {
    assert.deepEqual(motion.step(1 / 30), baseline.step(1 / 30));
    assert.equal(motion.activity, baseline.activity);
  }
});

test('offsets, holds and a large dt keep the Python segment semantics', () => {
  const motion = new MotionController(LIMITS);
  motion.setTargets({ head_pitch_joint: 0.1 });
  motion.step(10);
  motion.play(gesture([
    { offsets: { head_pitch_joint: 0.2 } }, { targets: { base_yaw_joint: 0.4 } },
    { offsets: { head_pitch_joint: -0.2 } },
  ]));
  motion.step(0.25);
  assert.ok(Math.abs(motion.positions.head_pitch_joint - 0.3) < 1e-12);
  motion.step(0.4);
  assert.ok(Math.abs(motion.positions.base_yaw_joint - 0.4) < 1e-12);
  assert.ok(Math.abs(motion.positions.head_pitch_joint - 0.3) < 1e-12);
  motion.step(10);
  assert.ok(Math.abs(motion.positions.head_pitch_joint + 0.1) < 1e-12);
  motion.reset();
  motion.step(10);
  motion.play(gesture([
    { targets: { head_pitch_joint: 0.1 }, duration_seconds: 0.5 },
    { hold_seconds: 0.05 }, { return_to_start: true },
  ]));
  motion.step(0.5);
  motion.step(0.04);
  assert.equal(motion.positions.head_pitch_joint, 0.1);
  motion.step(0.02);
  assert.ok(motion.positions.head_pitch_joint < 0.1);
  motion.step(10);
  assert.equal(motion.moving, false);
  assert.equal(motion.positions.head_pitch_joint, 0);
});

test('fast pace shortens authored motion without jumping or exceeding velocity', () => {
  for (const speed of [2, 3]) {
    const motion = new MotionController(LIMITS);
    const normal = new MotionController(LIMITS);
    const clip = gesture([{ targets: { base_yaw_joint: 0.1 }, duration_seconds: 1 }]);
    motion.play(clip);
    normal.play(clip);
    assert.deepEqual(motion.step(0.2), normal.step(0.2));
    const held: JointPositions = { ...motion.positions };
    motion.setSpeed(speed);
    assert.deepEqual(motion.step(0), held);
    assert.deepEqual(motion.step(0.3 / speed), normal.step(0.3));
    motion.step(0.5 / speed);
    assert.equal(motion.moving, false);
    assert.equal(motion.positions.base_yaw_joint, 0.1);
    assert.equal(normal.moving, true);

    motion.reset();
    normal.reset();
    motion.step(10);
    normal.step(10);
    motion.setTargets({ base_yaw_joint: 1.5 });
    normal.setTargets({ base_yaw_joint: 1.5 });
    for (let index = 0; index < 240; index++) {
      const before = motion.positions.base_yaw_joint;
      assert.deepEqual(motion.step(1 / 120), normal.step(1 / 120));
      assert.ok(Math.abs(motion.positions.base_yaw_joint - before) <= 1.5 / 120 + 1e-12);
    }
    assert.equal(motion.moving, false);
  }
});

test('fast pace preserves wall time across holds and capped moves', () => {
  for (const speed of [2, 3]) {
    const motion = new MotionController(LIMITS);
    motion.play(gesture([
      { targets: { base_yaw_joint: 0.1 }, duration_seconds: 1 },
      { hold_seconds: 0.4 },
      { targets: { base_yaw_joint: 1.3 }, duration_seconds: 0.25 },
    ]));
    motion.setSpeed(speed);
    // The move and hold accelerate; the final capped move advances 0.3 seconds.
    motion.step(1.4 / speed + 0.3);
    assert.ok(Math.abs(motion.positions.base_yaw_joint - 0.2875) < 1e-12);
    motion.step(0.91);
    assert.equal(motion.moving, false);
    assert.equal(motion.positions.base_yaw_joint, 1.3);
  }
});
