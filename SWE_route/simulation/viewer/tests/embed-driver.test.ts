/**
 * Checks LampDriver on a fake clock: the step cap, interruption from the current pose, the idle reset,
 * pace, and pause and resume.
 *
 * APIs: none; node:test cases only.
 */
import assert from 'node:assert/strict';
import test from 'node:test';
import { LampDriver } from '../src/embed/driver.ts';
import { parseGesture, parseLibrary, type Gesture } from '../src/embed/gestures.ts';
import { MotionController } from '../src/embed/motion.ts';
import { parseUrdf } from '../src/embed/urdf.ts';
import { gestureFiles, URDF_TEXT } from './library.ts';

/**
 * Description: The parsed lamp URDF.
 */
const MODEL = parseUrdf(URDF_TEXT);

/**
 * Description: The real library plus two test clips: one that ends away from home and a slow one.
 */
const LIBRARY: Map<string, Gesture> = (() => {
  const joints = Object.keys(MODEL.limits);
  const library = parseLibrary(gestureFiles(), joints);
  const clip = (name: string, steps: unknown[]) => parseGesture({ schema_version: 1, name, label: name, units: 'radians', steps }, joints);
  library.set('away', clip('away', [{ targets: { base_yaw_joint: 0.5 } }]));
  library.set('slow', clip('slow', [{ targets: { base_yaw_joint: 0.2 }, duration_seconds: 2 }]));
  return library;
})();

/**
 * Description: A reference controller playing a library clip from home.
 * Inputs:
 * - name: the clip name.
 * Outputs: the controller.
 */
function reference(name: string): MotionController {
  const motion = new MotionController(MODEL.limits);
  motion.play(LIBRARY.get(name)!);
  return motion;
}

test('a long frame advances at most 0.25 s, and the first frame advances nothing', () => {
  const driver = new LampDriver(MODEL, LIBRARY);
  driver.play('nod');
  driver.frame(1000);
  assert.deepEqual(driver.motion.positions, reference('nod').positions, 'the first frame only sets the clock');
  driver.frame(11000);
  const expected = reference('nod');
  expected.step(0.25);
  assert.deepEqual(driver.motion.positions, expected.positions);
  driver.frame(10000);
  assert.deepEqual(driver.motion.positions, expected.positions, 'a clock running backwards advances nothing');
});

test('frames return a pose for every link', () => {
  const driver = new LampDriver(MODEL, LIBRARY);
  const poses = driver.frame(0);
  assert.deepEqual(Object.keys(poses).sort(), [...MODEL.links].sort());
  assert.ok(poses.lamp_head_link.position[0] < 0, 'the head faces -x at home');
});

test('a new clip interrupts from the current pose without a jump', () => {
  const driver = new LampDriver(MODEL, LIBRARY);
  driver.frame(0);
  driver.play('emotion_sad');
  for (let now = 50; now <= 300; now += 50) driver.frame(now);
  const pose = { ...driver.motion.positions };
  assert.notDeepEqual(pose, reference('emotion_sad').positions, 'the first clip has moved');
  driver.play('emotion_happy');
  assert.deepEqual(driver.motion.positions, pose, 'playing does not move the lamp by itself');
  driver.frame(300);
  assert.deepEqual(driver.motion.positions, pose, 'a zero-length frame does not move the lamp');
  const expected = new MotionController(MODEL.limits);
  expected.synchronize(pose);
  expected.play(LIBRARY.get('emotion_happy')!);
  for (let now = 350; now <= 1000; now += 50) {
    driver.frame(now);
    expected.step(0.05);
    assert.deepEqual(driver.motion.positions, expected.positions);
  }
  assert.equal(driver.activity, 'Playing Happy');
});

test('the lamp returns home after 0.4 s of lamp time at rest away from home, and only then', () => {
  const log: string[] = [];
  const driver = new LampDriver(MODEL, LIBRARY, { onActivity: (text) => log.push(text) });
  let now = 0;
  driver.frame(now);
  driver.play('away');
  while (driver.moving) driver.frame((now += 100));
  assert.equal(driver.motion.positions.base_yaw_joint, 0.5);
  assert.equal(driver.activity, 'Ready');
  for (let index = 0; index < 3; index++) {
    driver.frame((now += 100));
    assert.equal(driver.activity, 'Ready', `still waiting after ${(index + 1) * 0.1} s`);
  }
  driver.frame((now += 100));
  assert.equal(driver.activity, 'Resetting');
  while (driver.moving) driver.frame((now += 100));
  assert.ok(Object.values(driver.motion.positions).every((value) => value === 0));
  for (let index = 0; index < 20; index++) driver.frame((now += 100));
  assert.equal(driver.activity, 'Ready');
  assert.deepEqual(log, ['Playing away', 'Ready', 'Resetting', 'Ready']);
});

test('paused time does not count toward the idle reset, and a new clip restarts the wait', () => {
  const driver = new LampDriver(MODEL, LIBRARY);
  let now = 0;
  driver.frame(now);
  driver.play('away');
  while (driver.moving) driver.frame((now += 100));
  driver.frame((now += 100));
  driver.frame((now += 100));
  driver.setRunning(false);
  driver.frame((now += 60000));
  driver.setRunning(true);
  driver.frame((now += 100));
  driver.frame((now += 100));
  assert.equal(driver.activity, 'Ready', 'the resume frame advanced nothing, so only 0.3 s has passed');
  driver.frame((now += 100));
  assert.equal(driver.activity, 'Resetting');
  driver.play('away');
  while (driver.moving) driver.frame((now += 100));
  for (let index = 0; index < 3; index++) driver.frame((now += 100));
  assert.equal(driver.activity, 'Ready');
});

test('pace changes playback time as the controller does', () => {
  const normal = new LampDriver(MODEL, LIBRARY);
  const fast = new LampDriver(MODEL, LIBRARY);
  fast.setPace(2);
  const expected = reference('slow');
  expected.setSpeed(2);
  normal.frame(0);
  fast.frame(0);
  normal.play('slow');
  fast.play('slow');
  let now = 0;
  while (fast.moving) {
    now += 20;
    normal.frame(now);
    fast.frame(now);
    expected.step(0.02);
    assert.deepEqual(fast.motion.positions, expected.positions);
  }
  assert.ok(now <= 1100, `2x finished the 2 s clip in ${now} ms`);
  assert.equal(normal.moving, true);
  assert.throws(() => fast.setPace(0));
  assert.throws(() => fast.setPace(8));
  fast.setPace(7.5);
  assert.equal(fast.motion.speed, 7.5);
  fast.setPace(3);
  assert.equal(fast.motion.speed, 3);
});

test('pause freezes the lamp and resume never jumps', () => {
  const driver = new LampDriver(MODEL, LIBRARY);
  driver.frame(0);
  driver.play('slow');
  driver.frame(100);
  const pose = { ...driver.motion.positions };
  driver.setRunning(false);
  driver.frame(5000);
  assert.deepEqual(driver.motion.positions, pose);
  driver.setRunning(false);
  driver.setRunning(true);
  driver.frame(9000);
  assert.deepEqual(driver.motion.positions, pose, 'the first frame after resume advances nothing');
  driver.frame(9100);
  const expected = reference('slow');
  expected.step(0.1);
  expected.step(0.1);
  assert.deepEqual(driver.motion.positions, expected.positions);
});

test('an unknown clip throws and leaves the motion alone; replays and resets are always reported', () => {
  const log: string[] = [];
  const driver = new LampDriver(MODEL, LIBRARY, { onActivity: (text) => log.push(text) });
  driver.frame(0);
  driver.play('nod');
  driver.frame(100);
  const pose = { ...driver.motion.positions };
  assert.throws(() => driver.play('missing'), /Unknown gesture: missing/);
  assert.deepEqual(driver.motion.positions, pose);
  assert.equal(driver.activity, 'Playing Nod');
  driver.play('nod');
  driver.reset();
  assert.deepEqual(log, ['Playing Nod', 'Playing Nod', 'Resetting']);
  assert.throws(() => new LampDriver(MODEL, LIBRARY, { maxStepSeconds: 0 }));
});
