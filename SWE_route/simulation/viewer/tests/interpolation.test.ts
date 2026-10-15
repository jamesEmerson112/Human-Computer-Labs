import assert from 'node:assert/strict';
import test from 'node:test';
import { PoseBuffer, interpolatePose } from '../src/interpolation.ts';
import type { State } from '../src/protocol.ts';

function state(sequence: number, time: number, x: number): State {
  return { v: 1, type: 'state', sequence, server_time_s: time, simulation_time_s: time,
    requested: {}, actual: {}, activity: 'Idle', speed: 0.5, light: { enabled: true, brightness: 1 },
    links: { lamp: { position: [x, 0, 0], quaternion: [0, 0, 0, 1] } }, diagnostics: {}, timing: {} };
}

test('interpolates world positions and takes the shortest rotation across a quaternion sign flip', () => {
  const pose = interpolatePose({ position: [0, 0, 0], quaternion: [0, 0, 0, 1] }, { position: [2, 4, 6], quaternion: [0, 0, 0, -1] }, 0.5);
  assert.deepEqual(pose.position, [1, 2, 3]);
  assert.equal(Math.abs(pose.quaternion[3]), 1);
});

test('buffers one frame, interpolates by server timestamps, and never extrapolates on a stalled connection', () => {
  const buffer = new PoseBuffer(100);
  buffer.push(state(1, 1, 0), 1000);
  buffer.push(state(2, 1.1, 10), 1100);
  assert.ok(Math.abs(buffer.sample(1150)!.lamp.position[0] - 5) < 1e-10);
  assert.equal(buffer.sample(5000)!.lamp.position[0], 10);
});

test('drops old snapshots and clears prior connection state before accepting a restarted server', () => {
  const buffer = new PoseBuffer();
  buffer.push(state(7, 7, 7), 7000);
  assert.equal(buffer.push(state(6, 6, 6), 7010), false);
  assert.equal(buffer.push(state(7, 7, 8), 7020), false);
  assert.equal(buffer.sample(8000)!.lamp.position[0], 7);
  buffer.clear();
  assert.equal(buffer.sample(8000), undefined);
  assert.equal(buffer.push(state(1, 0, 1), 8010), true);
});
