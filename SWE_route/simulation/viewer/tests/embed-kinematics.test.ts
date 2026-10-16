/**
 * Checks the embed's URDF parser and forward kinematics against yourdfpy, through tests/python_kinematics.py.
 *
 * APIs: none; node:test cases only.
 */
import assert from 'node:assert/strict';
import test from 'node:test';
import { forwardKinematics, jointOrigin, poseMatrix, quaternionFromRpy } from '../src/embed/kinematics.ts';
import { parseUrdf } from '../src/embed/urdf.ts';
import { SCENE, URDF_TEXT } from './library.ts';
import { streamJsonLines } from './python-reference.ts';

/**
 * Description: The largest absolute difference between two equal-length number lists.
 * Inputs:
 * - a, b: the lists.
 * Outputs: the largest difference, or Infinity when the lengths differ.
 */
function largestDifference(a: readonly number[], b: readonly number[]): number {
  if (a.length !== b.length) return Infinity;
  return a.reduce((worst, value, index) => Math.max(worst, Math.abs(value - b[index])), 0);
}

test('the chain and every link transform match yourdfpy at zero, every soft limit and 20 random poses', async (t) => {
  const model = parseUrdf(URDF_TEXT);
  let chainChecked = false;
  let poses = 0;
  let ended: number | undefined;
  let chainWorst = 0;
  let poseWorst = 0;
  await streamJsonLines('viewer/tests/python_kinematics.py', (data) => {
    if (data.type === 'chain') {
      assert.deepEqual(model.links, data.links);
      assert.equal(model.joints.length, data.joints.length);
      data.joints.forEach((expected: any, index: number) => {
        const joint = model.joints[index];
        assert.deepEqual([joint.name, joint.type, joint.parent, joint.child], [expected.name, expected.type, expected.parent, expected.child]);
        const origin = largestDifference(poseMatrix(jointOrigin(joint)), expected.origin);
        const axis = joint.type === 'revolute' ? largestDifference(joint.axis, expected.axis) : 0;
        chainWorst = Math.max(chainWorst, origin, axis);
        assert.ok(origin < 1e-12, `${joint.name} origin differs by ${origin}`);
        assert.ok(axis < 1e-12, `${joint.name} axis differs by ${axis}`);
      });
      chainChecked = true;
      return;
    }
    if (data.type === 'pose') {
      assert.ok(chainChecked, 'the chain line comes first');
      const positions = data.positions as Record<string, number>;
      assert.deepEqual(Object.keys(positions).sort(), Object.keys(model.limits).sort());
      for (const [name, value] of Object.entries(positions)) {
        const limit = model.limits[name];
        assert.ok(value >= limit.lower - 1e-12 && value <= limit.upper + 1e-12, `${name} ${value} is outside the soft limits`);
      }
      const links = forwardKinematics(model, positions);
      assert.deepEqual(Object.keys(links).sort(), Object.keys(data.links).sort());
      for (const [link, expected] of Object.entries(data.links as Record<string, number[]>)) {
        const difference = largestDifference(poseMatrix(links[link]), expected);
        poseWorst = Math.max(poseWorst, difference);
        assert.ok(difference < 1e-9, `${link} differs by ${difference} at ${JSON.stringify(positions)}`);
      }
      poses += 1;
      return;
    }
    assert.equal(data.type, 'end');
    ended = data.poses;
  });
  assert.equal(ended, poses);
  assert.equal(poses, 1 + 2 * Object.keys(model.limits).length + 2 + 20);
  t.diagnostic(`checked ${model.joints.length} joints and ${poses} poses of ${model.links.length} links`);
  t.diagnostic(`largest difference: chain ${chainWorst}, link transforms ${poseWorst}`);
});

test('the URDF limits are the hard range intersected with the soft range and equal scene.json', () => {
  const model = parseUrdf(URDF_TEXT);
  assert.deepEqual(Object.keys(model.limits), ['base_yaw_joint', 'shoulder_pitch_joint', 'elbow_pitch_joint', 'neck_yaw_joint', 'head_pitch_joint']);
  assert.deepEqual(model.limits, SCENE.limits);
  assert.equal(model.links.length, 10);
  assert.equal(model.joints.length, 9);
});

test('the head pitch origin carries the 180 degree yaw with no special case', () => {
  const model = parseUrdf(URDF_TEXT);
  const zero = Object.fromEntries(Object.keys(model.limits).map((name) => [name, 0]));
  const head = poseMatrix(forwardKinematics(model, zero).lamp_head_link);
  assert.ok(Math.abs(head[0] + 1) < 1e-12, 'the head x axis points along base -x');
  assert.ok(head[3] < 0, 'the head sits on the -x side');
  assert.deepEqual(quaternionFromRpy([0, 0, 0]), [0, 0, 0, 1]);
});

/**
 * Description: A minimal two-link URDF for parser edge cases.
 * Inputs:
 * - joint: the joint element text.
 * Outputs: the URDF text.
 */
function tiny(joint: string): string {
  return `<robot name="t"><link name="a"/><link name="b"/>${joint}</robot>`;
}

test('the URDF parser strips comments, intersects limits and rejects bad joints', () => {
  const limit = '<limit lower="-1" upper="1" velocity="2"/>';
  const joint = (body: string, type = 'revolute') => `<joint name="j" type="${type}"><parent link="a"/><child link="b"/>${body}</joint>`;
  const commented = parseUrdf(tiny(`<!-- <joint name="ghost" type="prismatic"></joint> -->${joint(limit)}`));
  assert.deepEqual(Object.keys(commented.limits), ['j']);
  assert.deepEqual(commented.joints[0].axis, [1, 0, 0]);
  const soft = parseUrdf(tiny(joint(`${limit}<safety_controller soft_lower_limit="-2" soft_upper_limit="0.5"/>`)));
  assert.deepEqual(soft.limits.j, { lower: -1, upper: 0.5, velocity: 2 });
  assert.throws(() => parseUrdf(tiny(joint(''))), /Missing limits/);
  assert.throws(() => parseUrdf(tiny(joint(limit, 'prismatic'))), /unsupported type/);
  assert.throws(() => parseUrdf(tiny(joint('<limit lower="" upper="1" velocity="2"/>'))), /Invalid/);
  assert.throws(() => parseUrdf(tiny(joint('<limit lower="0x1" upper="1" velocity="2"/>'))), /Invalid/);
  assert.throws(() => parseUrdf(tiny(joint('<limit lower="-1" upper="1" velocity="0"/>'))), /Invalid limits/);
  assert.throws(() => parseUrdf(tiny(joint('<limit lower="1" upper="-1" velocity="2"/>'))), /Invalid limits/);
  assert.throws(() => parseUrdf(tiny(joint(`${limit}<origin xyz="0 0"/>`))), /Invalid/);
  assert.throws(() => parseUrdf(tiny('<joint name="j" type="fixed"/>')), /lacks a name, parent or child/);
  assert.throws(() => parseUrdf(tiny(joint('', 'fixed'))), /no revolute joints/);
});
