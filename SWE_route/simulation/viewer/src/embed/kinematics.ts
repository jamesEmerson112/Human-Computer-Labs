/**
 * Computes every link pose of a parsed URDF in the base link frame.
 *
 * APIs:
 * - quaternionFromRpy: URDF roll, pitch and yaw to an xyzw quaternion
 * - jointOrigin: a joint's fixed origin as a pose
 * - poseMatrix: a pose as a row-major 4x4 transform
 * - forwardKinematics: joint positions to link poses
 */
import type { Pose, Quat, Vec3 } from '../protocol';
import type { UrdfJoint, UrdfModel } from './urdf';

/**
 * Description: Converts URDF roll, pitch and yaw to a quaternion, as quaternion(rpy) in
 * tools/export_native_model.py does.
 * Inputs:
 * - rpy: roll, pitch and yaw in radians.
 * Outputs: the xyzw quaternion.
 */
export function quaternionFromRpy(rpy: Vec3): Quat {
  const [roll, pitch, yaw] = rpy.map((angle) => angle / 2);
  const cr = Math.cos(roll), sr = Math.sin(roll);
  const cp = Math.cos(pitch), sp = Math.sin(pitch);
  const cy = Math.cos(yaw), sy = Math.sin(yaw);
  return [sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy,
    cr * cp * sy - sr * sp * cy, cr * cp * cy + sr * sp * sy];
}

/**
 * Description: Multiplies two xyzw quaternions.
 * Inputs:
 * - a, b: the quaternions, applied as a then b in the local frame.
 * Outputs: the product a times b.
 */
function multiply(a: Quat, b: Quat): Quat {
  const [ax, ay, az, aw] = a;
  const [bx, by, bz, bw] = b;
  return [
    aw * bx + ax * bw + ay * bz - az * by,
    aw * by - ax * bz + ay * bw + az * bx,
    aw * bz + ax * by - ay * bx + az * bw,
    aw * bw - ax * bx - ay * by - az * bz,
  ];
}

/**
 * Description: Rotates a vector by a unit quaternion.
 * Inputs:
 * - q: the xyzw unit quaternion.
 * - v: the vector.
 * Outputs: the rotated vector.
 */
function rotate(q: Quat, v: Vec3): Vec3 {
  const [x, y, z, w] = q;
  const tx = 2 * (y * v[2] - z * v[1]);
  const ty = 2 * (z * v[0] - x * v[2]);
  const tz = 2 * (x * v[1] - y * v[0]);
  return [
    v[0] + w * tx + (y * tz - z * ty),
    v[1] + w * ty + (z * tx - x * tz),
    v[2] + w * tz + (x * ty - y * tx),
  ];
}

/**
 * Description: Builds the rotation of a revolute joint about its axis.
 * Inputs:
 * - axis: the joint axis, normalised here.
 * - angle: the joint position in radians.
 * Outputs: the xyzw quaternion.
 * Constraints: throws on a zero-length axis.
 */
function axisAngle(axis: Vec3, angle: number): Quat {
  const length = Math.hypot(...axis);
  if (!(length > 0)) throw new Error('A joint axis has zero length.');
  const s = Math.sin(angle / 2) / length;
  return [axis[0] * s, axis[1] * s, axis[2] * s, Math.cos(angle / 2)];
}

/**
 * Description: A joint's fixed origin, Trans(xyz) times R(rpy), as a pose.
 * Inputs:
 * - joint: the parsed joint.
 * Outputs: the origin pose in the parent link frame.
 */
export function jointOrigin(joint: UrdfJoint): Pose {
  return { position: [...joint.xyz], quaternion: quaternionFromRpy(joint.rpy) };
}

/**
 * Description: Writes a pose as a homogeneous transform.
 * Inputs:
 * - pose: a position and a unit xyzw quaternion.
 * Outputs: the 4x4 transform as 16 numbers in row-major order.
 */
export function poseMatrix(pose: Pose): number[] {
  const [x, y, z, w] = pose.quaternion;
  const [px, py, pz] = pose.position;
  return [
    1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w), px,
    2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w), py,
    2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y), pz,
    0, 0, 0, 1,
  ];
}

/**
 * Description: Computes each link pose as T_parent times Trans(xyz) times R(rpy) times R(axis, q).
 * Inputs:
 * - model: the parsed URDF.
 * - positions: radians for every revolute joint.
 * Outputs: the pose of every link in the root link frame, with xyzw quaternions.
 * Constraints: throws when a revolute joint has no finite position or a link is unreachable.
 */
export function forwardKinematics(model: UrdfModel, positions: Record<string, number>): Record<string, Pose> {
  const children = new Set(model.joints.map((joint) => joint.child));
  const roots = model.links.filter((link) => !children.has(link));
  if (roots.length !== 1) throw new Error('The model must have exactly one root link.');
  const poses: Record<string, Pose> = { [roots[0]]: { position: [0, 0, 0], quaternion: [0, 0, 0, 1] } };
  let pending = model.joints;
  while (pending.length > 0) {
    const waiting = pending.filter((joint) => !poses[joint.parent]);
    if (waiting.length === pending.length) throw new Error('Some links are not connected to the root link.');
    for (const joint of pending) {
      const parent = poses[joint.parent];
      if (!parent) continue;
      let rotation = multiply(parent.quaternion, quaternionFromRpy(joint.rpy));
      if (joint.type === 'revolute') {
        const angle = positions[joint.name];
        if (!Number.isFinite(angle)) throw new Error(`No position for ${joint.name}.`);
        rotation = multiply(rotation, axisAngle(joint.axis, angle));
      }
      const offset = rotate(parent.quaternion, joint.xyz);
      poses[joint.child] = {
        position: [parent.position[0] + offset[0], parent.position[1] + offset[1], parent.position[2] + offset[2]],
        quaternion: rotation,
      };
    }
    pending = waiting;
  }
  return poses;
}
