"""Prints yourdfpy's kinematic chain and link transforms for the lamp URDF as JSON lines.

The TypeScript embed's URDF parser and forward kinematics are checked against
this output, so it runs fresh on every test and no saved result can go stale.
Run it with the simulation's locked Python environment, from any directory.

Output, one JSON object per line on stdout:
- Line 1: {"type": "chain", "links": [link names in URDF order],
  "joints": [{"name", "type", "parent", "child", "origin": [16 floats,
  row-major 4x4], "axis": [3 floats]}, ...] in URDF order}.
- Then one line per configuration: {"type": "pose", "positions": {joint:
  radians for all 5 actuated joints}, "links": {link: [16 floats, row-major
  4x4 transform of the link in base_link]}} for every link.
- Last line: {"type": "end", "poses": N}.

Configurations, in order: zero; each actuated joint alone (URDF order) at its
soft lower and then its soft upper limit; all lower; all upper; 20 poses drawn
from random.Random(7), uniform inside the soft limits, joints in URDF order.
Soft limits are lamp_sim.motion.load_limits: the hard limit intersected with
the safety_controller range.

Contents:
- matrix: flattens a 4x4 transform to 16 row-major floats
- chain_record: describes the URDF's links and joints
- configurations: lists the joint positions to dump
- pose_record: describes every link's transform at one configuration
- main: prints all lines
"""

import json
from pathlib import Path
import random
import sys

import numpy as np
import yourdfpy

SIMULATION = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SIMULATION))

from lamp_sim.motion import load_limits  # noqa: E402

URDF_PATH = SIMULATION.parent / "starter_pack" / "robot" / "dummy_lamp_5dof.urdf"
RANDOM_SEED = 7
RANDOM_POSES = 20


def matrix(transform):
    """
    Description: Flattens a 4x4 transform to 16 row-major floats.
    Inputs:
    - transform: a 4x4 array-like.
    Outputs: a list of 16 Python floats.
    Constraints: raises ValueError when the shape is not 4x4.
    """
    array = np.asarray(transform, dtype=float)
    if array.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 transform, got shape {array.shape}")
    return [float(value) for value in array.reshape(16)]


def chain_record(robot):
    """
    Description: Describes the URDF's links and joints as yourdfpy parsed them.
    Inputs:
    - robot: a yourdfpy.URDF.
    Outputs: the "chain" record, links and joints in URDF order.
    Constraints: a joint without an axis reports yourdfpy's default axis.
    """
    joints = []
    for joint in robot.robot.joints:
        origin = np.eye(4) if joint.origin is None else joint.origin
        joints.append({
            "name": joint.name,
            "type": joint.type,
            "parent": joint.parent,
            "child": joint.child,
            "origin": matrix(origin),
            "axis": [float(value) for value in np.asarray(joint.axis, dtype=float)],
        })
    return {
        "type": "chain",
        "links": [link.name for link in robot.robot.links],
        "joints": joints,
    }


def configurations(names, limits):
    """
    Description: Lists the joint positions to dump, in output order.
    Inputs:
    - names: actuated joint names in URDF order.
    - limits: {joint: JointLimit} soft limits.
    Outputs: a list of {joint: radians} with every actuated joint set.
    """
    zero = {name: 0.0 for name in names}
    poses = [zero]
    for name in names:
        for bound in (limits[name].lower, limits[name].upper):
            poses.append(zero | {name: bound})
    poses.append({name: limits[name].lower for name in names})
    poses.append({name: limits[name].upper for name in names})
    rng = random.Random(RANDOM_SEED)
    for _ in range(RANDOM_POSES):
        poses.append({
            name: rng.uniform(limits[name].lower, limits[name].upper)
            for name in names
        })
    return poses


def pose_record(robot, positions):
    """
    Description: Describes every link's transform at one configuration.
    Inputs:
    - robot: a yourdfpy.URDF.
    - positions: {joint: radians} for every actuated joint.
    Outputs: the "pose" record.
    Constraints: updates the robot's configuration in place.
    """
    robot.update_cfg(positions)
    return {
        "type": "pose",
        "positions": positions,
        "links": {
            link.name: matrix(robot.get_transform(link.name, "base_link"))
            for link in robot.robot.links
        },
    }


def main():
    """
    Description: Prints the chain, every pose, and the end marker.
    Inputs: none.
    Outputs: JSON lines on stdout.
    Constraints: raises ValueError when yourdfpy's actuated joints differ from
    the joints load_limits returns.
    """
    robot = yourdfpy.URDF.load(str(URDF_PATH), load_meshes=False)
    limits = load_limits(URDF_PATH)
    names = list(robot.actuated_joint_names)
    if sorted(names) != sorted(limits):
        raise ValueError(f"Actuated joints {names} differ from limits {sorted(limits)}")
    print(json.dumps(chain_record(robot)))
    poses = configurations(names, limits)
    for positions in poses:
        print(json.dumps(pose_record(robot, positions)))
    print(json.dumps({"type": "end", "poses": len(poses)}))


if __name__ == "__main__":
    main()
