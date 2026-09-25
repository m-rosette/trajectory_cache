""" Visualize the approach poses PathCache.find_high_manip_ik samples for a single target point.

Builds the same scene as parallel_cache.py (UR5e on the Amiga via PathCache) and samples approach
poses with the same functions and settings (sample_hemisphere_suface_pts + hemisphere_orientations),
then draws:
  - the target point (green sphere) and the filter cone the samples must fall in (30 deg about -y
    from the target)
  - every sampled approach pose: a line to the target (green if the global IK solve reaches it
    collision-free within ik_tol, red otherwise) and a small RGB triad of the tool frame

Step through the samples to pose the robot at each one's IK solution:
  - GUI window: Left/Right arrow keys
  - terminal: Enter = next, 'p' = previous, a number = jump to that sample, 'q' = quit

Example:
    python -m trajectory_cache.helper.align_ee_hemisphere --num-points 10 10
    python -m trajectory_cache.helper.align_ee_hemisphere --num-points 20 20 --voxel-index 5000
"""
import argparse
import os
import queue
import sys
import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation as R

from trajectory_cache.path_cache import PathCache
from trajectory_cache.sample_approach_points import sample_hemisphere_suface_pts, hemisphere_orientations

# Match parallel_cache.py's __main__ setup
Z_BASE_ROTATION = np.pi / 4
ROBOT_HOME_POS = [Z_BASE_ROTATION, -np.pi / 2, 2 * np.pi / 3, 5 * np.pi / 6, -np.pi / 2, 0]
VOXEL_TRANSLATION = np.array([-0.092075, 1.0, 0.5])
ANGLE_THRESHOLD = np.pi / 6  # sample_hemisphere_suface_pts' default, passed explicitly so the drawn cone matches

PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_URDF = os.path.join(PACKAGE_DIR, 'urdf', 'ur5e', 'ur5e.urdf')
DEFAULT_VOXEL_FILE = os.path.join(os.path.dirname(PACKAGE_DIR), 'data', 'voxel_data_parallelepiped.csv')


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize sampled approach poses for one target point")
    parser.add_argument('--num-points', type=int, nargs=2, default=[10, 10], metavar=('NUM_THETA', 'NUM_PHI'),
                        help="num_hemisphere_points passed to the sampler (default: 10 10, as in parallel_cache).")
    parser.add_argument('--radius', type=float, default=0.15, help="hemisphere_radius (default: 0.15).")
    parser.add_argument('--offset', type=float, default=0.0, help="look_at_point_offset (default: 0.0).")
    target = parser.add_mutually_exclusive_group()
    target.add_argument('--point', type=float, nargs=3, metavar=('X', 'Y', 'Z'),
                        help="World-frame target point. Defaults to the voxel closest to the voxel cloud's centroid.")
    target.add_argument('--voxel-index', type=int,
                        help="Row of --voxel-file to target (after the parallel_cache translation).")
    parser.add_argument('--voxel-file', default=DEFAULT_VOXEL_FILE,
                        help="Voxel centers CSV used for --voxel-index / the default target.")
    parser.add_argument('--ee-link', default='gripper_link', help="End-effector link (default: gripper_link).")
    return parser.parse_args()


def resolve_target(args):
    if args.point is not None:
        return np.array(args.point, dtype=float)
    voxels = np.loadtxt(args.voxel_file)[:, :3] + VOXEL_TRANSLATION
    if args.voxel_index is not None:
        return voxels[args.voxel_index]
    return voxels[np.argmin(np.linalg.norm(voxels - voxels.mean(axis=0), axis=1))]


def draw_cone(con, apex, radius, half_angle, num_segments=48):
    """ Draws the sampler's filter region: a cone of `half_angle` about -y from `apex`, capped at `radius`. """
    axis, u, v = np.array([0, -1.0, 0]), np.array([1.0, 0, 0]), np.array([0, 0, 1.0])
    alphas = np.linspace(0, 2 * np.pi, num_segments + 1)
    rim = [apex + radius * (np.cos(half_angle) * axis + np.sin(half_angle) * (np.cos(a) * u + np.sin(a) * v))
           for a in alphas]
    for p0, p1 in zip(rim[:-1], rim[1:]):
        con.addUserDebugLine(p0, p1, [1, 0.6, 0], 2)
    for k in range(0, num_segments, num_segments // 8):
        con.addUserDebugLine(apex, rim[k], [1, 0.6, 0], 1)
    # Straight-on (+y) approach axis
    con.addUserDebugLine(apex, apex + radius * axis, [0, 0.4, 1], 2)


def draw_triad(con, position, orientation, length=0.03, width=2):
    rot = R.from_quat(orientation).as_matrix()
    for i, color in enumerate(([1, 0, 0], [0, 1, 0], [0, 0, 1])):
        con.addUserDebugLine(position, position + length * rot[:, i], color, width)


def main():
    args = parse_args()
    target = resolve_target(args)

    path_cache = PathCache(robot_urdf_path=DEFAULT_URDF, robot_home_pos=ROBOT_HOME_POS, renders=True,
                           ee_link_name=args.ee_link, robot_base_ori=[0, 0, Z_BASE_ROTATION])
    con, robot = path_cache.pyb.con, path_cache.robot
    collision_objects = path_cache.object_loader.collision_objects

    pts = sample_hemisphere_suface_pts(target, args.offset, args.radius, args.num_points,
                                       angle_threshold=ANGLE_THRESHOLD)
    oris = hemisphere_orientations(target, pts) if len(pts) else np.zeros((0, 4))
    num_theta, num_phi = args.num_points

    print(f"\nTarget {np.round(target, 3).tolist()}: {len(pts)} of {num_theta * num_phi} grid samples "
          f"survive the filter (num_points={args.num_points}, radius={args.radius})")

    # Draw the target and filter cone
    target_visual = con.createVisualShape(con.GEOM_SPHERE, radius=0.015, rgbaColor=[0, 1, 0, 1])
    con.createMultiBody(baseVisualShapeIndex=target_visual, basePosition=target)
    apex = target - np.array([0, args.offset, 0])
    draw_cone(con, apex, args.radius, ANGLE_THRESHOLD)

    # Solve IK for every sample the same way find_high_manip_ik does, and draw each pose
    samples = []
    print(f"\n{'#':>3}  {'off +y':>7}  {'elev':>6}  {'yaw':>6}  {'IK err':>7}  {'manip':>7}  status")
    for i, (position, orientation) in enumerate(zip(pts, oris)):
        robot.reset_joint_positions(ROBOT_HOME_POS)  # seed from home, as find_high_manip_ik does
        joint_angles, collision_free = robot.inverse_kinematics((position, orientation),
                                                                collision_objects=collision_objects,
                                                                return_status=True)
        ee_pos, _ = robot.get_link_state(robot.end_effector_index)
        ik_err = np.linalg.norm(ee_pos - position)
        ok = collision_free and ik_err <= path_cache.ik_tol
        manip = robot.calculate_manipulability(joint_angles)
        samples.append((position, orientation, joint_angles, ok))

        d = (target - position) / np.linalg.norm(target - position)
        off_y = np.degrees(np.arccos(np.clip(d[1], -1, 1)))
        elev = np.degrees(np.arcsin(d[2]))  # + = approaching upward
        yaw = np.degrees(np.arctan2(d[0], d[1]))  # + = approaching toward +x
        status = 'ok' if ok else ('collision' if not collision_free else 'IK outside tol')
        print(f"{i:>3}  {off_y:>6.1f}°  {elev:>5.1f}°  {yaw:>5.1f}°  {ik_err:>7.4f}  {manip:>7.4f}  {status}")

        con.addUserDebugLine(position, target, [0, 0.8, 0] if ok else [1, 0, 0], 1)
        draw_triad(con, position, orientation)

    if not samples:
        print("\nNo samples to show - try a larger --num-points.")

    # Stepping through samples: GUI arrow keys, or terminal lines read on a background thread
    commands = queue.Queue()

    def read_stdin():
        for line in sys.stdin:
            commands.put(line.strip().lower())
        commands.put('q')

    threading.Thread(target=read_stdin, daemon=True).start()
    prompt = "Enter = next, 'p' = previous, number = jump, 'q' = quit (or Left/Right arrows in the GUI): "

    current = None
    highlight_id = None

    def show(i):
        nonlocal current, highlight_id
        current = i % len(samples)
        position, orientation, joint_angles, ok = samples[current]
        robot.reset_joint_positions(joint_angles)
        if highlight_id is not None:
            con.removeUserDebugItem(highlight_id)
        highlight_id = con.addUserDebugLine(position, target, [1, 1, 0], 5)
        print(f"Showing sample {current} ({'ok' if ok else 'rejected'})")

    if samples:
        show(0)
    print(prompt, end='', flush=True)

    try:
        while con.isConnected():
            quit_requested = False
            while not commands.empty():
                reply = commands.get()
                if reply in ('q', 'quit', 'exit'):
                    quit_requested = True
                    break
                if samples:
                    if reply == '':
                        show(current + 1)
                    elif reply == 'p':
                        show(current - 1)
                    else:
                        try:
                            show(int(reply))
                        except ValueError:
                            print(f"Not a sample index: {reply!r}")
                print(prompt, end='', flush=True)
            if quit_requested:
                break

            keys = con.getKeyboardEvents()
            for key, step in ((con.B3G_RIGHT_ARROW, 1), (con.B3G_LEFT_ARROW, -1)):
                if samples and keys.get(key, 0) & con.KEY_WAS_TRIGGERED:
                    print()
                    show(current + step)
                    print(prompt, end='', flush=True)

            con.stepSimulation()
            time.sleep(1. / 60.)
    except KeyboardInterrupt:
        pass
    print()


if __name__ == '__main__':
    main()
