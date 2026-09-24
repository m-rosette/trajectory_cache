import numpy as np
from pathlib import Path
import os
import itertools
from datetime import datetime
from pybullet_robokit.pyb_utils import PybUtils
from pybullet_robokit.load_objects import LoadObjects
from pybullet_robokit.load_robot import LoadRobot
from pybullet_robokit.motion_planners import KinematicChainMotionPlanner
from trajectory_cache.sample_approach_points import sample_hemisphere_suface_pts, hemisphere_orientations
from scipy.spatial.transform import Rotation as R

# Motion planners available to PathCache.find_high_manip_ik's `motion_planner_type` param.
MOTION_PLANNER_TYPES = ('interpolate', 'two_stage_cartesian', 'rrt', 'approach_cartesian')


def get_data_dir(base_name: str = "data") -> Path:
    """
    Returns the `data/` directory one level above this script, creating it if missing.
    """
    data_dir = Path(__file__).resolve().parent.parent / base_name
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def timestamped_filename(prefix: str, ext: str = "", timestamp_fmt: str = "%Y%m%d_%H%M%S") -> str:
    """
    Builds a filename with current timestamp.
    """
    ts = datetime.now().strftime(timestamp_fmt)
    return f"{prefix}_{ts}{ext}"


class PathCache:
    def __init__(self, robot_urdf_path: str, robot_home_pos, ik_tol=0.05, renders=True, ee_link_name='tool0',
                 robot_base_ori=[0, 0, 0], data_dir=None):
        """ Generate a cache of paths to high scored manipulability configurations

        Args:
            robot_urdf_path (str): filename/path to urdf file of robot
            renders (bool, optional): visualize the robot in the PyBullet GUI. Defaults to True.
            data_dir (str or Path, optional): directory find_high_manip_ik saves output to.
                Defaults to None, meaning get_data_dir()'s package-relative data/ directory.
        """
        self.pyb = PybUtils(renders=renders)
        self.object_loader = LoadObjects(self.pyb.con)

        # self.amiga_id = self.object_loader.load_urdf(
        #     "trajectory_cache/urdf/amiga/amiga.urdf", 
        #     [0, 0, 0], 
        #     [0, 0, 0]
        # )
        # self.object_loader.collision_objects.append(self.amiga_id)
        self.temp_amiga_collision_obj_gen()

        self.robot_home_pos = robot_home_pos
        self.robot = LoadRobot(
            self.pyb.con, 
            robot_urdf_path, 
            [-0.092075, 0.29845, 1.04775], 
            self.pyb.con.getQuaternionFromEuler(robot_base_ori), 
            self.robot_home_pos, 
            collision_objects=self.object_loader.collision_objects,
            ee_link_name=ee_link_name)
        
        start_position, start_orientation = self.robot.get_link_state(self.robot.end_effector_index)
        self.start_pose = np.concatenate((start_position, start_orientation))

        self.ik_tol = ik_tol
        self.motion_planner = KinematicChainMotionPlanner(self.robot)

        # Empirically estimate max reach once so find_high_manip_ik can skip points that are
        # geometrically out of reach before running the (expensive) hemisphere IK search on them.
        self.max_reach = self._estimate_max_reach()

        # Get data directory
        if data_dir is None:
            self.data_dir = get_data_dir()
        else:
            self.data_dir = Path(data_dir)
            self.data_dir.mkdir(parents=True, exist_ok=True)

    def _estimate_max_reach(self, num_random_samples=2000, margin=1.1, corner_dof_cap=10, seed=0):
        """ Empirically estimates the robot's max reach (end-effector distance from its base
        position) by sampling joint configurations and taking the largest end-effector distance
        observed, inflated by `margin`.

        For a serial revolute chain, the reach boundary is usually found at or near a
        combination of joint limit extremes, so all 2**n joint-limit "corners" are sampled
        directly (skipped if there are more than corner_dof_cap joints, to avoid a combinatorial
        blowup), on top of `num_random_samples` uniform-random configurations for coverage
        in between. Since this only needs to run once and doesn't need contact info, it drives
        PyBullet directly (bare resetJointState + getLinkState(computeForwardKinematics=True)),
        bypassing LoadRobot.reset_joint_positions's collision-detection refresh entirely.

        A margin > 1 is used deliberately: underestimating max reach would silently drop
        genuinely-reachable points from the cache, while overestimating only costs a bit of
        wasted search time on borderline points - the two failure modes are not symmetric.

        Returns:
            float: estimated max reach, in meters, inflated by `margin`.
        """
        lower = np.array(self.robot.lower_limits)
        upper = np.array(self.robot.upper_limits)
        n = len(lower)

        configs = []
        if n <= corner_dof_cap:
            corners = np.array(list(itertools.product([0.0, 1.0], repeat=n)))
            configs.append(lower + corners * (upper - lower))

        rng = np.random.default_rng(seed)
        configs.append(rng.uniform(lower, upper, size=(num_random_samples, n)))
        configs = np.vstack(configs)

        base_pos = np.array(self.robot.start_pos)
        max_dist = 0.0
        for config in configs:
            for i, joint_idx in enumerate(self.robot.controllable_joint_idx):
                self.pyb.con.resetJointState(self.robot.robotId, joint_idx, config[i])
            link_state = self.pyb.con.getLinkState(
                self.robot.robotId, self.robot.end_effector_index, computeForwardKinematics=True)
            dist = np.linalg.norm(np.array(link_state[0]) - base_pos)
            if dist > max_dist:
                max_dist = dist

        # Leave the robot at its home position for whatever search follows
        self.robot.reset_joint_positions()

        return max_dist * margin

    def temp_amiga_collision_obj_gen(self):
        slider_collision = self.pyb.con.createCollisionShape(self.pyb.con.GEOM_BOX, halfExtents=[0.6, 0.1, 0.075])
        slider_body_id = self.pyb.con.createMultiBody(baseMass=0,
                                baseCollisionShapeIndex=slider_collision,
                                basePosition=[0.0, 0.29845, 0.96])
        # --- Mast ---
        mast_collision = self.pyb.con.createCollisionShape(
            self.pyb.con.GEOM_BOX,
            halfExtents=[0.035, 0.035, 0.475]
        )
        mast_body_id = self.pyb.con.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=mast_collision,
            basePosition=[0.57, -0.15, 1.03]
        )

        # --- GPS & Oak cameras ---
        gps_oak_collision = self.pyb.con.createCollisionShape(
            self.pyb.con.GEOM_BOX,
            halfExtents=[0.075, 0.10, 0.10]
        )
        gps_oak_body_id = self.pyb.con.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=gps_oak_collision,
            basePosition=[0.57, -0.15, 1.62]
        )

        # --- Amiga Brain ---
        brain_collision = self.pyb.con.createCollisionShape(
            self.pyb.con.GEOM_BOX,
            halfExtents=[0.145, 0.10, 0.0875]
        )
        brain_body_id = self.pyb.con.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=brain_collision,
            basePosition=[0.57, -0.315, 1.25]
        )
        
        amiga_mesh_path = Path(__file__).resolve().parent / "urdf" / "amiga" / "visual" / "frame_on_amiga_v2_simplified.stl"
        amiga_shape = self.pyb.con.createCollisionShape(
            shapeType=self.pyb.con.GEOM_MESH,
            fileName=str(amiga_mesh_path),
            flags=self.pyb.con.GEOM_FORCE_CONCAVE_TRIMESH
        )
        self.amiga_id = self.pyb.con.createMultiBody(
            baseCollisionShapeIndex=amiga_shape,
            baseVisualShapeIndex=-1,
            basePosition=[0, 0, 0]
        )
        # Add all created collision objects to the loader
        self.object_loader.collision_objects.extend([
            self.amiga_id,
            slider_body_id,
            mast_body_id,
            gps_oak_body_id,
            brain_body_id,
        ])

    def show_voxels_debug_points(self, points, rgb=(1, 0.2, 0), size=4, lifetime=0.0):
        """
        Draw voxel centers as debug points.

        Args:
            points (np.ndarray): (N,3) array of voxel center coordinates in world frame.
            rgb (tuple): RGB color in [0,1].
            size (int): Pixel size of points in GUI.
            lifetime (float): Seconds to persist; 0 -> persistent.
        """
        pts = np.asarray(points, dtype=float)
        colors = np.tile(np.asarray(rgb, dtype=float), (pts.shape[0], 1))
        # One call draws them all
        vid = self.pyb.con.addUserDebugPoints(
            pointPositions=pts.tolist(),
            pointColorsRGB=colors.tolist(),
            pointSize=size,
            lifeTime=lifetime,
        )

        axis_length = 0.2
        self.pyb.con.addUserDebugLine([0, 0, 0], [axis_length, 0, 0], [1, 0, 0], 3)  # X-axis (red)
        self.pyb.con.addUserDebugLine([0, 0, 0], [0, axis_length, 0], [0, 1, 0], 3)  # Y-axis (green)
        self.pyb.con.addUserDebugLine([0, 0, 0], [0, 0, axis_length], [0, 0, 1], 3)  # Z-axis (blue)
        while True:
            self.pyb.con.stepSimulation()

    def _plan_trajectory(self, start_config, end_config, num_steps, collision_objects, motion_planner_type,
                         goal_pose=None):
        """ Dispatches to the selected motion planner and normalizes its result so that,
        regardless of planner, a successful plan is always a (num_steps, n_joints) array.

        Args:
            start_config (array-like): starting joint configuration
            end_config (array-like): target joint configuration. Ignored by 'approach_cartesian',
                which plans to `goal_pose` and ends wherever its continuous path reaches that pose.
            num_steps (int): required number of joint configurations in the returned path
            collision_objects (list): body IDs to check the path against
            motion_planner_type (str): one of MOTION_PLANNER_TYPES
            goal_pose (tuple, optional): (position, quaternion) goal, required by 'approach_cartesian'

        Returns:
            path (np.ndarray or None): (num_steps, n_joints) joint trajectory, or None if
                planning failed or the path is in collision.
            collision_in_path (bool): True if planning failed or the path is in collision.
        """
        if motion_planner_type == 'interpolate':
            path, collision_in_path = self.motion_planner.interpolate_joint_trajectory(
                start_config, end_config, num_steps=num_steps, collision_objects=collision_objects)

        elif motion_planner_type == 'two_stage_cartesian':
            path, collision_in_path = self.motion_planner.two_stage_cartesian_path_avoid_collisions(
                start_config, end_config, num_steps=num_steps, collision_objects=collision_objects)

        elif motion_planner_type == 'approach_cartesian':
            path, info = self.motion_planner.approach_cartesian_path(
                start_config, goal_pose, num_steps=num_steps, collision_objects=collision_objects)
            collision_in_path = path is None

        elif motion_planner_type == 'rrt':
            path = self.motion_planner.rrt_path(
                start_config, end_config, collision_objects=collision_objects, steps=num_steps)
            if path is None:
                return None, True
            path = np.asarray(path)
            if path.shape[0] != num_steps:
                # rrt_path's "start already close to goal" early exit returns an unresampled
                # 2-waypoint path, bypassing its own `steps` resampling - normalize here.
                path = self.motion_planner.sample_path_to_length(path, num_steps)
            collision_in_path = False

        else:
            raise ValueError(f"Unknown motion_planner_type: {motion_planner_type!r} (expected one of {MOTION_PLANNER_TYPES})")

        if collision_in_path:
            return None, True
        return np.asarray(path), False

    def find_high_manip_ik(self, points, num_hemisphere_points, look_at_point_offset, hemisphere_radius,
                            num_configs_in_path=100, motion_planner_type='interpolate', save_data=True,
                            filename_tag="", verbose=True):
        """ Find the inverse kinematic solutions that result in the highest manipulability

        Args:
            points (float list): target end-effector points
            num_hemisphere_points (int list): number of points along each dimension [num_theta, num_pi]
            look_at_point_offset (float): distance to offset the sampled hemisphere from the target point
            hemisphere_radius (float): radius of generated hemisphere
            num_configs_in_path (int, optional): number of joint configurations within path. Defaults to 100.
            motion_planner_type (str, optional): which motion planner to use to connect the robot's
                home position to each candidate configuration. One of MOTION_PLANNER_TYPES:
                'interpolate' (straight joint-space interpolation), 'two_stage_cartesian' (world
                X/Z-then-Y end-effector sweep), 'rrt' (RRT-Connect with shortcut smoothing), or
                'approach_cartesian' (retract / traverse / straight approach along the tool axis,
                tracked continuously; the stored IK is the path's endpoint, not the hemisphere
                IK solution, since that may sit on a different IK branch). Every option returns exactly num_configs_in_path joint configurations when
                successful. Defaults to 'interpolate'.
            save_data (bool, optional): whether to save the resultant data to `self.data_dir`. Defaults to True.
            filename_tag (str, optional): extra tag inserted into saved filenames (e.g. a worker/chunk
                id) so concurrent callers writing to the same data_dir don't collide on the same
                second-resolution timestamp. Defaults to "" (no tag, original filenames).
            verbose (bool, optional): print per-point progress and the reachability pre-filter
                summary. Defaults to True; set False when running under a parallel harness that
                reports its own aggregate progress instead. Defaults to True.

        Returns:
            list[Path] | None: the three saved file paths (csv, npy, csv) if save_data=True,
                else None.
        """
        if motion_planner_type not in MOTION_PLANNER_TYPES:
            raise ValueError(f"Unknown motion_planner_type: {motion_planner_type!r} (expected one of {MOTION_PLANNER_TYPES})")

        points = np.asarray(points)
        num_points = len(points)

        # Initialize arrays for saving data
        best_iks = np.zeros((num_points, len(self.robot.controllable_joint_idx)))
        best_ee_positions = np.zeros((num_points, 3))
        best_orienations = np.zeros((num_points, 4))
        best_manipulabilities = np.zeros((num_points, 1))
        best_paths = np.zeros((num_configs_in_path, len(self.robot.controllable_joint_idx), num_points))

        # Reachability pre-filter: a hemisphere sample can land up to hemisphere_radius closer to
        # the base than the voxel itself (and the hemisphere is centered look_at_point_offset away
        # from it), so a point only has any chance of a valid IK solution if it's within
        # max_reach + that slack. Skipping the full hemisphere search for points that fail this is
        # the cheapest possible rejection - one norm per point instead of up to
        # num_hemisphere_points IK solves each.
        robot_base_pos = np.array(self.robot.start_pos)
        reach_slack = hemisphere_radius + abs(look_at_point_offset)
        in_reach = np.linalg.norm(points - robot_base_pos, axis=1) <= (self.max_reach + reach_slack)
        if verbose:
            print(f"Reachability pre-filter: {in_reach.sum()}/{num_points} points within reach "
                  f"({num_points - in_reach.sum()} skipped, max_reach={self.max_reach:.3f}m)")

        # increment*num_points can floor to 0 for num_points < 20, which would divide by zero below
        print_every = max(1, int(0.05 * num_points))  # ~5% print increment

        for i, pt in enumerate(points):
            if verbose and i % print_every == 0:
                print(f"{np.round(i / num_points, 2) * 100}% Complete")

            if not in_reach[i]:
                best_iks[i, :] = np.nan
                best_ee_positions[i, :] = np.nan
                best_orienations[i, :] = np.nan
                best_manipulabilities[i, :] = np.nan
                best_paths[:, :, i] = np.nan
                continue

            # Sample target points
            hemisphere_pts = sample_hemisphere_suface_pts(pt, look_at_point_offset, hemisphere_radius, num_hemisphere_points)
            hemisphere_oris = hemisphere_orientations(pt, hemisphere_pts)

            best_ik = None
            best_ee_pos = None
            best_orienation = None
            best_manipulability = 0
            best_path = None

            # Collect every hemisphere sample with a valid, collision-free IK solution
            candidates = []
            for target_position, target_orientation in zip(hemisphere_pts, hemisphere_oris):
                # inverse_kinematics already checks self- and environment-collision internally
                # (retrying with a perturbed rest config on failure) and leaves the robot reset
                # to the returned joint_angles, so no separate reset/collision check is needed here.
                joint_angles, collision_free = self.robot.inverse_kinematics(
                    (target_position, target_orientation),
                    collision_objects=self.object_loader.collision_objects,
                    return_status=True,
                )
                # 'approach_cartesian' checks reachability and collision along its own continuous
                # path, and the global IK solution here may sit on a different IK branch than that
                # path reaches - so a failed global solve doesn't rule the pose out, it's only used
                # to rank candidates.
                if motion_planner_type != 'approach_cartesian':
                    if not collision_free:
                        # print('Collision at target config')
                        continue

                    ee_pos, _ = self.robot.get_link_state(self.robot.end_effector_index)

                    # If the distance between the desired point and found ik solution ee-point is greater than the tol, then skip the iteration
                    distance = np.linalg.norm(ee_pos - target_position)
                    if distance > self.ik_tol:
                        # print('IK solution not within IK tol')
                        continue

                manipulability = self.robot.calculate_manipulability(joint_angles)
                candidates.append((manipulability, joint_angles, target_position, target_orientation))

            # Plan paths in descending manipulability order and keep the first that succeeds.
            # Path planning dominates the cost, so this plans as few candidates as possible.
            candidates.sort(key=lambda c: c[0], reverse=True)
            for manipulability, joint_angles, target_position, target_orientation in candidates:
                # Plan a joint trajectory between the robot home position and the desired target
                # configuration using the selected motion planner; always num_configs_in_path long.
                path, collision_in_path = self._plan_trajectory(
                    self.robot_home_pos, joint_angles, num_configs_in_path,
                    self.object_loader.collision_objects, motion_planner_type,
                    goal_pose=(target_position, target_orientation),
                )
                if collision_in_path:
                    continue

                if motion_planner_type == 'approach_cartesian':
                    # The continuous path defines the goal configuration
                    joint_angles = path[-1]
                    manipulability = self.robot.calculate_manipulability(joint_angles)

                best_path = path
                best_ik = joint_angles
                best_ee_pos = target_position
                best_orienation = target_orientation
                best_manipulability = manipulability
                break

            if best_ik is None:
                # No hemisphere sample for this point produced a collision-free IK solution with
                # a collision-free path - mark the row NaN so the mask below drops it, instead of
                # crashing on `best_iks[i, :] = None`.
                best_iks[i, :] = np.nan
                best_ee_positions[i, :] = np.nan
                best_orienations[i, :] = np.nan
                best_manipulabilities[i, :] = np.nan
                best_paths[:, :, i] = np.nan
            else:
                best_iks[i, :] = best_ik
                best_ee_positions[i, :] = best_ee_pos
                best_orienations[i, :] = best_orienation
                best_manipulabilities[i, :] = best_manipulability
                best_paths[:, :, i] = best_path

        # Stack & filter
        combined = np.hstack((best_iks, best_ee_positions, best_orienations, best_manipulabilities))
        mask = ~np.isnan(combined).any(axis=1) 

        if save_data:
            # filename_tag disambiguates concurrent callers (e.g. parallel workers) writing to the
            # same data_dir within the same second-resolution timestamp.
            tag_suffix = f"_{filename_tag}" if filename_tag else ""

            # End-effector CSV
            csv_path = self.data_dir / timestamped_filename(f"voxel_ik_data{tag_suffix}", ".csv")
            np.savetxt(
                csv_path,
                combined[mask],
                delimiter=",",
                header="j1,j2,j3,j4,j5,j6,x,y,z,ox,oy,oz,ow,manip",
                comments="",
            )
            # Paths NPY
            paths_path = self.data_dir / timestamped_filename(f"reachable_paths{tag_suffix}", ".npy")
            np.save(paths_path, best_paths[:, :, mask])

            voxels_path = self.data_dir / timestamped_filename(f"reachable_voxels{tag_suffix}", ".csv")
            np.savetxt(voxels_path, points[mask])

            return [csv_path, paths_path, voxels_path]

        return None

            
if __name__ == "__main__":
    z_base_rotation = np.pi/4  # Rotate base of robot by 45 degrees

    robot_home_pos = [z_base_rotation, -np.pi/2, 2*np.pi/3, 5*np.pi/6, -np.pi/2, 0]

    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_urdf_dir = os.path.join(script_dir, 'urdf', 'ur5e')
    default_urdf_file = os.path.join(default_urdf_dir, "ur5e.urdf")

    path_cache = PathCache(
        robot_urdf_path=default_urdf_file,
        renders=False, 
        robot_home_pos=robot_home_pos,
        ee_link_name='gripper_link',
        robot_base_ori=[0, 0, z_base_rotation]
        )

    # Get presaved target points
    voxel_data_filename = '/home/marcus/imml/trajectory_cache/data/voxel_data_parallelepiped.csv'
    voxel_data = np.loadtxt(os.path.join(path_cache.data_dir, voxel_data_filename))
    voxel_centers = voxel_data[:, :3]

    # Translate voxels in front of robot (compact version)
    translation = np.array([-0.092075, 1.0, 0.5])
    voxel_centers_shifted = voxel_centers + translation

    # # visualize the voxels in PyBullet alongside the robot
    # path_cache.show_voxels_debug_points(
    #     voxel_centers_shifted,
    #     rgb=(1.0, 0.4, 0.0),  # orange
    #     size=4,
    #     lifetime=0.0          # 0 = persist until removed/reset
    # )
    
    # Find highest manipulable poses
    # motion_planner_type selects how each path to a candidate configuration is planned;
    # see MOTION_PLANNER_TYPES ('interpolate', 'two_stage_cartesian', 'rrt', 'approach_cartesian').
    saved_paths = path_cache.find_high_manip_ik(points=voxel_centers_shifted,
                                             num_hemisphere_points=[16, 16],
                                             look_at_point_offset=0.0,
                                             hemisphere_radius=0.10,
                                             num_configs_in_path=100,
                                             motion_planner_type='interpolate')