import numpy as np
from pathlib import Path
import os
from datetime import datetime
from pybullet_robokit.pyb_utils import PybUtils
from pybullet_robokit.load_objects import LoadObjects
from pybullet_robokit.load_robot import LoadRobot
from pybullet_robokit.motion_planners import KinematicChainMotionPlanner
from trajectory_cache.sample_approach_points import sample_hemisphere_suface_pts, hemisphere_orientations
from scipy.spatial.transform import Rotation as R


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
    def __init__(self, robot_urdf_path: str, robot_home_pos, ik_tol=0.05, renders=True, ee_link_name='tool0', robot_base_ori=[0, 0, 0]):
        """ Generate a cache of paths to high scored manipulability configurations

        Args:
            robot_urdf_path (str): filename/path to urdf file of robot
            renders (bool, optional): visualize the robot in the PyBullet GUI. Defaults to True.
        """
        self.pyb = PybUtils(renders=renders)
        self.object_loader = LoadObjects(self.pyb.con)
        self.amiga_id = self.object_loader.load_urdf(
            "trajectory_cache/urdf/amiga/amiga.urdf", 
            [0, 0, 0], 
            [0, 0, 0]
        )
        self.object_loader.collision_objects.append(self.amiga_id)

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

        # Get data directory
        self.data_dir = get_data_dir()

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

    def find_high_manip_ik(self, points, num_hemisphere_points, look_at_point_offset, hemisphere_radius, num_configs_in_path=100, save_data=True):
        """ Find the inverse kinematic solutions that result in the highest manipulability

        Args:
            points (float list): target end-effector points
            num_hemisphere_points (int list): number of points along each dimension [num_theta, num_pi]
            look_at_point_offset (float): distance to offset the sampled hemisphere from the target point
            hemisphere_radius (float): radius of generated hemisphere
            num_configs_in_path (int, optional): number of joint configurations within path. Defaults to 100.
            save_data_filename (str, optional): file name/path for saving inverse kinematics data. Defaults to None.
            path_filename (str, optional): file name/path for saving resultant paths. Defaults to None.
        """
        num_points = len(points)

        # Initialize arrays for saving data
        best_iks = np.zeros((num_points, len(self.robot.controllable_joint_idx)))
        best_ee_positions = np.zeros((num_points, 3))
        best_orienations = np.zeros((num_points, 4))
        best_manipulabilities = np.zeros((num_points, 1))
        best_paths = np.zeros((num_configs_in_path, len(self.robot.controllable_joint_idx), num_points))

        nan_mask = None
        increment = 0.05  # 5% print increment

        for i, pt in enumerate(points):
            if i % int(increment * num_points) == 0:
                print(f"{np.round(i / num_points, 2) * 100}% Complete")

            # Sample target points
            hemisphere_pts = sample_hemisphere_suface_pts(pt, look_at_point_offset, hemisphere_radius, num_hemisphere_points)
            hemisphere_oris = hemisphere_orientations(pt, hemisphere_pts)

            best_ik = None
            best_ee_pos = None
            best_orienation = None
            best_manipulability = 0
            best_path = None

            # Get IK solution for each target point on hemisphere and save the one with the highest manipulability 
            for target_position, target_orientation in zip(hemisphere_pts, hemisphere_oris):
                joint_angles = self.robot.inverse_kinematics((target_position, target_orientation))

                self.robot.reset_joint_positions(joint_angles)
                ee_pos, ee_ori = self.robot.get_link_state(self.robot.end_effector_index)
                ee_pose = np.concatenate((ee_pos, ee_ori))

                # If the target joint angles result in a collision with the ground plane, skip the iteration
                ground_collision = self.robot.collision_check(self.robot.robotId, [self.object_loader.planeId])
                if ground_collision:
                    # print('ground collision')
                    continue

                # If the distance between the desired point and found ik solution ee-point is greater than the tol, then skip the iteration
                distance = np.linalg.norm(ee_pos - target_position)
                if distance > self.ik_tol:
                    # print('not within tol')
                    continue

                # Interpolate a joint trajectory between the robot home position and the desired target configuration
                path, collision_in_path = self.motion_planner.interpolate_joint_trajectory(robot_home_pos, joint_angles, num_steps=num_configs_in_path)
                if collision_in_path:
                    # print('collision in path')
                    continue

                manipulability = self.robot.calculate_manipulability(joint_angles)
                if manipulability > best_manipulability:
                    best_path = path
                    best_ik = joint_angles
                    best_ee_pos = target_position
                    best_orienation = target_orientation
                    best_manipulability = manipulability

            best_iks[i, :] = best_ik
            best_ee_positions[i, :] = best_ee_pos
            best_orienations[i, :] = best_orienation
            best_manipulabilities[i, :] = best_manipulability
            best_paths[:, :, i] = best_path
        
        # Stack & filter
        combined = np.hstack((best_iks, best_ee_positions, best_orienations, best_manipulabilities))
        mask = ~np.isnan(combined).any(axis=1) 

        if save_data:
            # End-effector CSV
            csv_path = self.data_dir / timestamped_filename("voxel_ik_data", ".csv")
            np.savetxt(
                csv_path,
                combined[mask],
                delimiter=",",
                header="j1,j2,j3,j4,j5,j6,x,y,z,ox,oy,oz,ow,manip",
                comments="",
            )
            # Paths NPY
            np.save(
                self.data_dir / timestamped_filename("reachable_paths", ".npy"),
                best_paths[:, :, mask],
            )
            np.savetxt(
                self.data_dir / timestamped_filename("reachable_voxels", ".csv"),
                points[mask],
            )

        return nan_mask

            
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
    voxel_data_filename = 'new_target_points.csv'
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
    nan_mask = path_cache.find_high_manip_ik(points=voxel_centers_shifted, 
                                             num_hemisphere_points=[10, 10], 
                                             look_at_point_offset=0.0, 
                                             hemisphere_radius=0.15, 
                                             num_configs_in_path=100)