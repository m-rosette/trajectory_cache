import os
from pathlib import Path
from datetime import datetime
import numpy as np

from pybullet_robokit.pyb_utils import PybUtils
from pybullet_robokit.load_objects import LoadObjects
from pybullet_robokit.load_robot import LoadRobot
from pybullet_robokit.motion_planners import KinematicChainMotionPlanner
from trajectory_cache.sample_approach_points import (
    sample_hemisphere_suface_pts,
    hemisphere_orientations,
)


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
    def __init__(
        self,
        urdf_path: str,
        home_position: np.ndarray,
        ik_tolerance: float = 0.05,
        render: bool = True,
        ee_link_name: str = 'tool0'
    ):
        self.data_dir = get_data_dir()
        self.pyb = PybUtils(renders=render)
        self.object_loader = LoadObjects(self.pyb.con)

        self.robot = LoadRobot(
            self.pyb.con,
            urdf_path,
            [0, 0, 0],
            self.pyb.con.getQuaternionFromEuler([0, 0, 0]),
            home_position,
            collision_objects=self.object_loader.collision_objects,
            ee_link_name=ee_link_name
        )
        self.home_position = home_position
        self.ik_tolerance = ik_tolerance
        self.motion_planner = KinematicChainMotionPlanner(self.robot)

        pos, ori = self.robot.get_link_state(self.robot.end_effector_index)
        self.start_pose = np.concatenate((pos, ori))

    def find_high_manip_ik(
        self,
        points: np.ndarray,
        hemisphere_samples: list[int],
        offset: float,
        radius: float,
        steps: int = 100,
        save: bool = True,
    ) -> np.ndarray:
        """
        Samples IK solutions on a hemisphere and picks those with max manipulability.
        Saves results under `self.data_dir` if requested.
        """
        n_pts = len(points)
        dims = len(self.robot.controllable_joint_idx)

        # Preallocate result arrays
        iks = np.full((n_pts, dims), np.nan)
        positions = np.full((n_pts, 3), np.nan)
        orientations = np.full((n_pts, 4), np.nan)
        manips = np.full((n_pts, 1), np.nan)
        paths = np.full((steps, dims, n_pts), np.nan)

        print_increment = max(1, n_pts // 20)
        for i, target in enumerate(points):
            if i % print_increment == 0:
                print(f"{100 * i / n_pts:.1f}% complete")

            hemi_pts = sample_hemisphere_suface_pts(target, offset, radius, hemisphere_samples)
            hemi_oris = hemisphere_orientations(target, hemi_pts)

            # Initialize best defaults
            best = {
                "ik": np.full(dims, np.nan),
                "pos": np.full(3, np.nan),
                "ori": np.full(4, np.nan),
                "manip": -np.inf,
                "path": np.full((steps, dims), np.nan),
            }

            for pt, ori in zip(hemi_pts, hemi_oris):
                joints = self.robot.inverse_kinematics((pt, ori))
                self.robot.set_joint_configuration(joints)

                ee_pos, _ = self.robot.get_link_state(self.robot.end_effector_index)
                if np.linalg.norm(ee_pos - pt) > self.ik_tolerance:
                    continue

                traj, collision = self.motion_planner.interpolate_joint_trajectory(
                    self.home_position, joints, num_steps=steps
                )
                if collision:
                    continue

                m_val = self.robot.calculate_manipulability(joints)
                if m_val > best["manip"]:
                    best["ik"] = joints
                    best["pos"] = pt
                    best["ori"] = ori
                    best["manip"] = m_val
                    best["path"] = traj

            # Store best results
            iks[i] = best["ik"]
            positions[i] = best["pos"]
            orientations[i] = best["ori"]
            manips[i] = best["manip"]
            paths[:, :, i] = best["path"]

        # Stack & filter
        combined = np.hstack((iks, positions, orientations, manips))
        mask = ~np.isnan(combined).any(axis=1)

        if save:
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
                paths[:, :, mask],
            )

        return mask


if __name__ == "__main__":
    home = [0] * 7

    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_urdf_dir = os.path.join(script_dir, 'urdf')
    default_urdf_file = os.path.join(default_urdf_dir, "example_6dof_manipulator.urdf")
    
    cache = PathCache(
        urdf_path=default_urdf_file,
        home_position=home, 
        render=True,
        ee_link_name='end_effector'
    )

    data = np.loadtxt(cache.data_dir / "demo_voxel_data.csv")
    centers = data[:, :3]
    centers[:, 1] += 0.9

    mask = cache.find_high_manip_ik(
        points=centers,
        hemisphere_samples=[27, 27],
        offset=0.0,
        radius=0.15,
        steps=100,
    )