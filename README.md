# Trajectory Cache

A toolkit for sampling and caching high-manipulability robot arm approach poses and trajectories using PyBullet.

## Description

This package provides a script to search over a hemisphere of candidate approach poses around target points and select configurations that maximize manipulability. It relies on a custom PyBullet robot toolkit (`pybullet_robokit`) to:

- Load a robot URDF and collision objects
- Perform inverse kinematics with collision checks
- Calculate manipulability metrics
- Plan kinematic joint trajectories

The resulting joint configurations and interpolated paths are saved for downstream use in motion planning or trajectory optimization pipelines.

## Installation

1. Clone the repository:
   ```bash
   git clone <your-repo-url>.git
   cd <your-repo-directory>
   ```

2. Initialize and update submodules:
   ```bash
   git submodule update --init --recursive
   ```

3. Install the `pybullet_robokit` toolkit and this package in editable mode:
   ```bash
   pip install -e external/pybullet_robokit -e .
   ```

## Usage

Run the main script to generate a manipulability cache:

```bash
python trajectory_cache/path_cache.py
```

By default, the script will:

1. Load the `ur5e.urdf` robot model from `urdf/ur5e/`.
2. Read pre-saved target points from `data/demo_voxel_data.csv`.
3. Shift them in the y‑axis (to be in-front of the robot) and sample approach points on a hemisphere.
4. Solve inverse kinematics and filter by manipulability, collisions, and IK tolerance.
5. Interpolate joint trajectories and save the best solutions.

You can customize parameters such as:

* `robot_home_pos` (default joint angles)
* Number of hemisphere samples (`num_hemisphere_points`)
* IK tolerance (`ik_tol`)
* Hemisphere radius (`hemisphere_radius`)

## Output

All output files are saved under the `data/` directory at the root of the repo (automatically created if missing). The outputs include:

* **CSV files** (`voxel_ik_data_<timestamp>.csv`):

  * Columns: `j1,j2,...,j6,x,y,z,ox,oy,oz,ow,manip`
  * Contains joint angles, end-effector poses, and manipulability scores for each target point.

* **NumPy archives** (`reachable_paths_<timestamp>.npy`):

  * Shape `(N_steps, N_joints, N_targets)`
  * Stores the interpolated joint trajectories for the best configurations.

## Project Structure

```
├── data/                 # Auto-generated output files
├── external/             # Submodules (e.g., pybullet_robokit)
├── helper/               # Helpful visual debugging scripts
├── trajectory_cache/     # Core package modules
│   ├── path_cache.py     # Main script
│   └── ...               # Other scripts
├── urdf/
│   └── ur5e/
│       └── ur5e.urdf     # Robot description
├── pyroject.toml         # Package setup
└── README.md             # This file
```