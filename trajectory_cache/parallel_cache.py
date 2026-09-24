import argparse
import atexit
import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from trajectory_cache.path_cache import PathCache, get_data_dir, timestamped_filename, MOTION_PLANNER_TYPES


def build_chunks(points, chunk_size, seed=0):
    """ Shuffles point order, then splits into chunks of chunk_size (last chunk may be smaller).

    Shuffling matters here, not just cosmetic: points are typically generated in scan order
    (see voxel_gen.py), so a naive contiguous split would likely hand one worker an entire
    spatially-clustered region (e.g. everything behind an obstacle). RRT's cost in particular is
    heavy-tailed - a handful of expensive searches dominate total runtime - so if those cluster
    into one chunk, that chunk becomes the bottleneck regardless of core count. Shuffling first
    spreads both spatial clustering and cost variance evenly across chunks.

    Returns:
        list[np.ndarray]: point chunks, each of shape (<=chunk_size, 3)
    """
    points = np.asarray(points)
    rng = np.random.default_rng(seed)
    shuffled = points[rng.permutation(len(points))]
    return [shuffled[i:i + chunk_size] for i in range(0, len(shuffled), chunk_size)]


def select_center_points(points, fraction):
    """ Keeps the `fraction` of points closest to the point cloud's centroid (a roughly ball-shaped
    core of the voxel grid), e.g. for a quick test run on a representative subset.

    Returns:
        np.ndarray: the selected points, in their original order
    """
    points = np.asarray(points)
    num_keep = max(1, int(round(len(points) * fraction)))
    dists = np.linalg.norm(points - points.mean(axis=0), axis=1)
    keep = np.sort(np.argsort(dists)[:num_keep])
    return points[keep]


def _process_chunk(chunk_id, points_chunk, robot_urdf_path, robot_home_pos, ik_tol, ee_link_name,
                    robot_base_ori, num_hemisphere_points, look_at_point_offset, hemisphere_radius,
                    num_configs_in_path, motion_planner_type, data_dir):
    """ Runs in a worker process: builds its own PathCache (own PyBullet DIRECT client, own robot,
    own collision environment - nothing is shared with the parent or other workers) and searches
    its assigned chunk of points. Must be a top-level function (not a method/closure) so it can be
    pickled and sent to worker processes.

    ProcessPoolExecutor reuses worker processes across many submitted chunks rather than spawning
    a fresh process per task, so a worker handling e.g. 100 chunks over its lifetime would
    construct 100 PathCache instances in the same process. PathCache never disconnects its
    PyBullet DIRECT client, so each one leaks (measured: ~90MB/chunk, unbounded) unless we
    explicitly disconnect it here once this chunk's work is done - confirmed via a real OOM
    (other apps force-closed, a worker got killed mid-run) on the first full-scale attempt.

    Returns:
        dict: chunk_id, n_points, elapsed seconds, and the 3 saved file paths (or None each).
    """
    import time
    t0 = time.time()

    path_cache = PathCache(
        robot_urdf_path=robot_urdf_path,
        renders=False,  # GUI mode isn't meaningful across parallel workers
        robot_home_pos=robot_home_pos,
        ik_tol=ik_tol,
        ee_link_name=ee_link_name,
        robot_base_ori=robot_base_ori,
        data_dir=data_dir,
    )

    try:
        saved_paths = path_cache.find_high_manip_ik(
            points=points_chunk,
            num_hemisphere_points=num_hemisphere_points,
            look_at_point_offset=look_at_point_offset,
            hemisphere_radius=hemisphere_radius,
            num_configs_in_path=num_configs_in_path,
            motion_planner_type=motion_planner_type,
            save_data=True,
            filename_tag=f"w{chunk_id:04d}",
            verbose=False,
        )
    finally:
        path_cache.pyb.disconnect()
        # PybUtils also registers this same disconnect() at atexit; without unregistering it
        # here, it fires again at process exit against an already-disconnected client and prints
        # a harmless but alarming traceback for every chunk this worker ever processed.
        atexit.unregister(path_cache.pyb.disconnect)

    return {
        "chunk_id": chunk_id,
        "n_points": len(points_chunk),
        "elapsed": time.time() - t0,
        "saved_paths": saved_paths,
    }


def merge_outputs(chunk_results, data_dir):
    """ Concatenates each chunk's saved (csv, npy, csv) outputs into one final combined set of
    files, tagged with the merge's own timestamp. Reads the exact paths each chunk reported
    saving (rather than globbing data_dir) so a merge can't accidentally pick up unrelated files
    left over from another run.

    Args:
        chunk_results (list[dict]): results as returned by _process_chunk
        data_dir (Path): directory the chunk files live in and the merged files are written to

    Returns:
        tuple[Path, Path, Path]: merged (voxel_ik_data csv, reachable_paths npy, reachable_voxels csv)
    """
    csv_rows, path_arrays, voxel_rows = [], [], []

    for result in sorted(chunk_results, key=lambda r: r["chunk_id"]):
        saved = result["saved_paths"]
        if saved is None:
            continue
        csv_path, paths_path, voxels_path = saved

        csv_data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
        if csv_data.size:
            csv_rows.append(csv_data.reshape(-1, 14))

        path_arr = np.load(paths_path)
        if path_arr.shape[-1]:
            path_arrays.append(path_arr)

        voxel_data = np.loadtxt(voxels_path)
        if voxel_data.size:
            voxel_rows.append(voxel_data.reshape(-1, 3))

    combined_csv = np.vstack(csv_rows) if csv_rows else np.empty((0, 14))
    combined_paths = np.concatenate(path_arrays, axis=2) if path_arrays else np.empty((0, 0, 0))
    combined_voxels = np.vstack(voxel_rows) if voxel_rows else np.empty((0, 3))

    csv_out = data_dir / timestamped_filename("voxel_ik_data_merged", ".csv")
    np.savetxt(
        csv_out, combined_csv, delimiter=",",
        header="j1,j2,j3,j4,j5,j6,x,y,z,ox,oy,oz,ow,manip", comments="",
    )
    paths_out = data_dir / timestamped_filename("reachable_paths_merged", ".npy")
    np.save(paths_out, combined_paths)
    voxels_out = data_dir / timestamped_filename("reachable_voxels_merged", ".csv")
    np.savetxt(voxels_out, combined_voxels)

    return csv_out, paths_out, voxels_out


def run_parallel(points, robot_urdf_path, robot_home_pos, num_hemisphere_points, look_at_point_offset,
                  hemisphere_radius, num_configs_in_path=100, motion_planner_type='interpolate',
                  ik_tol=0.05, ee_link_name='tool0', robot_base_ori=[0, 0, 0], num_workers=6,
                  chunk_size=200, data_dir=None, seed=0):
    """ Splits points across num_workers processes, each running its own independent PathCache
    search, then merges every chunk's output into one combined result.

    Every call gets its own timestamped subdirectory under data_dir (run_<YYYYMMDD_HHMMSS>/) so
    repeated runs never collide or overwrite each other's chunk/merged files - data_dir is the
    parent location runs are created under, not the exact directory files land in.

    num_workers/chunk_size are yours to tune: this machine has 8 physical cores (4P+4E hybrid, no
    hyperthreading) - num_workers=6 leaves headroom for the OS and whatever else is running.
    chunk_size trades per-chunk PathCache setup overhead (~0.3s, negligible in aggregate) against
    load-balancing granularity; smaller chunks balance better across cost variance, which matters
    most for motion_planner_type='rrt' given its heavy-tailed per-point cost (a handful of
    expensive tree searches can dominate a chunk's runtime) - consider chunk_size ~50-100 for rrt
    vs ~200-300 for interpolate/two_stage_cartesian.

    Returns:
        tuple[Path, Path, Path]: merged (voxel_ik_data csv, reachable_paths npy, reachable_voxels csv)
    """
    data_dir = get_data_dir() if data_dir is None else Path(data_dir)
    run_dir = data_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    chunks = build_chunks(points, chunk_size, seed=seed)
    print(f"Split {len(points)} points into {len(chunks)} chunks of up to {chunk_size} "
          f"across {num_workers} workers, writing to {run_dir}")

    results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(
                _process_chunk, chunk_id, chunk, robot_urdf_path, robot_home_pos, ik_tol,
                ee_link_name, robot_base_ori, num_hemisphere_points, look_at_point_offset,
                hemisphere_radius, num_configs_in_path, motion_planner_type, run_dir,
            ): chunk_id
            for chunk_id, chunk in enumerate(chunks)
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(f"[{len(results)}/{len(chunks)} chunks done] "
                  f"chunk {result['chunk_id']}: {result['n_points']} points in {result['elapsed']:.1f}s")

    print("Merging chunk outputs...")
    merged = merge_outputs(results, run_dir)
    print(f"Merged output: {merged}")
    return merged


def parse_args():
    parser = argparse.ArgumentParser(description="Parallel PathCache.find_high_manip_ik runner")
    parser.add_argument('--motion-planner-type', choices=MOTION_PLANNER_TYPES, default='interpolate',
                         help="Motion planner used to connect the robot's home position to each "
                              "candidate configuration. 'approach_cartesian' is the only option that "
                              "guarantees a continuous joint path. 'rrt' is far slower and much higher-"
                              "variance than the others - consider a smaller --chunk-size with it.")
    parser.add_argument('--num-workers', type=int, default=6,
                         help="Worker processes. This machine has 8 physical cores; the default "
                              "leaves 2 free for the OS/interactive use.")
    parser.add_argument('--chunk-size', type=int, default=200,
                         help="Points per chunk. Smaller chunks balance load better across cost "
                              "variance (worth it for 'rrt') at the cost of more per-chunk PathCache "
                              "setup overhead (~0.3s/chunk, negligible at these scales).")
    parser.add_argument('--voxel-file', default='/home/marcus/imml/trajectory_cache/data/voxel_data_parallelepiped.csv',
                         help="Path to the voxel centers CSV (first 3 columns are x,y,z).")
    parser.add_argument('--data-dir', default=None,
                         help="Parent directory each run's timestamped run_<YYYYMMDD_HHMMSS>/ "
                              "subdirectory (holding that run's chunk and merged files) is created "
                              "under. Defaults to the package's data/ directory (see get_data_dir).")
    parser.add_argument('--seed', type=int, default=0,
                         help="Shuffle seed used when splitting points into chunks (see build_chunks).")
    parser.add_argument('--center-fraction', type=float, default=None,
                         help="Only search this fraction of voxels closest to the voxel cloud's "
                              "centroid (e.g. 0.1 for the middle 10%%). Defaults to all voxels.")
    args = parser.parse_args()
    if args.center_fraction is not None and not 0 < args.center_fraction <= 1:
        parser.error("--center-fraction must be in (0, 1]")
    return args


if __name__ == "__main__":
    args = parse_args()

    z_base_rotation = np.pi / 4  # Rotate base of robot by 45 degrees
    # robot_home_pos = [z_base_rotation, -np.pi / 2, 2 * np.pi / 3, 5 * np.pi / 6, -np.pi / 2, 0]
    robot_home_pos = [z_base_rotation, -2.755, 1.72, 4.71, -1.58, 0]

    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_urdf_file = os.path.join(script_dir, 'urdf', 'ur5e', 'ur5e.urdf')

    data_dir = get_data_dir() if args.data_dir is None else args.data_dir
    voxel_data = np.loadtxt(args.voxel_file)
    voxel_centers = voxel_data[:, :3]

    translation = np.array([-0.092075, 1.0, 0.5])
    voxel_centers_shifted = voxel_centers + translation

    if args.center_fraction is not None:
        num_total = len(voxel_centers_shifted)
        voxel_centers_shifted = select_center_points(voxel_centers_shifted, args.center_fraction)
        print(f"Using the {len(voxel_centers_shifted)}/{num_total} voxels closest to the centroid "
              f"(--center-fraction {args.center_fraction})")

    run_parallel(
        points=voxel_centers_shifted,
        robot_urdf_path=default_urdf_file,
        robot_home_pos=robot_home_pos,
        num_hemisphere_points=[16, 16],
        look_at_point_offset=0.0,
        hemisphere_radius=0.10,
        num_configs_in_path=100,
        motion_planner_type=args.motion_planner_type,
        ee_link_name='gripper_link',
        robot_base_ori=[0, 0, z_base_rotation],
        num_workers=args.num_workers,
        chunk_size=args.chunk_size,
        data_dir=data_dir,
        seed=args.seed,
    )
