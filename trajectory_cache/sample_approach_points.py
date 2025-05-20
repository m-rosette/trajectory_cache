import pybullet as p
import numpy as np
from scipy.spatial.transform import Rotation as R


def prune_arc(prune_point, radius, allowance_angle, num_arc_points, y_ori_default=0.0, z_ori_default=0):
    # Define theta as a descrete array
    theta = np.linspace(3 * np.pi/2 - allowance_angle, 3 * np.pi/2 + allowance_angle, num_arc_points)

    # Set up arc length coordinate
    x = np.full_like(theta, prune_point[0])  # x-coordinate remains constant
    z = - radius * np.cos(theta) + prune_point[2] # multiply by a negative to mirror on other side of axis
    y = radius * np.sin(theta) + prune_point[1] 

    # Calculate orientation angles
    arc_angles = np.arctan2(prune_point[1] - y, prune_point[2] - z)

    arc_coords = np.vstack((x, y, z))

    goal_coords = np.zeros((num_arc_points, 3)) # 3 for x, y, z
    goal_orientations = np.zeros((num_arc_points, 4)) # 4 for quaternion
    for i in range(num_arc_points):
        goal_coords[i] = [arc_coords[0][i], arc_coords[1][i], arc_coords[2][i]]
        goal_orientations[i] = p.getQuaternionFromEuler([-arc_angles[i], y_ori_default, z_ori_default])

    return goal_coords, goal_orientations

def sample_hemisphere_suface_pts(look_at_point, look_at_point_offset, radius, num_points=[30, 30], angle_threshold=np.pi/6):
    """
    Generates points on the surface of a hemisphere that lies on the xz-plane protruding along the y-axis 
    """
    # Number of points along each dimension
    num_theta = num_points[0]
    num_phi = num_points[1]

    # Create a meshgrid for theta and phi
    theta = np.linspace(-np.pi, 0, num_theta)  # Azimuthal angle
    phi = np.linspace(0, np.pi, num_phi)  # Elevation angle (0 to pi/2 for hemisphere)

    theta, phi = np.meshgrid(theta, phi)

    # Define the hemisphere origin
    origin = np.copy(look_at_point)
    origin[1] -= look_at_point_offset 

    # Convert spherical coordinates to Cartesian and scale by radius
    x = radius * np.sin(phi) * np.cos(theta) + origin[0]
    y = radius * np.sin(phi) * np.sin(theta) + origin[1]
    z = radius * np.cos(phi) + origin[2] 

    # Flatten the coordinate arrays to create a list of points
    x_flat = x.flatten()
    y_flat = y.flatten()
    z_flat = z.flatten()

    # Combine the x, y, z arrays into a single array of points
    coordinates = np.vstack((x_flat, y_flat, z_flat)).T
    
    # Remove duplicate rows from the coordinates array
    coordinates_unique = np.unique(coordinates, axis=0) 

    # Filter the points to keep only those within ±30 degrees from the y-axis
    # Calculate the angle from the y-axis for each point
    angles_from_y = np.arccos(np.abs((coordinates_unique[:, 1] - origin[1]) / radius))

    # Filter the coordinates based on the angle threshold
    filtered_coordinates = coordinates_unique[(angles_from_y <= angle_threshold) & (coordinates_unique[:, 2] <= origin[2])]

    # Extract the filtered x, y, z coordinates
    x_filtered = filtered_coordinates[:, 0]
    y_filtered = filtered_coordinates[:, 1]
    z_filtered = filtered_coordinates[:, 2]

    return np.vstack((x_filtered, y_filtered, z_filtered)).T

def hemisphere_orientations(point1, points2):
    """
    Calculate the orientation quaternions of 3D lines passing through a single point1 and multiple points in points2.

    Parameters:
    - point1: Tuple or list of 3 coordinates (x1, y1, z1) for the first point.
    - points2: Array of shape (N, 3) where each row represents a point (x2, y2, z2) for the second point.

    Returns:
    - Array of shape (N, 4) where each row is a quaternion (x, y, z, w) representing the rotation from the z-axis to the line direction.
    """
    # Convert point1 to numpy array
    point1 = np.array(point1)
    
    # Compute the direction vectors for each point in points2
    direction_vectors = point1 - points2  # Subtract points2 from point1 (reverse direction)
    direction_vectors = direction_vectors / np.linalg.norm(direction_vectors, axis=1)[:, np.newaxis]  # Normalize direction vectors
    
    # Define the reference direction (z-axis)
    reference_direction = np.array([0, 0, 1])
    
    quaternions = np.zeros((direction_vectors.shape[0], 4))  # Prepare an array for quaternions
    
    # Compute the quaternion for each direction vector
    for i, direction_vector in enumerate(direction_vectors):
        rotation = R.align_vectors([direction_vector], [reference_direction])[0]

        # Convert the quaternion to Euler angles (roll, pitch, yaw)
        euler_angles = rotation.as_euler('xyz')  # Get Euler angles in x-y-z order (roll, pitch, yaw)
        
        # Fix the roll angle to maintain the camera being upright
        euler_angles[1] = 0  # Fix the roll (around y-axis)
        
        # Convert the modified Euler angles back to a quaternion
        fixed_rotation = R.from_euler('xyz', euler_angles)

        quaternions[i] = rotation.as_quat()  # scipy returns in (x, y, z, w) order by default
    
    return quaternions