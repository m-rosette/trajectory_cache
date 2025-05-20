import numpy as np
import matplotlib.pyplot as plt
import pybullet as p

def prune_arc(prune_point, radius, allowance_angle, num_points, x_ori_default=0.0, y_ori_default=np.pi/2):
    # Define theta as a discrete array starting from positive y-axis and moving to negative y-axis
    theta = np.linspace(3 * np.pi/2 - allowance_angle, 3 * np.pi/2 + allowance_angle, num_points)

    # Set up arc length coordinate
    x = np.full_like(theta, prune_point[0])  # x-coordinate remains constant
    z = -radius * np.cos(theta) + prune_point[2]  # z calculation
    y = radius * np.sin(theta) + prune_point[1]  # y calculation

    # Calculate orientation angles
    arc_angles = np.arctan2(prune_point[1] - y, prune_point[2] - z)

    arc_coords = np.vstack((x, y, z))

    goal_coords = np.zeros((num_points, 3))  # 3 for x, y, z
    goal_orientations = np.zeros((num_points, 3))  # 3 for euler angles
    for i in range(num_points):
        goal_coords[i] = [arc_coords[0][i], arc_coords[1][i], arc_coords[2][i]]
        goal_orientations[i] = [x_ori_default, arc_angles[i], y_ori_default]

    return goal_coords, goal_orientations

def plot_orientations(goal_coords, goal_orientations, length):
    plt.figure()
    for i in range(len(goal_coords)):
        start_point = goal_coords[i]
        angle = goal_orientations[i]
        print(f"Start point: {start_point} - Angle: {angle[1]}")
        
        end_point = [start_point[0],
                     start_point[1] + length * np.sin(angle[1]),
                     start_point[2] + length * np.cos(angle[1])
                     ]
        
        plt.plot([start_point[1], end_point[1]], [start_point[2], end_point[2]], marker='o')
        
    plt.grid(True)
    plt.xlabel("Y")
    plt.ylabel("Z")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

if __name__ == "__main__":
    prune_point = [0.5, 0.5, 0.8]
    radius = 0.1
    allowance_angle = np.deg2rad(30)
    num_points = 3
    length = 0.1

    goal_coords, goal_orientations = prune_arc(prune_point, radius, allowance_angle, num_points)
    plot_orientations(goal_coords, goal_orientations, length)
