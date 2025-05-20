import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R

# Number of points along each dimension
num_theta = 30
num_phi = 30

# Parameters for the hemisphere
origin = np.array([1.0, 1.0, 0.0])  # New origin of the hemisphere
radius = 1.0  # Radius of the hemisphere

# Create a meshgrid for theta and phi
theta = np.linspace(-np.pi, 0, num_theta)  # Azimuthal angle
phi = np.linspace(0, np.pi, num_phi)  # Elevation angle (0 to pi/2 for hemisphere)

theta, phi = np.meshgrid(theta, phi)

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

# Hemisphere vertex (center of the hemisphere)
vertex = np.array(origin)

# Calculate vectors from each point to the vertex
vectors = vertex - np.stack((x, y, z), axis=-1)

# Normalize vectors
norms = np.linalg.norm(vectors, axis=2)
vectors /= norms[:, :, np.newaxis]

# Filter the points to keep only those within ±30 degrees from the y-axis
# Calculate the angle from the y-axis for each point
angles_from_y = np.arccos(np.abs((coordinates_unique[:, 1] - origin[1]) / radius))

# Convert 30 degrees to radians
angle_threshold = np.deg2rad(30)

# Filter the coordinates based on the angle threshold
filtered_coordinates = coordinates_unique[(angles_from_y <= angle_threshold)  & (coordinates_unique[:, 2] <= vertex[2])]

# Extract the filtered x, y, z coordinates
x_filtered = filtered_coordinates[:, 0]
y_filtered = filtered_coordinates[:, 1]
z_filtered = filtered_coordinates[:, 2]

coordinates_filtered = np.vstack((x_filtered, y_filtered, z_filtered)).T

# Plot the hemisphere and discretized points
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the hemisphere surface
ax.plot_surface(x, y, z, color='c', alpha=0.6, rstride=5, cstride=5)

# Plot the discrete points
# ax.scatter(x, y, z, color='r', s=10)
ax.scatter(x_filtered, y_filtered, z_filtered, color='r', s=10)

# # Plot the vectors pointing towards the center
# for i in range(num_theta):
#     for j in range(num_phi):
#         ax.quiver(x[i, j], y[i, j], z[i, j],
#                   vectors[i, j, 0], vectors[i, j, 1], vectors[i, j, 2],
#                   length=0.25, color='k')

# Set axis limits to fit the hemisphere
ax.set_xlim([origin[0] - radius - 0.5, origin[0] + radius + 0.5])
ax.set_ylim([origin[1] - radius - 0.5, origin[1] + radius + 0.5])
ax.set_zlim([origin[2] - radius - 0.5, origin[2] + radius + 0.5])

# Set axis labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Make the axes have the same scale
ax.set_box_aspect([1, 1, 1])  # Aspect ratio is 1:1:1

plt.show()