import numpy as np
import bfield
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# Helical Solenoid Parameters
I0 = 1000.0
Ra = 0.05
La = 0.50
Nturns = 10.0
Npoints = 500
phi0 = 0.0
Center = np.array([0, 0, 0.0])
EulerAngles = np.array([0, 0, 0]) * np.pi / 180.0

# Define the grid
X_grid = np.array([0.0])
Y_grid = np.array([0.0])
Z_grid = np.array([0.0])

_, _, _, filament = bfield.helix(X_grid, Y_grid, Z_grid, I0, Ra, La, Nturns, Npoints, phi0, Center, EulerAngles)

# Create a 3D plot for the helical filament
fig1 = plt.figure(figsize=(10, 8))
ax = fig1.add_subplot(111, projection='3d')

# Plot the helical filament
ax.plot(filament[0, :], filament[1, :], filament[2, :], 'r-', linewidth=2, label='Helical Filament')

# Set plot labels and title
ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.set_zlabel('Z [m]')
ax.set_title('3D Visualization of Helical Filament')

# Set equal scaling for all axes
ax.set_box_aspect([1, 1, 1])  # Aspect ratio is 1:1:1
ax.set_aspect('auto')
ax.set_xlim([-0.1, 0.1])
ax.set_ylim([-0.1, 0.1])
ax.set_zlim([0, La])

# Show legend
ax.legend()

# Save and display the plot
plt.savefig('M1-figs/ex14_plot_helical_filament_3D.png', dpi=600)
plt.show()

# Define the grid
X_grid = np.linspace(-0.10, 0.10, 20)
Y_grid = np.linspace(-0.10, 0.10, 20)
Z_grid = np.linspace(-0.025, La+0.025, 50)

Bx, By, Bz, filament = bfield.helix(X_grid, Y_grid, Z_grid, I0, Ra, La, Nturns, Npoints, phi0, Center, EulerAngles)

Bnorm = np.sqrt(Bx**2 + By**2 + Bz**2)

# Plot the B-field magnitude in the XZ-plane
Y_target = 0.0
j = np.abs(Y_grid - Y_target).argmin()

XX, ZZ = np.meshgrid(X_grid, Z_grid)

# Create plot for magnetic field magnitude |B| in the XZ-plane
fig2 = plt.figure(figsize=(10, 6))
contour = plt.contourf(XX, ZZ, Bnorm[:, j, :].T, levels=50, cmap='viridis')
plt.colorbar(contour, label='|B| [T]')

# Overlay the helical filament as a dashed red line
filament_indices = np.abs(filament[1, :] - Y_target).argmin()  # Closest to Y_target
plt.plot(filament[0, :], filament[2, :], 'r--', linewidth=1)  # Dashed red line

# Set plot labels and title
plt.xlabel('X [m]')
plt.ylabel('Z [m]')
plt.title('Magnetic Field Magnitude [T] and Helical Current in the XZ-plane')

# Show legend once
# plt.legend()

# Save and display the plot
plt.savefig('M1-figs/ex14_plot_helical_filament_XZ_Bnorm_fixed.png', dpi=600)
plt.show()

# Create the plot of Bz component in the XZ-plane at Y=Y_target
fig3 = plt.figure(figsize=(10, 6))
contour = plt.contourf(XX, ZZ, Bz[:, j, :].T, levels=50, cmap='viridis')
plt.colorbar(contour, label='Bz [T]')

# Overlay the helical filament again with the dashed red line
plt.plot(filament[0, :], filament[2, :], 'r--', linewidth=1)  # No label here

# Set plot labels and title
plt.xlabel('X [m]')
plt.ylabel('Z [m]')
plt.title('Magnetic Field Component Bz [T] and Helical Current in the XZ-plane')

# Save and display the plot
plt.savefig('M1-figs/ex14_plot_helical_filament_XZ_Bz_fixed.png', dpi=600)
plt.show()

# Create a 3D plot for the magnetic field vectors and the helical coil
fig4 = plt.figure(figsize=(12, 10))
ax = fig4.add_subplot(111, projection='3d')

# Create a grid for quiver plot
X, Y, Z = np.meshgrid(X_grid, Y_grid, Z_grid, indexing='ij')

# Plot the magnetic field vectors using quiver
skip = (slice(None, None, 2), slice(None, None, 2), slice(None, None, 2))  # Skip some points for better visualization
ax.quiver(X[skip], Y[skip], Z[skip], Bx[skip], By[skip], Bz[skip], length=0.01, normalize=True, color='b', label='Magnetic Field Vectors')

# Overlay the helical filament
ax.plot(filament[0, :], filament[1, :], filament[2, :], 'r-', linewidth=2, label='Helical Filament')

# Set plot labels and title
ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.set_zlabel('Z [m]')
ax.set_title('3D Visualization of Magnetic Field Vectors and Helical Filament')

# Set equal scaling for all axes
ax.set_box_aspect([1, 1, 1])  # Aspect ratio is 1:1:2
ax.set_xlim([-0.1, 0.1])
ax.set_ylim([-0.1, 0.1])
ax.set_zlim([-0.1, La+0.1])

# Show legend
ax.legend()

# Save and display the plot
plt.savefig('M1-figs/ex14_plot_magnetic_field_vectors_3D.png', dpi=600)
plt.show()

# # ------------------ Added Code for Average Bz along Z-axis ------------------

# # Find indices where X = 0 and Y = 0
# i = np.abs(X_grid - 0).argmin()
# j = np.abs(Y_grid - 0).argmin()

# # Extract the Bz components along the Z-axis at X=0, Y=0
# Bz_z_axis = Bz[i, j, :]

# # Calculate the average Bz along the Z-axis
# average_Bz_z_axis = np.mean(Bz_z_axis)

# print("Average Bz along the Z-axis:", average_Bz_z_axis)

# # Optional: Plot Bz along the Z-axis at X=0, Y=0
# fig3 = plt.figure(figsize=(8, 6))
# plt.plot(Z_grid, Bz_z_axis, 'b-', label='Bz along Z-axis at X=0, Y=0')
# plt.xlabel('Z [m]')
# plt.ylabel('Bz [T]')
# plt.title('Bz Component along Z-axis at X=0, Y=0')
# plt.legend()
# plt.grid(True)
# plt.savefig('M1-figs/ex14_Bz_along_Z_axis_fixed.png', dpi=600)
# plt.show()
