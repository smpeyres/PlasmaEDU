import numpy as np
import bfield
import matplotlib.pyplot as plt

def make_helix(Ra, La, Nturns, Npoints, Center=np.array([0,0,0]), EulerAngles=np.array([0,0,0])):
    """
    Generate a helical filament representing a finite solenoid, starting from Z=0 and building up.
    """
    phi = np.linspace(0, 2*np.pi*Nturns, Npoints)

    X = Ra * np.cos(phi)
    Y = Ra * np.sin(phi)
    Z = La * phi / (2*np.pi*Nturns)

    filament_local = np.vstack((X, Y, Z))

    R = bfield.roto(EulerAngles)
    filament_rotated = R @ filament_local
    filament = filament_rotated + Center[:, np.newaxis]

    return filament

# Helical Solenoid Parameters
Ra = 0.05
I0 = 1000.0
Nturns = 10
La = 0.50
Npoints = 500
Center = np.array([0, 0, 0.0])
EulerAngles = np.array([0, 0, 0]) * np.pi / 180.0

# Generate the helical filament
filament = make_helix(Ra, La, Nturns, Npoints, Center, EulerAngles)

# Define the grid
X_grid = np.linspace(-0.15, 0.15, 30)
Y_grid = np.linspace(-0.15, 0.15, 30)
Z_grid = np.linspace(0.0, La, 50)

# Initialize the B-field magnitude arrays
Bnorm = np.zeros((X_grid.size, Y_grid.size, Z_grid.size))
Bz_field = np.zeros((X_grid.size, Y_grid.size, Z_grid.size))

# Compute the magnetic field at each grid point
for i, x in enumerate(X_grid):
    for j, y in enumerate(Y_grid):
        for k, z in enumerate(Z_grid):
            point = np.array([[x], [y], [z]])
            Bfield = bfield.biotsavart(filament, I0, point)
            Bnorm[i, j, k] = np.linalg.norm(Bfield)
            Bz_field[i, j, k] = Bfield[2]

# Plot the B-field magnitude in the XZ-plane
Y_target = 0.0
j = np.abs(Y_grid - Y_target).argmin()

XX, ZZ = np.meshgrid(X_grid, Z_grid)

# Create plot for magnetic field magnitude |B| in the XZ-plane
fig1 = plt.figure(figsize=(10, 6))
contour = plt.contourf(XX, ZZ, Bnorm[:, j, :].T, levels=50, cmap='viridis')
plt.colorbar(contour, label='|B| [T]')

# Overlay the helical filament as a dashed red line
filament_indices = np.abs(filament[1, :] - Y_target).argmin()  # Closest to Y_target
plt.plot(filament[0, :], filament[2, :], 'r--', linewidth=1, label='Helical Filament')  # Dashed red line

# Set plot labels and title
plt.xlabel('X [m]')
plt.ylabel('Z [m]')
plt.title('Magnetic Field Magnitude [T] and Helical Current in the XZ-plane')

# Show legend once
plt.legend()

# Save and display the plot
plt.savefig('M1-figs/ex13_plot_helical_filament_XZ_Bnorm_fixed.png', dpi=150)
plt.show()

# Create the plot of Bz component in the XZ-plane at Y=Y_target
fig2 = plt.figure(figsize=(10, 6))
contour = plt.contourf(XX, ZZ, Bz_field[:, j, :].T, levels=50, cmap='viridis')
plt.colorbar(contour, label='Bz [T]')

# Overlay the helical filament again with the dashed red line
plt.plot(filament[0, :], filament[2, :], 'r--', linewidth=1)  # No label here

# Set plot labels and title
plt.xlabel('X [m]')
plt.ylabel('Z [m]')
plt.title('Magnetic Field Component Bz [T] in XZ-plane at Y={:.3f} m'.format(Y_grid[j]))

# Save and display the plot
plt.savefig('M1-figs/ex13_plot_helical_filament_XZ_Bz_fixed.png', dpi=150)
plt.show()

# ------------------ Added Code for Average Bz along Z-axis ------------------

# Find indices where X = 0 and Y = 0
i = np.abs(X_grid - 0).argmin()
j = np.abs(Y_grid - 0).argmin()

# Extract the Bz components along the Z-axis at X=0, Y=0
Bz_z_axis = Bz_field[i, j, :]

# Calculate the average Bz along the Z-axis
average_Bz_z_axis = np.mean(Bz_z_axis)

print("Average Bz along the Z-axis:", average_Bz_z_axis)

# Optional: Plot Bz along the Z-axis at X=0, Y=0
fig3 = plt.figure(figsize=(8, 6))
plt.plot(Z_grid, Bz_z_axis, 'b-', label='Bz along Z-axis at X=0, Y=0')
plt.xlabel('Z [m]')
plt.ylabel('Bz [T]')
plt.title('Bz Component along Z-axis at X=0, Y=0')
plt.legend()
plt.grid(True)
plt.savefig('M1-figs/ex13_Bz_along_Z_axis_fixed.png', dpi=150)
plt.show()
