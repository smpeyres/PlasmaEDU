import numpy as np
import bfield
import matplotlib.pyplot as plt

def make_helix(Ra, La, Nturns, Npoints, Center=np.array([0,0,0]), EulerAngles=np.array([0,0,0])):
    """
    Generate a helical filament representing a finite solenoid.
    """
    # Parametric angle
    phi = np.linspace(0, 2*np.pi*Nturns, Npoints)

    # Helix equations before rotation and translation
    X = Ra * np.cos(phi)
    Y = Ra * np.sin(phi)
    Z = (La / (2*np.pi)) * phi  # Ensures La is the pitch length per turn

    # Combine into filament array
    filament_local = np.vstack((X, Y, Z))

    # Rotation matrix
    R = bfield.roto(EulerAngles)

    # Rotate filament
    filament_rotated = R @ filament_local

    # Translate filament to Center
    filament = filament_rotated + Center[:, np.newaxis]

    return filament

# ------------------------------ Main Script ----------------------------------

# Helical Solenoid Parameters
Ra = 0.05            # Helix radius [m]
I0 = 1000.0          # Current [A]
Nturns = 10          # Number of turns (set so the filament covers the full range)
La = 0.10           # Adjust pitch length to cover Z = [-0.5, 0.5]
Npoints = 1000        # Total number of discretization points
Center = np.array([0, 0, -0.5])    # Shift center so helix starts at Z = -0.5
EulerAngles = np.array([0, 0, 0]) * np.pi / 180.0  # Euler angles [rad] (no rotation)

# Generate the helical filament
filament = make_helix(Ra, La, Nturns, Npoints, Center, EulerAngles)

# Define the grid where the B-field will be calculated (side view in XZ-plane)
X_grid = np.linspace(-0.15, 0.15, 100)  # X coordinates [m]
Z_grid = np.linspace(-0.5, 0.5, 100)    # Z coordinates [m]
Y_plane = 0.0                           # Y coordinate [m] for the XZ-plane

# Initialize the B-field magnitude array
Bnorm = np.zeros((X_grid.size, Z_grid.size))
Bz_field = np.zeros((X_grid.size, Z_grid.size))

# Initialize the point as a (3,1) array to match biotsavart's expectation
point = np.zeros((3, 1))

# Compute the magnetic field at each grid point using Biot-Savart
print("Calculating magnetic field... This may take a while depending on grid size and filament points.")
for i in range(X_grid.size):
    for j in range(Z_grid.size):
        point[0, 0] = X_grid[i]
        point[1, 0] = Y_plane
        point[2, 0] = Z_grid[j]
        Bx, By, Bz = bfield.biotsavart(filament, I0, point)
        Bnorm[i, j] = np.sqrt(Bx**2 + By**2 + Bz**2)
        Bz_field[i, j] = Bz


print("Magnetic field calculation completed.")

# Find the index where X = 0
x_zero_index = np.abs(X_grid - 0).argmin()

# Extract the B-field magnitudes along the Z-axis (at X = 0)
Bz_z_axis = Bz_field[x_zero_index, :]

# Calculate the average B-field magnitude along the Z-axis
average_Bz_z_axis = np.mean(Bz_z_axis)

print(average_Bz_z_axis)

# Create the plot in the XZ-plane (side view)
fig1 = plt.figure(figsize=(10, 6))
XX, ZZ = np.meshgrid(X_grid, Z_grid)

# Plot the B-field magnitude using contourf
contour = plt.contourf(XX, ZZ, Bnorm.T, levels=50, cmap='viridis')
plt.colorbar(contour, label='|B| [T]')

# Overlay the helical filament projection onto the XZ-plane
plt.plot(filament[0, :], filament[2, :], 'r--', linewidth=1, label='Helical Filament')

# Set plot labels and title
plt.xlabel('X [m]')
plt.ylabel('Z [m]')
plt.xlim(np.min(X_grid), np.max(X_grid))
plt.ylim(np.min(Z_grid), np.max(Z_grid))
plt.title('Magnetic Field Magnitude [T] and Helical Current in the XZ-plane')
plt.legend()

# Save and display the plot
plt.savefig('ex13_plot_helical_filament_XZ_view_fixed.png', dpi=150)
plt.show()

fig2 = plt.figure(figsize=(10, 6))
XX, ZZ = np.meshgrid(X_grid, Z_grid)

# Plot the B-field magnitude using contourf
contour = plt.contourf(XX, ZZ, Bz_field.T, levels=50, cmap='viridis')
plt.colorbar(contour, label='|B| [T]')

# Overlay the helical filament projection onto the XZ-plane
plt.plot(filament[0, :], filament[2, :], 'r--', linewidth=1, label='Helical Filament')

# Set plot labels and title
plt.xlabel('X [m]')
plt.ylabel('Z [m]')
plt.xlim(np.min(X_grid), np.max(X_grid))
plt.ylim(np.min(Z_grid), np.max(Z_grid))
plt.title('Bz [T] and Helical Current in the XZ-plane')
plt.legend()

# Save and display the plot
plt.savefig('ex13_plot_helical_filament_XZ_Bz.png', dpi=150)
plt.show()

