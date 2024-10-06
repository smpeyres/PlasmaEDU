################################################################################
#
#  BFIELD
#
#   Example of plotting the magnitude of the magnetic field
#   produced by a current loop using the Biot-Savart law with filament discretization
#   Modified to compute over a 3D domain
#
################################################################################

import numpy as np
import bfield
import matplotlib.pyplot as plt

# Current Loop Parameters
Ra = 0.05               # Loop radius [m]
I0 = 1000.0             # Loop current [A]
Nturns = 1              # Number of turns
Center = np.array([0, 0, 0])       # Center of the loop [m]
Angles = np.array([90, 0, 0]) * np.pi / 180.0  # Euler angles [rad]
Npoints = 100           # Number of discretization points for the filament

# Create the current filament using makeloop
filament = bfield.makeloop(Ra, Center, Angles, Npoints)

# Define the grid in the XYZ where the B-field will be calculated
X = np.linspace(-0.1, 0.1, 50)    # X coordinates [m]
Y = np.linspace(-0.1, 0.1, 50)    # Y coordinates [m]
Z = np.linspace(-0.1, 0.1, 50)    # Z coordinates [m]

# Initialize the B-field magnitude array
Bnorm = np.zeros((X.size, Y.size, Z.size))

# Initialize the point as a (3,1) array to match biotsavart's expectation
point = np.zeros((3, 1))

# Compute the magnetic field at each grid point using Biot-Savart
for i in range(X.size):
    for j in range(Y.size):
        for k in range(Z.size):
            point[0, 0] = X[i]
            point[1, 0] = Y[j]
            point[2, 0] = Z[k]
            Bx, By, Bz = bfield.biotsavart(filament, I0 * Nturns, point)
            Bnorm[i, j, k] = np.sqrt(Bx**2 + By**2 + Bz**2)

# Find the index of Z closest to zero
Z_target = 0.0
k = np.argmin(np.abs(Z - Z_target))

# Create the plot at Z=0
plt.figure(figsize=(8, 6))
XX, YY = np.meshgrid(X, Y)

# Plot the B-field magnitude using contourf at Z=0
contour = plt.contourf(XX, YY, Bnorm[:, :, k].T, levels=30, cmap='viridis')
plt.colorbar(contour, label='|B| [T] at Z={}'.format(Z[k]))

# Overlay the current loop for visualization
plt.plot(filament[0, :], filament[1, :], 'w--', linewidth=1, label='Current Loop')

# Set plot labels and title
plt.xlabel('X [m]')
plt.ylabel('Y [m]')
plt.title('Magnetic Field Magnitude [T] of a Current Loop (Biot-Savart) at Z={}'.format(Z[k]))
plt.legend()

# Save and display the plot
plt.savefig('ex12_plot_filament_loopxyz_Z{}.png'.format(k), dpi=150)
plt.show()
