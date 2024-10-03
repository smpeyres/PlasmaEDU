################################################################################
#
#  BFIELD
#
#   Example of plotting the magnitude of the magnetic field
#   produced by a current loop using the Biot-Savart law with filament discretization
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

# Define the grid in the XY-plane where the B-field will be calculated
X = np.linspace(-0.1, 0.1, 50)    # X coordinates [m]
Y = np.linspace(-0.1, 0.1, 50)    # Y coordinates [m]
Z = 0.0                            # Z coordinate [m] (XY-plane)

# Initialize the B-field magnitude array
Bnorm = np.zeros((X.size, Y.size))

# Initialize the point as a (3,1) array to match biotsavart's expectation
point = np.zeros((3, 1))

# Compute the magnetic field at each grid point using Biot-Savart
for i in range(X.size):
    for j in range(Y.size):
        point[0, 0] = X[i]
        point[1, 0] = Y[j]
        point[2, 0] = Z
        Bx, By, Bz = bfield.biotsavart(filament, I0 * Nturns, point)
        Bnorm[i, j] = np.sqrt(Bx**2 + By**2 + Bz**2)

# Create the plot
plt.figure(figsize=(8, 6))
XX, YY = np.meshgrid(X, Y)

# Plot the B-field magnitude using contourf
contour = plt.contourf(XX, YY, Bnorm.T, levels=30, cmap='viridis')
plt.colorbar(contour, label='|B| [T]')

# Overlay the current loop for visualization
plt.plot(filament[0, :], filament[1, :], 'w--', linewidth=1, label='Current Loop')

# Set plot labels and title
plt.xlabel('X [m]')
plt.ylabel('Y [m]')
plt.title('Magnetic Field Magnitude [T] of a Current Loop (Biot-Savart)')
plt.legend()

# Save and display the plot
plt.savefig('ex12_plot_filament_loopxyz.png', dpi=150)
plt.show()
