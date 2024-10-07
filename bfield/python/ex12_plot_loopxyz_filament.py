import numpy as np
import bfield
import matplotlib.pyplot as plt

# Loop Parameters
Ra = 0.05  # Radius of the loop
Npoints = 100  # Number of points to resolve the loop
Center = np.array([0, 0, 0])  # Center of the loop

# Adjust Euler Angles to orient the loop flat in the XY-plane
# Try setting the second angle (theta) to rotate the loop flat in the XY-plane
EulerAngles = np.array([0, 90, 0]) * np.pi / 180.0  # 90-degree rotation around X-axis

# Create the loop using bfield.makeloop (correct orientation)
filament = bfield.makeloop(Ra, Center, EulerAngles, Npoints)

# Plot the loop to verify orientation
fig3D = plt.figure(figsize=(8, 6))
ax = fig3D.add_subplot(111, projection='3d')

# Plot the filament (loop)
ax.plot(filament[0, :], filament[1, :], filament[2, :], 'r-', label='Loop')

# Set labels and title
ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.set_zlabel('Z [m]')
ax.set_title('3D Visualization of the Current Loop')

# Show the plot
plt.legend()
plt.show()

# 2D Plot of the loop in the XY-plane
fig2D = plt.figure(figsize=(8, 6))

# Plot the loop in the XY-plane
plt.plot(filament[0, :], filament[1, :], 'r-', label='Loop in XY-plane')

# Set labels and title for the 2D plot
plt.xlabel('X [m]')
plt.ylabel('Y [m]')
plt.title('2D Visualization of the Current Loop in the XY-plane')

# Show the plot with legend
plt.legend()
plt.grid(True)
plt.axis('equal')  # Ensure equal scaling of axes
plt.show()

# Loop Parameters
Ra = 0.05  # Radius of the loop
I0 = 100.0  # Current [A]
Npoints = 50  # Number of points to resolve the loop
Center = np.array([0, 0, 0])  # Center of the loop

# Euler Angles for the loop in the XY-plane
# I don't know why I have twist it this way in order for the signs to make sense.
EulerAngles = np.array([0, 270, 0]) * np.pi / 180.0

# Create the loop using bfield.makeloop
filament = bfield.makeloop(Ra, Center, EulerAngles, Npoints)

# Define the grid where we want to compute the B-field (X, Y, Z grid)
X_grid = np.linspace(-0.1, 0.1, 51)  # X-range [m]
Y_grid = np.array([0.0])  # Y-value, we keep this as a single slice at Y=0
Z_grid = np.linspace(-0.05, 0.05, 50)  # Z-range [m]

# Initialize arrays for storing the B-field magnitude and components
Bnorm = np.zeros((X_grid.size, Y_grid.size, Z_grid.size))
Bx = np.zeros((X_grid.size, Y_grid.size, Z_grid.size))
By = np.zeros((X_grid.size, Y_grid.size, Z_grid.size))
Bz = np.zeros((X_grid.size, Y_grid.size, Z_grid.size))

# Compute the B-field at each point in the XYZ grid
for i in range(0,X_grid.size):
    for j in range(0,Y_grid.size):
        for k in range(0,Z_grid.size):
            point = np.array([X_grid[i], Y_grid[j], Z_grid[k]])  # Point in space
            Bx_value, By_value, Bz_value = bfield.biotsavart(filament, I0, point)

            # Fix any dimensionality issues by explicitly assigning scalar values
            Bx[i, j, k] = Bx_value
            By[i, j, k] = By_value
            Bz[i, j, k] = Bz_value

            Bnorm[i, j, k] = np.sqrt(Bx[i, j, k]**2 + By[i, j, k]**2 + Bz[i, j, k]**2)

# Since Y_grid is just [0.0], we can take the slice at Y=0 and plot the B-field in the XZ-plane
XX, ZZ = np.meshgrid(X_grid, Z_grid)

# Plot the B-field magnitude in the XZ-plane
# Plot the B-field magnitude in the XZ-plane
Y_target = 0.0
j = np.abs(Y_grid - Y_target).argmin()

figA = plt.figure(figsize=(8, 6))
contour = plt.contourf(XX, ZZ, Bnorm[:, j, :].T, levels=50, cmap='viridis')
plt.colorbar(contour, label='|B| [T]')
plt.plot(filament[0, :], filament[2, :], 'r--', label='Current Loop')  # Plot the loop in XZ-plane
plt.xlabel('X [m]')
plt.ylabel('Z [m]')
plt.ylim(-0.05, 0.05)
plt.xlim(0.0, 0.1)
plt.title('Magnetic Field Magnitude [T] of a Current Loop (Biot-Savart) at Y=0.0')
plt.legend()
plt.savefig("M1-figs/ex12_plot_magnitude.png",dpi=600)
plt.show()

figB = plt.figure(figsize=(8, 6))
contour = plt.contourf(XX, ZZ, Bz[:, j, :].T, levels=50, cmap='viridis')
plt.colorbar(contour, label='|B| [T]')
plt.plot(filament[0, :], filament[2, :], 'r--', label='Current Loop')  # Plot the loop in XZ-plane
plt.xlabel('X [m]')
plt.ylabel('Z [m]')
plt.ylim(-0.05, 0.05)
plt.xlim(0.0, 0.1)
plt.title('Magnetic Field Magnitude [T] of a Current Loop (Biot-Savart) at Y=0.0')
plt.legend()
plt.savefig("M1-figs/ex12_plot_Bz.png",dpi=600)
plt.show()

R = np.linspace(  0.0,  0.1, 50 )
Z = np.linspace(-0.05, 0.05, 50 )

# B-field magnitude
Bnorm_brz = np.zeros((R.size,Z.size))
Bz_brz = np.zeros((R.size,Z.size))

for i in range(0,R.size):
  for j in range(0,Z.size):
      Br_b, Bz_b = bfield.loopbrz( Ra, I0, 1, R[i], Z[j])
      Bz_brz[i,j] = Bz_b
      Bnorm_brz[i,j] = np.sqrt( Br_b*Br_b + Bz_b*Bz_b )

# Plot the B-field magnitude in the XZ-plane
X_target = 0.0
i = np.abs(X_grid - X_target).argmin()

R_target = 0.0
r = np.abs(R - R_target).argmin()

fig2 = plt.figure()
plt.plot(Z_grid, np.squeeze(Bz[i, 0, :]), label="filament")
plt.plot(Z, np.squeeze(Bz_brz[r, :]), label="loopbrz")
plt.xlabel("Z [m]")
plt.ylabel("$B_z$ [T]")
plt.title("r = 0 m")
plt.legend()
plt.savefig("M1-figs/ex12_center_Bz.png",dpi=600)
plt.show()

# Plot the B-field magnitude in the XZ-plane
X_target = 0.04
i = np.abs(X_grid - X_target).argmin()

R_target = 0.04
r = np.abs(R - R_target).argmin()

fig3 = plt.figure()
plt.plot(Z_grid, np.squeeze(Bz[i, 0, :]), label="filament")
plt.plot(Z, np.squeeze(Bz_brz[r, :]), label="loopbrz")
plt.xlabel("Z [m]")
plt.ylabel("$B_z$ [T]")
plt.title("r = 0.04 m")
plt.legend()
plt.savefig("M1-figs/ex12_004m_Bz.png",dpi=600)
plt.show()

# Plot the B-field magnitude in the XZ-plane
X_target = 0.06
i = np.abs(X_grid - X_target).argmin()

R_target = 0.06
r = np.abs(R - R_target).argmin()

fig4 = plt.figure()
plt.plot(Z_grid, np.squeeze(Bz[i, 0, :]), label="filament")
plt.plot(Z, np.squeeze(Bz_brz[r, :]), label="loopbrz")
plt.xlabel("Z [m]")
plt.ylabel("$B_z$ [T]")
plt.title("r = 0.06 m")
plt.legend()
plt.savefig("M1-figs/ex12_006m_Bz.png",dpi=600)
plt.show()


# fig2 = plt.figure()



# ################################################################################
# #
# #  BFIELD
# #
# #   Example of plotting the magnitude of the magnetic field
# #   produced by a current loop using the Biot-Savart law with filament discretization
# #   Modified to compute over a 3D domain
# #
# ################################################################################

# import numpy as np
# import bfield
# import matplotlib.pyplot as plt

# # Current Loop Parameters
# Ra = 0.05               # Loop radius [m]
# I0 = 1000.0             # Loop current [A]
# Nturns = 1              # Number of turns
# Center = np.array([0, 0, 0])       # Center of the loop [m]
# Angles = np.array([0, 0, 0]) * np.pi / 180.0  # Euler angles [rad]
# Npoints = 100           # Number of discretization points for the filament

# # Create the current filament using makeloop
# filament = bfield.makeloop(Ra, Center, Angles, Npoints)

# # Define the grid in the XYZ where the B-field will be calculated
# X = np.linspace( 0.0, 0.1, 50)    # X coordinates [m]
# Y = np.array([0.0])    # Y coordinates [m]
# Z = np.linspace(-0.05, 0.5, 50)    # Z coordinates [m]

# # Initialize the B-field magnitude array
# Bnorm = np.zeros((X.size, Y.size, Z.size))

# # Initialize the point as a (3,1) array to match biotsavart's expectation
# point = np.zeros((3, 1))

# # Compute the magnetic field at each grid point using Biot-Savart
# for i in range(X.size):
#     for j in range(Y.size):
#         for k in range(Z.size):
#             point[0, 0] = X[i]
#             point[1, 0] = Y[j]
#             point[2, 0] = Z[k]
#             Bx, By, Bz = bfield.biotsavart(filament, I0 * Nturns, point)
#             Bnorm[i, j, k] = np.sqrt(Bx**2 + By**2 + Bz**2)

# # Find the index of Z closest to zero
# Y_target = 0.0
# j = np.argmin(np.abs(Y - Y_target))

# # Create the plot at Z=0
# plt.figure(figsize=(8, 6))
# XX, ZZ = np.meshgrid(X, Z)

# # Plot the B-field magnitude using contourf at Z=0
# contour = plt.contourf(XX, ZZ, Bnorm[:, j, :].T, levels=30, cmap='viridis')
# plt.colorbar(contour, label='|B| [T] at Y={}'.format(Y[j]))

# # Overlay the current loop for visualization
# plt.plot(filament[0, :], filament[1, :], 'r--', linewidth=1, label='Current Loop')

# # Set plot labels and title
# plt.xlabel('X [m]')
# plt.ylabel('Z [m]')
# plt.title('Magnetic Field Magnitude [T] of a Current Loop (Biot-Savart) at Y={}'.format(Y[j]))
# plt.legend()

# # Save and display the plot
# plt.savefig('ex12_plot_filament_loopxyz_Z{}.png'.format(k), dpi=150)
# plt.show()

# # Create 3D figure
# fig2 = plt.figure(figsize=(10, 6))
# ax = fig2.add_subplot(111, projection='3d')

# # Plot the first helix (red)
# ax.plot(filament[0, :], filament[1, :], filament[2, :], 'r-', label='Helix')


# # Set axis labels
# ax.set_xlabel('X [m]')
# ax.set_ylabel('Y [m]')
# ax.set_zlabel('Z [m]')
# ax.set_title('3D Visualization of Staggered Helical Coils')

# # Add a legend
# ax.legend()

# # Enable rotation for interactive viewing
# plt.show()
