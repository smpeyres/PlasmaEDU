import numpy as np
import bfield
import matplotlib.pyplot as plt

# Helical Solenoid Parameters
I0 = 10000.0
Ra = 0.05
La = 0.50
Nturns = 5.0  # Reduced to 5 turns for visualization
Npoints = 500
phi0 = 0.0
phase_shift = np.pi
Center1 = np.array([0, 0, 0.0])
Center2 = np.array([0, 0, 0])  # Align both helices at the same center
EulerAngles1 = np.array([0, 0, 0]) * np.pi / 180.0
EulerAngles2 = np.array([0, 0, 0]) * np.pi / 180.0

# Define the grid
X_grid = np.linspace(-0.10, 0.10, 20)
Y_grid = np.linspace(-0.10, 0.10, 20)
Z_grid = np.linspace(-0.05, La+0.05, 50)

# Call the undulator function with a phase shift
Bx, By, Bz, filament1, filament2 = bfield.undulator(X_grid, Y_grid, Z_grid, I0, Ra, La, Nturns, Npoints, phi0, phase_shift, Center1, Center2, EulerAngles1, EulerAngles2)

Bnorm = np.sqrt(Bx**2 + By**2 + Bz**2)

# Save the magnetic field components to a .npz file
np.savez('M1-data/magnetic_field_components.npz', X=X_grid, Y=Y_grid, Z=Z_grid, Bx=Bx, By=By, Bz=Bz)

# Plot the B-field magnitude in the XZ-plane
Y_target = 0.0
j = np.abs(Y_grid - Y_target).argmin()

XX, ZZ = np.meshgrid(X_grid, Z_grid)

# Create plot for magnetic field magnitude |B| in the XZ-plane
fig1 = plt.figure(figsize=(10, 6))
contour = plt.contourf(XX, ZZ, Bnorm[:, j, :].T, levels=50, cmap='viridis')
plt.colorbar(contour, label='|B| [T]')

# Overlay the helical filaments with better colors
plt.plot(filament1[0, :], filament1[2, :], 'r--', linewidth=1, label='Helix 1')
plt.plot(filament2[0, :], filament2[2, :], 'cyan', linewidth=1, label='Helix 2')  # Use cyan for better visibility

# Set plot labels and title
plt.xlabel('X [m]')
plt.ylabel('Z [m]')
plt.title('Magnetic Field Magnitude [T] and Staggered Helical Undulator in the XZ-plane')

# Show legend once
plt.legend()

# Save and display the plot
plt.savefig('M1-figs/ex15_undulator_XZ_Bnorm_staggered.png', dpi=600)
plt.show()

# Create 3D figure
fig2 = plt.figure(figsize=(10, 6))
ax = fig2.add_subplot(111, projection='3d')

# Plot the first helix (red)
ax.plot(filament1[0, :], filament1[1, :], filament1[2, :], 'r-', label='Helix 1')

# Plot the second helix (cyan)
ax.plot(filament2[0, :], filament2[1, :], filament2[2, :], 'c-', label='Helix 2')

# Set axis labels
ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.set_zlabel('Z [m]')
ax.set_title('3D Visualization of Staggered Helical Coils')

# Set equal scaling for all axes
ax.set_box_aspect([1, 1, 1])  # Aspect ratio is 1:1:1
ax.set_aspect('auto')
ax.set_xlim([-0.1, 0.1])
ax.set_ylim([-0.1, 0.1])
ax.set_zlim([0, La])

# Add a legend
ax.legend()

# Enable rotation for interactive viewing
plt.savefig("M1-figs/ex15_undulator_3D_staggered.png", dpi=600)
plt.show()

# Create plot for magnetic field magnitude |B| in the XZ-plane
fig3 = plt.figure(figsize=(10, 6))
contour = plt.contourf(XX, ZZ, Bx[:, j, :].T, levels=50, cmap='viridis')
plt.colorbar(contour, label='|B| [T]')

# Overlay the helical filaments with better colors
plt.plot(filament1[0, :], filament1[2, :], 'r--', linewidth=1, label='Helix 1')
plt.plot(filament2[0, :], filament2[2, :], 'cyan', linewidth=1, label='Helix 2')  # Use cyan for better visibility

# Set plot labels and title
plt.xlabel('X [m]')
plt.ylabel('Z [m]')
plt.title('Bx [T] and Staggered Helical Undulator in the XZ-plane')

# Show legend once
plt.legend()

# Save and display the plot
# plt.savefig('M1-figs/ex15_undulator_XZ_Bnorm_staggered.png', dpi=600)
plt.show()

# Create plot for magnetic field magnitude |B| in the XZ-plane
fig4 = plt.figure(figsize=(10, 6))
contour = plt.contourf(XX, ZZ, By[:, j, :].T, levels=50, cmap='viridis')
plt.colorbar(contour, label='|B| [T]')

# Overlay the helical filaments with better colors
plt.plot(filament1[0, :], filament1[2, :], 'r--', linewidth=1, label='Helix 1')
plt.plot(filament2[0, :], filament2[2, :], 'cyan', linewidth=1, label='Helix 2')  # Use cyan for better visibility

# Set plot labels and title
plt.xlabel('X [m]')
plt.ylabel('Z [m]')
plt.title('By [T] and Staggered Helical Undulator in the XZ-plane')

# Show legend once
plt.legend()

# Save and display the plot
# plt.savefig('M1-figs/ex15_undulator_XZ_Bnorm_staggered.png', dpi=600)
plt.show()

# Create plot for magnetic field magnitude |B| in the XZ-plane
fig5 = plt.figure(figsize=(10, 6))
contour = plt.contourf(XX, ZZ, Bz[:, j, :].T, levels=50, cmap='viridis')
plt.colorbar(contour, label='|B| [T]')

# Overlay the helical filaments with better colors
plt.plot(filament1[0, :], filament1[2, :], 'r--', linewidth=1, label='Helix 1')
plt.plot(filament2[0, :], filament2[2, :], 'cyan', linewidth=1, label='Helix 2')  # Use cyan for better visibility

# Set plot labels and title
plt.xlabel('X [m]')
plt.ylabel('Z [m]')
plt.title('Bz [T] and Staggered Helical Undulator in the XZ-plane')

# Show legend once
plt.legend()

# Save and display the plot
# plt.savefig('M1-figs/ex15_undulator_XZ_Bnorm_staggered.png', dpi=600)
plt.show()
