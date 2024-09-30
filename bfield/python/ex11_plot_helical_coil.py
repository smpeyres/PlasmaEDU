import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import bfield  # Assuming bfield.py now contains the new Biot-Savart-based helix function

# Set up parameters for the helical coil
Ra = 0.1   # Radius of each loop [m]
La = 0.05  # Axial separation between turns [m]
N = 10     # Number of turns in the helix
I0 = 100.0  # Current in the wire [A] (change this to 1.0 for 1A, 100.0 for 100A)
Nturns = 1  # Single loop per turn
Center = (0.0, 0.0, 0.0)  # Center of the first loop

# Function to calculate the magnetic field at a point using the new helix function
def calculate_field_at_point(Ra, La, N, I0, Point):
    """ Wrapper for the helix function to calculate the field at a specific point """
    return bfield.helix(Ra, La, N, I0, Point)

# --- Plot 1: 3D vector plot of magnetic field ---
# Set up a 3D grid of points where we want to calculate the magnetic field
X, Y, Z = np.meshgrid(
    np.linspace(-0.2, 0.2, 10),  # Reduced resolution to make visualization clearer
    np.linspace(-0.2, 0.2, 10),
    np.linspace(0, 0.5, 10)
)

# Initialize arrays for the magnetic field components and magnitudes
Bx_total = np.zeros(X.shape)
By_total = np.zeros(X.shape)
Bz_total = np.zeros(X.shape)
B_magnitude = np.zeros(X.shape)

# Loop through each point in the grid and compute the magnetic field using the new helix function
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        for k in range(X.shape[2]):
            Point = (X[i,j,k], Y[i,j,k], Z[i,j,k])
            Bx, By, Bz = calculate_field_at_point(Ra, La, N, I0, Point)
            Bx_total[i,j,k] = Bx
            By_total[i,j,k] = By
            Bz_total[i,j,k] = Bz
            # Calculate the magnitude of the magnetic field
            B_magnitude[i,j,k] = np.sqrt(Bx**2 + By**2 + Bz**2)

# --- Plot 1: 3D vector plot of magnetic field ---
fig1 = plt.figure(figsize=(10,8))
ax1 = fig1.add_subplot(111, projection='3d')

# Create the helical coil points (for visualizing the coil itself)
theta = np.linspace(0, 2 * np.pi * N, 100 * N)
x_helix = Ra * np.cos(theta)
y_helix = Ra * np.sin(theta)
z_helix = La * theta / (2 * np.pi)

# Plot the helical coil
ax1.plot(x_helix, y_helix, z_helix, label='Helical Coil', color='r')

# Plot the magnetic field vectors, colored by magnitude
norm = plt.Normalize(vmin=np.min(B_magnitude), vmax=np.max(B_magnitude))
colors = plt.cm.viridis(norm(B_magnitude))

# Reshape arrays for quiver plotting
Bx_flat = Bx_total.flatten()
By_flat = By_total.flatten()
Bz_flat = Bz_total.flatten()
X_flat = X.flatten()
Y_flat = Y.flatten()
Z_flat = Z.flatten()
colors_flat = colors.reshape(-1, 4)

# Plot the quiver with colors representing the magnitude of the magnetic field
ax1.quiver(X_flat, Y_flat, Z_flat, Bx_flat, By_flat, Bz_flat, length=0.01, normalize=True, color=colors_flat)

# Add a color bar to indicate the magnitude of the field
mappable = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
mappable.set_array(B_magnitude)
cbar1 = fig1.colorbar(mappable, ax=ax1, shrink=0.7, aspect=10)
cbar1.set_label("Magnetic Field Magnitude [T]")

# Set plot labels for 3D plot
ax1.set_title('3D Magnetic Field of Helical Coil')
ax1.set_xlabel('X [m]')
ax1.set_ylabel('Y [m]')
ax1.set_zlabel('Z [m]')
ax1.legend()

# --- Plot 2: 2D heatmap of magnetic field magnitude in XZ-plane at Y=0 ---

# Set up a grid for the XZ-plane at Y = 0
x_vals = np.linspace(-0.2, 0.2, 40)
z_vals = np.linspace(0, 0.5, 40)
X_2D, Z_2D = np.meshgrid(x_vals, z_vals)
Y_2D = np.zeros_like(X_2D)  # Y is fixed at 0

# Initialize arrays for magnetic field components in the XZ plane
Bx_2D = np.zeros(X_2D.shape)
Bz_2D = np.zeros(X_2D.shape)
B_magnitude_2D = np.zeros(X_2D.shape)

# Calculate the magnetic field in the XZ plane
for i in range(X_2D.shape[0]):
    for j in range(X_2D.shape[1]):
        Point = (X_2D[i, j], Y_2D[i, j], Z_2D[i, j])
        Bx, By, Bz = calculate_field_at_point(Ra, La, N, I0, Point)
        Bx_2D[i, j] = Bx
        Bz_2D[i, j] = Bz
        B_magnitude_2D[i, j] = np.sqrt(Bx**2 + By**2 + Bz**2)

# --- Plot 3: Magnetic Field Magnitude in the XZ-plane at Y=0 ---
fig2, ax2 = plt.subplots(figsize=(8, 6))
heatmap_magnitude = ax2.pcolormesh(X_2D, Z_2D, B_magnitude_2D, shading='auto', cmap='viridis', norm=plt.Normalize(vmin=np.min(B_magnitude_2D), vmax=np.max(B_magnitude_2D)))
cbar2 = fig2.colorbar(heatmap_magnitude)
cbar2.set_label("Magnetic Field Magnitude [T]")

# Set plot labels for magnitude heatmap
ax2.set_title('Magnetic Field Magnitude in XZ-plane at Y=0')
ax2.set_xlabel('X [m]')
ax2.set_ylabel('Z [m]')

# --- Plot 4: 2D heatmap for Bz in the XZ-plane at Y=0 ---
fig3, ax3 = plt.subplots(figsize=(8, 6))
heatmap_bz = ax3.pcolormesh(X_2D, Z_2D, Bz_2D, shading='auto', cmap='plasma', norm=plt.Normalize(vmin=np.min(Bz_2D), vmax=np.max(Bz_2D)))
cbar3 = fig3.colorbar(heatmap_bz)
cbar3.set_label("Bz [T]")

# Set plot labels for Bz heatmap
ax3.set_title('Z-component of Magnetic Field (Bz) in XZ-plane at Y=0')
ax3.set_xlabel('X [m]')
ax3.set_ylabel('Z [m]')

# --- Plot 5: 2D heatmap for Bx in the XZ-plane at Y=0 ---
fig4, ax4 = plt.subplots(figsize=(8, 6))
heatmap_bx = ax4.pcolormesh(X_2D, Z_2D, Bx_2D, shading='auto', cmap='coolwarm', norm=plt.Normalize(vmin=np.min(Bx_2D), vmax=np.max(Bx_2D)))
cbar4 = fig4.colorbar(heatmap_bx)
cbar4.set_label("Bx [T]")

# Set plot labels for Bx heatmap
ax4.set_title('X-component of Magnetic Field (Bx) in XZ-plane at Y=0')
ax4.set_xlabel('X [m]')
ax4.set_ylabel('Z [m]')

plt.tight_layout()
plt.show()
