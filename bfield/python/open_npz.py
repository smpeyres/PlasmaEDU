import numpy as np
import matplotlib.pyplot as plt

def visualize_magnetic_field_components(file_path):
  # Load the .npz file
  data = np.load(file_path)

  # Extract the components
  Bx = data['Bx']
  By = data['By']
  Bz = data['Bz']
  X = data['X']
  Y = data['Y']
  Z = data['Z']

  # Plot the B-field magnitude in the XZ-plane
  Y_target = 0.0
  j = np.abs(Y - Y_target).argmin()

  # Calculate Bnorm
  Bnorm = np.sqrt(Bx**2 + By**2 + Bz**2)

  # Create subplots
  fig, ax = plt.subplots(figsize=(10, 8))

  # Create a meshgrid for X and Z
  X_grid, Z_grid = np.meshgrid(X, Z)

  # Plot Bnorm as a contour plot with 30 levels
  c = ax.contourf(X_grid, Z_grid, Bnorm[:, j, :].T, levels=30, cmap='viridis')
  ax.set_title('Bnorm')
  ax.set_xlabel('X')
  ax.set_ylabel('Z')

  # Add colorbar
  fig.colorbar(c, ax=ax, orientation='vertical')

  # Adjust layout
  plt.tight_layout()
  plt.savefig("M1-figs/npz_Bnorm.png")
  plt.show()

# Example usage
file_path = 'M1-data/magnetic_field_components.npz'
visualize_magnetic_field_components(file_path)
