import numpy as np
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt

def load_data(file_path):
  """Load the .npz file and extract components."""
  data = np.load(file_path)
  Bx = data['Bx']
  By = data['By']
  Bz = data['Bz']
  X = data['X']
  Y = data['Y']
  Z = data['Z']
  return Bx, By, Bz, X, Y, Z

def visualize_bnorm(Bx, By, Bz, X, Y, Z, Y_target=0.0):
  """Visualize the B-field magnitude in the XZ-plane."""
  j = np.abs(Y - Y_target).argmin()
  Bnorm = np.sqrt(Bx**2 + By**2 + Bz**2)
  fig, ax = plt.subplots(figsize=(10, 8))
  X_grid, Z_grid = np.meshgrid(X, Z)
  c = ax.contourf(X_grid, Z_grid, Bnorm[:, j, :].T, levels=30, cmap='viridis')
  ax.set_title('Bnorm')
  ax.set_xlabel('X')
  ax.set_ylabel('Z')
  fig.colorbar(c, ax=ax, orientation='vertical')
  plt.tight_layout()
  plt.savefig("M1-figs/npz_Bnorm.png")
  plt.show()

def interpolate_bfield(Bx, By, Bz, X, Y, Z, point):
  """Interpolate the B-field components at a given point."""
  interpolator_Bx = RegularGridInterpolator((X, Y, Z), Bx, bounds_error=False, fill_value=0)
  interpolator_By = RegularGridInterpolator((X, Y, Z), By, bounds_error=False, fill_value=0)
  interpolator_Bz = RegularGridInterpolator((X, Y, Z), Bz, bounds_error=False, fill_value=0)

  Bx_interp = interpolator_Bx(point)
  By_interp = interpolator_By(point)
  Bz_interp = interpolator_Bz(point)

  return Bx_interp, By_interp, Bz_interp


def main(file_path):
  """Main function to load data and visualize Bnorm."""
  Bx, By, Bz, X, Y, Z = load_data(file_path)
  visualize_bnorm(Bx, By, Bz, X, Y, Z)

  # Demonstrate interpolation at three random points
  points = [
      (np.random.uniform(X.min(), X.max()), np.random.uniform(Y.min(), Y.max()), np.random.uniform(Z.min(), Z.max())),
      (np.random.uniform(X.min(), X.max()), np.random.uniform(Y.min(), Y.max()), np.random.uniform(Z.min(), Z.max())),
      (np.random.uniform(X.min(), X.max()), np.random.uniform(Y.min(), Y.max()), np.random.uniform(Z.min(), Z.max()))
  ]

  for point in points:
    Bx_interp, By_interp, Bz_interp = interpolate_bfield(Bx, By, Bz, X, Y, Z, point)
    print(f"Interpolated B-field at ({point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f}):")
    print(f"  Bx = {Bx_interp:.3f}, By = {By_interp:.3f}, Bz = {Bz_interp:.3f}")

    # Find the indices of the nearest neighbors
    idx_x = np.searchsorted(X, point[0]) - 1
    idx_y = np.searchsorted(Y, point[1]) - 1
    idx_z = np.searchsorted(Z, point[2]) - 1

    # Ensure indices are within bounds
    idx_x = np.clip(idx_x, 0, len(X) - 2)
    idx_y = np.clip(idx_y, 0, len(Y) - 2)
    idx_z = np.clip(idx_z, 0, len(Z) - 2)

    neighbors = [
        (X[idx_x], Y[idx_y], Z[idx_z]),
        (X[idx_x + 1], Y[idx_y], Z[idx_z]),
        (X[idx_x], Y[idx_y + 1], Z[idx_z]),
        (X[idx_x], Y[idx_y], Z[idx_z + 1]),
        (X[idx_x + 1], Y[idx_y + 1], Z[idx_z]),
        (X[idx_x + 1], Y[idx_y], Z[idx_z + 1]),
        (X[idx_x], Y[idx_y + 1], Z[idx_z + 1]),
        (X[idx_x + 1], Y[idx_y + 1], Z[idx_z + 1])
    ]

    print("Nearest neighbors:")
    for neighbor in neighbors:
        Bx_neighbor = Bx[np.searchsorted(X, neighbor[0]) - 1, np.searchsorted(Y, neighbor[1]) - 1, np.searchsorted(Z, neighbor[2]) - 1]
        By_neighbor = By[np.searchsorted(X, neighbor[0]) - 1, np.searchsorted(Y, neighbor[1]) - 1, np.searchsorted(Z, neighbor[2]) - 1]
        Bz_neighbor = Bz[np.searchsorted(X, neighbor[0]) - 1, np.searchsorted(Y, neighbor[1]) - 1, np.searchsorted(Z, neighbor[2]) - 1]
        print(f"  Neighbor at ({neighbor[0]:.3f}, {neighbor[1]:.3f}, {neighbor[2]:.3f}):")
        print(f"    Bx = {Bx_neighbor:.3f}, By = {By_neighbor:.3f}, Bz = {Bz_neighbor:.3f}")

# Example usage
file_path = 'M1-data/magnetic_field_components.npz'
main(file_path)
