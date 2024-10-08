import numpy as np
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(1, '../../../ode/python/')
import ode

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

    Bx_interp = interpolator_Bx(point).item()
    By_interp = interpolator_By(point).item()
    Bz_interp = interpolator_Bz(point).item()

    return Bx_interp, By_interp, Bz_interp

def Efield(point):
    """
    Zero E-field
    """
    Ex = 0.0
    Ey = 0.0
    Ez = 0.0
    return Ex, Ey, Ez

# Interpolate B-field function
def interpolate_bfield_wrapper(point):
    global Bx, By, Bz, X, Y, Z
    return interpolate_bfield(Bx, By, Bz, X, Y, Z, point)

def MeV_to_mps(En):
    """
    Convert MeV to m/s
    En = 1/2 m v^2
    En [J] = En [MeV] * 1e6 [eV] * 1.6e-19 [J/eV]
    """
    En = En * 1e6 * 1.6e-19
    v = np.sqrt(2 * En / 9.11e-31)
    return v

def time_to_edge(v, La):
    """
    Time to reach edge of undulator
    """
    return La / v

def fun(t, X):
    if len(X) != 6:
        raise ValueError("Input array X must have exactly 6 elements.")
    x, y, z, vx, vy, vz = X
    point = (x, y, z)
    # E-field [V/m]
    Ex, Ey, Ez = Efield(point)
    # B-field [T]
    Bx, By, Bz = interpolate_bfield_wrapper(point)
    qm = -1.6e-19 / 9.11e-31
    Xdot = np.zeros(6)
    Xdot[0] = vx
    Xdot[1] = vy
    Xdot[2] = vz
    Xdot[3] = qm * (Ex + vy * Bz - vz * By)
    Xdot[4] = qm * (Ey + vz * Bx - vx * Bz)
    Xdot[5] = qm * (Ez + vx * By - vy * Bx)
    return Xdot

def main():
    """
    Use Runge-Kutta method to simulate the motion of an electron in an undulator field.
    """

    # Set initial conditions
    x0 = 0.0
    y0 = 0.0
    z0 = 0.0
    vx0 = 0.0
    vy0 = 0.0
    vz0 = MeV_to_mps(40)  # 40 MeV
    X0 = np.array([x0, y0, z0, vx0, vy0, vz0])

    # Set timesteps
    time = np.arange(0.0, time_to_edge(vz0, 0.5), 1.0e-12)

    # Load the magnetic field data
    global Bx, By, Bz, X, Y, Z
    Bx, By, Bz, X, Y, Z = load_data('../../../bfield/python/M1-data/magnetic_field_components.npz')

    # Simulate the trajectory using the Runge-Kutta method
    trajectory = ode.rk4(fun, time, X0)


    # Visualize the B-field magnitude in the XZ-plane with the trajectory
    i = np.abs(X - 0.0).argmin()  # Assuming X_target is 0.0 as in visualize_bnorm
    j = np.abs(Y - 0.0).argmin()  # Assuming Y_target is 0.0 as in visualize_bnorm



    Bnorm = np.sqrt(Bx**2 + By**2 + Bz**2)
    fig1, ax = plt.subplots(figsize=(10, 8))
    X_grid, Z_grid = np.meshgrid(X, Z)
    c = ax.contourf(X_grid, Z_grid, Bnorm[:, j, :].T, levels=30, cmap='viridis')
    ax.plot(trajectory[:, 0], trajectory[:, 2], color='white', linewidth=2)  # Plot trajectory
    ax.set_title('Electron Trajectory in Undulator - Runge-Kutta Method')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Z [m]')
    fig1.colorbar(c, ax=ax, orientation='vertical')
    plt.tight_layout()
    plt.savefig('M1-figs/undulator_electron_runge_kutta_with_bfield_XZ.png', dpi=300)
    plt.show()

    fig2, ax = plt.subplots(figsize=(10, 8))
    Y_grid, Z_grid = np.meshgrid(Y, Z)
    c = ax.contourf(Y_grid, Z_grid, Bnorm[i, :, :].T, levels=30, cmap='viridis')
    ax.plot(trajectory[:, 1], trajectory[:, 2], color='white', linewidth=2)  # Plot trajectory
    ax.set_title('Electron Trajectory in Undulator - Runge-Kutta Method')
    ax.set_xlabel('Y [m]')
    ax.set_ylabel('Z [m]')
    fig2.colorbar(c, ax=ax, orientation='vertical')
    plt.tight_layout()
    plt.savefig('M1-figs/undulator_electron_runge_kutta_with_bfield_YZ.png', dpi=300)
    plt.show()



if __name__ == '__main__':
    main()


# def main(file_path):
#   """Main function to load data and visualize Bnorm."""
#   Bx, By, Bz, X, Y, Z = load_data(file_path)
#   visualize_bnorm(Bx, By, Bz, X, Y, Z)

#   # Demonstrate interpolation at three random points
#   points = [
#       (np.random.uniform(X.min(), X.max()), np.random.uniform(Y.min(), Y.max()), np.random.uniform(Z.min(), Z.max())),
#       (np.random.uniform(X.min(), X.max()), np.random.uniform(Y.min(), Y.max()), np.random.uniform(Z.min(), Z.max())),
#       (np.random.uniform(X.min(), X.max()), np.random.uniform(Y.min(), Y.max()), np.random.uniform(Z.min(), Z.max()))
#   ]

#   for point in points:
#     Bx_interp, By_interp, Bz_interp = interpolate_bfield(Bx, By, Bz, X, Y, Z, point)
#     print(f"Interpolated B-field at ({point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f}):")
#     print(f"  Bx = {Bx_interp:.3f}, By = {By_interp:.3f}, Bz = {Bz_interp:.3f}")

#     # Find the indices of the nearest neighbors
#     idx_x = np.searchsorted(X, point[0]) - 1
#     idx_y = np.searchsorted(Y, point[1]) - 1
#     idx_z = np.searchsorted(Z, point[2]) - 1

#     # Ensure indices are within bounds
#     idx_x = np.clip(idx_x, 0, len(X) - 2)
#     idx_y = np.clip(idx_y, 0, len(Y) - 2)
#     idx_z = np.clip(idx_z, 0, len(Z) - 2)

#     neighbors = [
#         (X[idx_x], Y[idx_y], Z[idx_z]),
#         (X[idx_x + 1], Y[idx_y], Z[idx_z]),
#         (X[idx_x], Y[idx_y + 1], Z[idx_z]),
#         (X[idx_x], Y[idx_y], Z[idx_z + 1]),
#         (X[idx_x + 1], Y[idx_y + 1], Z[idx_z]),
#         (X[idx_x + 1], Y[idx_y], Z[idx_z + 1]),
#         (X[idx_x], Y[idx_y + 1], Z[idx_z + 1]),
#         (X[idx_x + 1], Y[idx_y + 1], Z[idx_z + 1])
#     ]

#     print("Nearest neighbors:")
#     for neighbor in neighbors:
#         Bx_neighbor = Bx[np.searchsorted(X, neighbor[0]) - 1, np.searchsorted(Y, neighbor[1]) - 1, np.searchsorted(Z, neighbor[2]) - 1]
#         By_neighbor = By[np.searchsorted(X, neighbor[0]) - 1, np.searchsorted(Y, neighbor[1]) - 1, np.searchsorted(Z, neighbor[2]) - 1]
#         Bz_neighbor = Bz[np.searchsorted(X, neighbor[0]) - 1, np.searchsorted(Y, neighbor[1]) - 1, np.searchsorted(Z, neighbor[2]) - 1]
#         print(f"  Neighbor at ({neighbor[0]:.3f}, {neighbor[1]:.3f}, {neighbor[2]:.3f}):")
#         print(f"    Bx = {Bx_neighbor:.3f}, By = {By_neighbor:.3f}, Bz = {Bz_neighbor:.3f}")

# # Example usage
# """
# Jumps up from M1 to Gh2,
# from Gh2 to python,
# from python to particle,
# from particle to home,
# from home to bfield,
# from bfield to python,
# from bfield to M1-data,
# where .npz file is located.
# """
# file_path = '../../../../bfield/python/M1-data/magnetic_field_components.npz'
# main(file_path)
