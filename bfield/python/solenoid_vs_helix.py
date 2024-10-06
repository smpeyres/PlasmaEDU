import numpy as np
import bfield
import matplotlib.pyplot as plt

# n = number of turns per unit length
# I = current
def analytical(n, I):
  mu0 = 4 * np.pi * 1e-7
  return mu0*n*I

# Helical Solenoid Parameters
I0 = 1000.0
Ra = 0.05
La = 0.50
Nturns = np.array([1, 2, 4, 8, 16, 32]) # number of turns

n_num = Nturns/La

phi0 = 0.0
Center = np.array([0, 0, 0.0])
EulerAngles = np.array([0, 0, 0]) * np.pi / 180.0

# Define the grid
X_grid = np.array([0.0])
Y_grid = np.array([0.0])
Z_grid = np.linspace(0.0, La, 50)

# create empty dictionaries
Bx_res = {}
By_res = {}
Bz_res = {}

# empty arrays to store the mean and max values
Bx_mean = []
Bx_max = []
By_mean = []
By_max = []
Bz_mean = []
Bz_max = []

for N in Nturns:
  # 50*N for 50 points per turn
  Bx, By, Bz, filament = bfield.helix(X_grid, Y_grid, Z_grid, I0, Ra, La, N, 50*N, phi0, Center, EulerAngles)
  Bx_res[f"Bx_{N}"] = Bx
  By_res[f"By_{N}"] = By
  Bz_res[f"Bz_{N}"] = Bz
  Bx_mean.append(np.mean(Bx))
  Bx_max.append(np.max(Bx))
  By_mean.append(np.mean(By))
  By_max.append(np.max(By))
  Bz_mean.append(np.mean(Bz))
  Bz_max.append(np.max(Bz))

n_vals = np.linspace(0, np.max(n_num), 51)

fig1 = plt.figure()
plt.plot(n_vals, analytical(n_vals, 1000.0), zorder=1, color='k', label="Eq. (2)")
plt.scatter(n_num, Bz_max, zorder=2, color='b', label="max($B_z$)")
plt.scatter(n_num, Bz_mean, zorder=3, color='r', label="$⟨ B_z ⟩ $")
plt.legend()
plt.ylabel("$B_z$ [T]")
plt.xlabel("$n$ [m${}^{-1}$]")
plt.savefig("M1-figs/solenoid_vs_helix.png", dpi=600)
plt.show()

fig2 = plt.figure()
plt.plot(Z_grid, np.squeeze(Bz_res["Bz_32"]), color='b', label="N = 32")
plt.plot(Z_grid, np.squeeze(Bz_res["Bz_8"]), color='r', label="N = 8")
plt.plot(Z_grid, np.squeeze(Bz_res["Bz_2"]), color='g', label="N = 2")
plt.ylabel("$B_z$ [T]")
plt.xlabel("Z [m]")
plt.legend()
plt.savefig("M1-figs/Bz_number_turns_axial.png", dpi=600)
plt.show()

print(Bx_max)
print(By_max)

print(Bx_mean)
print(By_mean)
