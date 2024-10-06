import numpy as np
import bfield
import matplotlib.pyplot as plt
import scipy.special as special

# I = current
# h = pitch
# R = radius
def transverse(I, h, R):
  mu0 = 4 * np.pi * 1e-7
  return (2*mu0*I/h)*((2*np.pi*R/h)*special.kv(0, 2*np.pi*R/h) + special.kv(1, 2*np.pi*R/h))

def axial(I, h):
  mu0 = 4 * np.pi * 1e-7
  return (mu0/h)*(I-I)

# Helical Solenoid Parameters
I0 = 1000.0
Ra = 0.05
La = 0.50
Nturns = np.array([1, 2, 4, 8, 16, 32]) # number of turns

h_num = La/Nturns

h_vals = np.linspace(0, np.max(La/Nturns), 51)

phi0 = 0.0
phase_shift = np.pi
Center1 = np.array([0, 0, 0.0])

EulerAngles1 = np.array([0, 0, 0]) * np.pi / 180.0
EulerAngles2 = np.array([0, 0, 0]) * np.pi / 180.0

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
  Bx, By, Bz, filament1, filament2 = bfield.undulator(X_grid, Y_grid, Z_grid, I0, Ra, La, N, 50*N, phi0, phase_shift, Center1, np.array([0, 0, 0.5*La/N]), EulerAngles1, EulerAngles2)
  Bx_res[f"Bx_{N}"] = Bx
  By_res[f"By_{N}"] = By
  Bz_res[f"Bz_{N}"] = Bz
  Bx_mean.append(np.mean(Bx))
  Bx_max.append(np.max(Bx))
  By_mean.append(np.mean(By))
  By_max.append(np.max(By))
  Bz_mean.append(np.mean(Bz))
  Bz_max.append(np.max(Bz))


fig1 = plt.figure()
plt.plot(h_vals, transverse(I0, h_vals, Ra), color='k', label="Eq. (3a)")
plt.scatter(h_num, Bx_max, zorder=2, color='b', label="max($B_x$)")
plt.scatter(h_num, By_max, zorder=4, color='orange', label="max($B_y$)")
plt.legend()
plt.xlabel("Pitch $h$ [m]")
plt.ylabel("$B_0$ [T]")
plt.savefig("M1-figs/transverse_verification.png")
plt.show()

fig2 = plt.figure()
plt.plot(Z_grid, np.squeeze(Bx_res["Bx_1"]), color='b', label="$B_x$, N = 1")
plt.plot(Z_grid, np.squeeze(By_res["By_1"]), color='r', label="$B_y$, N = 1")
plt.plot(Z_grid, np.squeeze(Bz_res["Bz_1"]), color='g', label="$B_z$, N = 1")
plt.plot(Z_grid, np.squeeze(Bx_res["Bx_8"]), color='magenta', label="$B_x$, N = 8")
plt.plot(Z_grid, np.squeeze(By_res["By_8"]), color='orange', label="$B_y$, N = 8")
plt.plot(Z_grid, np.squeeze(Bz_res["Bz_8"]), color='cyan', label="$B_z$, N = 8")
plt.ylabel("$B_i$ [T]")
plt.xlabel("Z [m]")
plt.legend()
plt.savefig("M1-figs/undulator_components.png")
plt.show()
