import numpy as np
import matplotlib.pyplot as plt

# n = number of turns per unit length
# I = current
def analytical(n, I):
  mu0 = 4 * np.pi * 1e-7
  return mu0*N*I/l





N_num = np.array([5, 10, 20, 40, 80])
B_num = np.array([0.00595, 0.0119, 0.0239, 0.0481, 0.09778])

N_vals = np.linspace(0, 100, 51)

fig = plt.figure()
plt.plot(N_vals, analytical(N_vals, 1000.0, 1.0), zorder=1, color='k',label="Analytical")
plt.scatter(N_num, B_num,zorder=2,color='r',label="Numerical")
plt.legend()
plt.xlabel("Number of Turns")
plt.ylabel("On-Axis $B_z$ [T]")
plt.savefig("theory_vs_num_solenoid_vs_helix.png",dpi=600)
plt.show()


