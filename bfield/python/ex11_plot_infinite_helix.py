import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import bfield 

# Set up parameters for the helical coil - following Hagel et al
Ra = 1.0e-3   # Radius of each loop [m] - 1 mm
La = 2.0e-2  # Axial separation between turns [m] - 2 cm
I0 = 1.0  # Current in the wire [A] 
phi0 = 0.0

# z-values for figures 2 thru 4
z_vals = np.linspace(0.0, 6.0e-2, 101)
B_234 = bfield.inf_helix(Ra, La, I0, phi0, x = 1.0e-2, y = 0.0, z = z_vals)

fig2 = plt.figure()
plt.plot(z_vals/1.0e-2, B_234[0]/1.0e-6)
plt.xlabel("z [cm]")
plt.ylabel("Br [uT]")
plt.xlim(0, 6)
plt.ylim(-1,1)
plt.draw()
plt.savefig("Fig2_Hagel.png",dpi=600)

fig3 = plt.figure()
plt.plot(z_vals/1.0e-2, B_234[1]/1.0e-6)
plt.xlabel("z [cm]")
plt.ylabel("Bphi [uT]")
plt.xlim(0, 6)
plt.ylim(19.60, 20.40)
plt.draw()
plt.savefig("Fig3_Hagel.png",dpi=600)

fig4 = plt.figure()
plt.plot(z_vals/1.0e-2, B_234[2]/1.0e-6)
plt.xlabel("z [cm]")
plt.ylabel("Bz [uT]")
plt.xlim(0, 6)
plt.ylim(-1,1)
plt.draw()
plt.savefig("Fig4_Hagel.png",dpi=600)

# r-values for figures 5 thru 7
x_vals = np.linspace(1.0e-3, 6.0e-2, 101)
# never go to r = 0
B_567 = bfield.inf_helix(Ra, La, I0, phi0, x_vals, y = 0.0, z = 1.5e-2)

fig5 = plt.figure()
plt.plot(x_vals/1.0e-2, B_567[0]/1.0e-6)
plt.xlabel("r [cm]")
plt.ylabel("Br [uT]")
plt.xlim(0, 6)
plt.ylim(0, 0.8)
plt.draw()
plt.savefig("Fig5_Hagel.png",dpi=600)

fig6 = plt.figure()
plt.plot(x_vals/1.0e-2, B_567[1]/1.0e-6)
plt.xlabel("r [cm]")
plt.ylabel("Bphi [uT]")
plt.xlim(0, 6)
plt.ylim(0, 20)
plt.draw()
plt.savefig("Fig6_Hagel.png",dpi=600)

fig7 = plt.figure()
plt.plot(x_vals/1.0e-2, B_567[2]/1.0e-9)
plt.xlabel("r [cm]")
plt.ylabel("Bz [nT]")
plt.xlim(0, 6)
plt.ylim(0, 15)
plt.draw()
plt.savefig("Fig7_Hagel.png",dpi=600)