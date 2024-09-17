import sys
import numpy as np
import matplotlib.pyplot as plt

# this is the field section
# this is where the magnetic fields and electric fields are defined
# the magnetic field in this example is torodial + polodial
# with the polodial field constructed from a provided flux function

# notation for functions:
# r, z denote point input
# R, Z denote vector input
# likewise for cartesian

# structure of this section:
# 1. definition of point input field generators for particle dynamics
# 2. definition of vector input field generators for visualization
# 3. visualization of fields

# provided flux function
# point-by-point input; does not accept vector/array inputs
def psi_iter_like(r,z):
  R0   = 6.2
  A    = -0.155
  Psi0 = 202.92

  # Normalized coordinates w.r.t. major radius
  x = r/R0
  y = z/R0

  # Powers of x and y and log
  x2 = x*x
  x4 = x2*x2
  y2 = y*y
  y4 = y2*y2
  lnx = np.log(x)

  # Single-null Grad Shafranov functions
  psi_i = np.zeros(12)
  coeff = np.zeros(12)

  psi_i[0] = 1.0
  psi_i[1] = x2
  psi_i[2] = y2 - x2*lnx
  psi_i[3] = x4 - 4.0*x2*y2
  psi_i[4] = 2.0*y4 - 9.0*y2*x2 + 3.0*x4*lnx - 12.0*x2*y2*lnx
  psi_i[5] = x4*x2 - 12.0*x4*y2 + 8.0*x2*y4
  psi_i[6] = 8.0*y4*y2 - 140.0*y4*x2 +75.0*y2*x4 -15.0*x4*x2*lnx + 180.0*x4*y2*lnx - 120.0*x2*y4*lnx
  psi_i[7] = y
  psi_i[8] = y*x2
  psi_i[9] = y*y2 - 3.0*y*x2*lnx
  psi_i[10] = 3.0*y*x4 - 4.0*x2*y2*y
  psi_i[11] = 8.0*y4*y - 45.0*y*x4 - 80.0*y2*y*x2*lnx + 60.0*y*x4*lnx

  # Coefficients for ITER-like magnetic equilibrium
  coeff[0] =  1.00687012e-1
  coeff[1] =  4.16274456e-1
  coeff[2] = -6.53880989e-1
  coeff[3] = -2.95392244e-1
  coeff[4] =  4.40037966e-1
  coeff[5] = -4.01807386e-1
  coeff[6] = -1.66351381e-2
  coeff[7] =  1.92944621e-1
  coeff[8] =  8.36039453e-1
  coeff[9] = -5.30670714e-1
  coeff[10]= -1.26671504e-1
  coeff[11]=  1.47140977e-2

  psi = np.dot(coeff, psi_i) + x4/8.0 + A * (0.5*x2*lnx - x4/8.0)

  Psi = Psi0*psi

  return Psi

# generate polodial field in radial coordinates
# single-point definition
def B_pol_rad(r,z):
  # major radius [m]
  R0 = 6.2
  # I can define the step size as big or as small as I want, but let's go with something simple
  h = R0/1000

  # radial component: simple midpoint
  Br_pol = (1/r)*(psi_iter_like(r,z+h) - psi_iter_like(r,z-h))/(2*h)

  # radial component: simple midpoint
  Bz_pol = -(1/r)*(psi_iter_like(r+h,z) - psi_iter_like(r-h,z))/(2*h)

  return Br_pol, Bz_pol

# generate torodial field in radial coordinates
# single-point definition
def B_tor_rad(r,z):
  # on-axis field [T]
  B0 = 5.3

  # major radius [m]
  R0 = 6.2

  # phi component
  Bphi_tor = B0*R0/r

  # z component
  Bz_tor = 0.0

  return Bphi_tor, Bz_tor

# total magnetic field in radial
def Bfield_rad(r,z):
  B_pol = B_pol_rad(r,z)
  B_tor = B_tor_rad(r,z)

  Br = B_pol[0]
  Bphi = B_tor[0]
  Bz = B_pol[1] + B_tor[1]

  return Br, Bphi, Bz

# convert polodial field to cartesian components
def B_pol_cart(x,y,z):
  # Note that \hat{x} = cos(phi) \hat{r} - sin(phi) \hat{phi}
  # and \hat{y} = sin(phi) \hat{r} + cos(phi) \hat{phi}
  # First term for polodial, second term for torodial

  # projection
  r = np.sqrt(x*x + y*y)

  # cosine and sine
  ca = x/r
  sa = y/r

  B_pol = B_pol_rad(r,z)

  Bx_pol = B_pol[0]*ca
  By_pol = B_pol[0]*sa
  Bz_pol = B_pol[1]

  return Bx_pol, By_pol, Bz_pol

# convert torodial field to cartesian components
def B_tor_cart(x,y,z):
  # Note that \hat{x} = cos(phi) \hat{r} - sin(phi) \hat{phi}
  # and \hat{y} = sin(phi) \hat{r} + cos(phi) \hat{phi}
  # First term for polodial, second term for torodial

  # projection
  r = np.sqrt(x*x + y*y)

  # cosine and sine
  ca = x/r
  sa = y/r

  B_tor = B_tor_rad(r,z)

  Bx_tor = -1*B_tor[0]*sa
  By_tor =    B_tor[0]*ca
  Bz_tor =    B_tor[1]

  return Bx_tor, By_tor, Bz_tor

# total magnetic field in cartestian
def Bfield_cart(x,y,z):
  B_pol = B_pol_cart(x,y,z)
  B_tor = B_tor_cart(x,y,z)

  Bx = B_pol[0] + B_tor[0]
  By = B_pol[1] + B_tor[1]
  Bz = B_pol[2] + B_tor[2]

  return Bx, By, Bz

# electric field in cartestian
def Efield_cart(x,y,z):
  Ex = 0.0
  Ey = 0.0
  Ez = 0.0
  return Ex, Ey, Ez

# generalized visualization generator for radial coordinates/components
# adapted from response from custom GPT "Code Copilot" by promptspellsmith.com
def viz_rad(R, Z, parent_function, num_outputs):
    # grab number of points
    N_r = len(R)
    N_z = len(Z)

    # create empty matrices for outputs based on num_outputs
    if num_outputs == 1:
        result = np.zeros((N_r, N_z))
    else:
        result = [np.zeros((N_r, N_z)) for _ in range(num_outputs)]

    # loop over all points and evaluate parent_function at each point
    for i in range(N_r):
        for j in range(N_z):
            values = parent_function(R[i], Z[j])
            if num_outputs == 1:
                result[i, j] = values
            else:
                for k in range(num_outputs):
                    result[k][i, j] = values[k]

    return result if num_outputs == 1 else tuple(result)

# Wrapper for psi
def psi_viz(R, Z):
    return viz_rad(R, Z, psi_iter_like, 1)

# Wrapper for B_pol_rad
def B_pol_rad_viz(R, Z):
    return viz_rad(R, Z, B_pol_rad, 2)

# Wrapper for B_tor_rad
def B_tor_rad_viz(R, Z):
    return viz_rad(R, Z, B_tor_rad, 2)

# Wrapper for Bfield_rad
def Bfield_rad_viz(R, Z):
    return viz_rad(R, Z, Bfield_rad, 3)

# need to make one for cartestian, which might be harder

# visualization

# simple domain for testing visualization
R0   = 6.2
R_vals = np.linspace(  0.6*R0, 1.4*R0, 100 )
Z_vals = np.linspace( -0.8*R0, 0.8*R0, 100 )

# psi - flux function
plt.figure(1)
RR,ZZ = np.meshgrid(R_vals/R0,Z_vals/R0)
plt.contourf(np.transpose(RR),np.transpose(ZZ),psi_viz(R_vals,Z_vals),30,cmap="RdBu_r")
plt.colorbar()
plt.xlabel('R/R0')
plt.ylabel('Z/R0')
plt.title('ITER Polodial Flux Function [Tm${}^2$]')
plt.savefig('iter_flux_function_wrapped.png')

# B_pol_rad - radial component
plt.figure(2)
RR,ZZ = np.meshgrid(R_vals/R0,Z_vals/R0)
plt.contourf(np.transpose(RR),np.transpose(ZZ),B_pol_rad_viz(R_vals,Z_vals)[0],30,cmap="RdBu_r")
plt.colorbar()
plt.xlabel('R/R0')
plt.ylabel('Z/R0')
plt.title('Radial Component of Polodial Field [T]')
plt.savefig('BR_polodial_iter_wrapped.png')

# B_pol_rad - axial component
plt.figure(3)
RR,ZZ = np.meshgrid(R_vals/R0,Z_vals/R0)
plt.contourf(np.transpose(RR),np.transpose(ZZ),B_pol_rad_viz(R_vals,Z_vals)[1],30,cmap="RdBu_r")
plt.colorbar()
plt.xlabel('R/R0')
plt.ylabel('Z/R0')
plt.title('Axial Component of Polodial Field [T]')
plt.savefig('BZ_polodial_iter_wrapped.png')

# B_pol_rad - magntitude

B_pol_mag_viz = np.sqrt(B_pol_rad_viz(R_vals,Z_vals)[0]**2 + B_pol_rad_viz(R_vals,Z_vals)[1]**2)

plt.figure(4)
RR,ZZ = np.meshgrid(R_vals/R0,Z_vals/R0)
plt.contour(np.transpose(RR),np.transpose(ZZ),B_pol_mag_viz,100,cmap="RdBu_r")
plt.colorbar()
plt.xlabel('R/R0')
plt.ylabel('Z/R0')
plt.title('Magnitude of Polodial Field [T]')
plt.savefig('Bmag_polodial_iter_wrapped.png')

# slightly limited range/domain

R_lim = np.linspace(  5, 8, 100 )
Z_lim = np.linspace( -4, 4, 100 )

B_pol_mag_lim = np.sqrt(B_pol_rad_viz(R_lim,Z_lim)[0]**2 + B_pol_rad_viz(R_lim,Z_lim)[1]**2)

plt.figure(5)
RR,ZZ = np.meshgrid(R_lim,Z_lim)
plt.contour(np.transpose(RR),np.transpose(ZZ),B_pol_mag_lim,100,cmap="RdBu_r")
plt.colorbar()
plt.xlabel('R [m]')
plt.ylabel('Z [m]')
plt.title('Magnitude of Polodial Field [T]')
plt.savefig('Bmag_polodial_iter_limited_wrapped.png')

# B_tor_rad - magnitude

B_tor_mag_lim = np.sqrt(B_tor_rad_viz(R_lim,Z_lim)[0]**2 + B_tor_rad_viz(R_lim,Z_lim)[1]**2)

plt.figure(6)
RR,ZZ = np.meshgrid(R_lim,Z_lim)
plt.contourf(np.transpose(RR),np.transpose(ZZ),B_tor_mag_lim,30,cmap="RdBu_r")
plt.colorbar()
plt.xlabel('R [m]')
plt.ylabel('Z [m]')
plt.title('Magnitude of Torodial Field [T]')
plt.savefig('Bmag_torodial_iter_limited_wrapped.png')

# This is the particle portion
# This is where the IVP of a particle placed within these fields is solved
# using different numerical schemes and their trajectories are plotted.

sys.path.insert(1, '../../ode/python/')
import ode

# physical constants
e =  1.60217662e-19 # elementary charge [C]
me =  9.10938356e-31 # electron mass [kg]
mDa = 1.66053906e-27 # unified atomic mass or Dalton [kg]
mp =  1.6726219e-27 # mass of proton [kg]
# kB =  1.38064852e-23 # Boltzmann constant [J/K]

A = 2.0141 # atomic mass of deuterium
Mi = A*mDa # mass of ion [kg]
Z = 1 # charge of ion

# Charge-to-mass ratio (q/m)
qm = Z*e/Mi

# RHS of the ODE problem, dy/dx = f(x,y)
def fun(t,X):
  # "extracts" positions and velocities from provided state function
  x, y, z, vx, vy, vz = X
  # calculates E-field [V/m] at given position from pre-defined function
  Ex, Ey, Ez = Efield_cart(x,y,z)
  # calculates B-field [T] at given position from pre-defined function
  Bx, By, Bz = Bfield_cart(x,y,z)
  # Newton-Lorentz equation in Cartesian coordinates
  Xdot = np.zeros(6)
  Xdot[0] = vx
  Xdot[1] = vy
  Xdot[2] = vz
  Xdot[3] = qm * ( Ex + vy*Bz - vz*By )
  Xdot[4] = qm * ( Ey + vz*Bx - vx*Bz )
  Xdot[5] = qm * ( Ez + vx*By - vy*Bx )
  return Xdot

def main():

  # major radius [m]
  R0 = 6.2

  # plasma temperature [keV]
  T = 10.0

  # most propable speed for Maxwellian [m/s] with keV input
  v_p = np.sqrt(2.0*e*1000.0*T/Mi)

  # initial positions [m]
  x0 = R0
  y0 = 0.0 # radially symmetric, so just picking the x-axis
  z0 = 0.0

  # Initial velocities [m/s]
  vx0 = 0.0
  vy0 = v_p/np.sqrt(2)
  vz0 = v_p/np.sqrt(2)

  # Initial conditions - same for all solves
  X0 = np.array( [ x0, y0, z0, vx0, vy0, vz0 ] )

  # some characteristic B-field
  B_char = 5.3 # guess for now
  # if it's too low,
  # I think the integration time is too long and the Euler solution explodes

  # Larmor pulsation [rad/s]
  w_L = np.abs(qm * B_char)

  # Larmor period [s]
  tau_L = 2.0*np.pi / w_L

  # Larmor radius [m]
  r_L = v_p / w_L

  # Number of Larmor gyrations - part 1 and part 2
  N_gyro_1 = 4
  N_gyro_2 = 4500

  # Number of steps per gyroperiod - same for both parts
  N_points_per_gyration = 30

  # Time grid - part 1 and part 2 [s]
  time_1 = np.linspace( 0.0, tau_L*N_gyro_1, N_gyro_1*N_points_per_gyration )
  time_2 = np.linspace( 0.0, tau_L*N_gyro_2, N_gyro_2*N_points_per_gyration )

  # Solve ODE - part 1 solves
  X_euler = ode.euler(    fun, time_1, X0 )
  X_mid   = ode.midpoint( fun, time_1, X0 )
  X_rk4_1 = ode.rk4(      fun, time_1, X0 )

  # Solve ODE - part 2 solves
  X_rk4_2 = ode.rk4( fun, time_2, X0 )

  # Components of state vector for each solve
  x_euler  = X_euler[:,0]
  y_euler  = X_euler[:,1]
  z_euler  = X_euler[:,2]
  vx_euler = X_euler[:,3]
  vy_euler = X_euler[:,4]
  vz_euler = X_euler[:,5]

  x_mid  = X_mid[:,0]
  y_mid  = X_mid[:,1]
  z_mid  = X_mid[:,2]
  vx_mid = X_mid[:,3]
  vy_mid = X_mid[:,4]
  vz_mid = X_mid[:,5]

  x_rk4_1  = X_rk4_1[:,0]
  y_rk4_1  = X_rk4_1[:,1]
  z_rk4_1  = X_rk4_1[:,2]
  vx_rk4_1 = X_rk4_1[:,3]
  vy_rk4_1 = X_rk4_1[:,4]
  vz_rk4_1 = X_rk4_1[:,5]

  x_rk4_2  = X_rk4_2[:,0]
  y_rk4_2  = X_rk4_2[:,1]
  z_rk4_2  = X_rk4_2[:,2]
  vx_rk4_2 = X_rk4_2[:,3]
  vy_rk4_2 = X_rk4_2[:,4]
  vz_rk4_2 = X_rk4_2[:,5]

  # radial projection for each solve
  R_euler = np.sqrt( x_euler*x_euler + y_euler*y_euler )
  R_mid   = np.sqrt( x_mid*x_mid + y_mid*y_mid )
  R_rk4_1 = np.sqrt( x_rk4_1*x_rk4_1 + y_rk4_1*y_rk4_1 )
  R_rk4_2 = np.sqrt( x_rk4_2*x_rk4_2 + y_rk4_2*y_rk4_2 )

  # let's move back to the field portion now

  plt.figure(7)
  plt.plot( x_euler, y_euler, 'b-', label="Euler")
  plt.plot( x_mid, y_mid, 'r-', label="Midpoint")
  plt.plot( x_rk4_1, y_rk4_1, 'g-', label="Runge-Kutta")
  plt.xlabel('x [m]')
  plt.ylabel('y [m]')
  plt.axis('equal')
  plt.legend(loc=3)
  plt.savefig('particle-iter-XY-1.png')

  plt.figure(8)
  plt.plot( R_euler, z_euler, 'b-', label="Euler")
  plt.plot( R_mid, z_mid, 'r-', label="Midpoint")
  plt.plot( R_rk4_1, z_rk4_1, 'g-', label="Runge-Kutta")
  plt.xlabel('R, Radius [m]')
  plt.ylabel('Z, Vertical Coordinate [m]')
  plt.axis('equal')
  plt.legend(loc=3)
  plt.savefig('particle-iter-RZ-1.png')

  plt.figure(9)
  plt.plot( R_rk4_2, z_rk4_2, 'k-')
  plt.axis('equal')
  plt.xlabel('R, Radius [m]')
  plt.ylabel('Z, Vertical Coordinate [m]')
  plt.savefig('particle-iter-RZ-2.png')

  plt.figure(10)
  plt.plot( x_rk4_2, y_rk4_2, 'k-')
  plt.xlabel('R, Radius [m]')
  plt.ylabel('Z, Vertical Coordinate [m]')
  plt.axis('equal')
  plt.xlabel('x [m]')
  plt.ylabel('y [m]')
  plt.savefig('particle-iter-XY-2.png')

  plt.figure(11)
  RR,ZZ = np.meshgrid(R_lim,Z_lim)
  plt.contour(np.transpose(RR),np.transpose(ZZ),psi_viz(R_lim,Z_lim),100,cmap="RdBu_r")
  plt.plot( R_rk4_2, z_rk4_2, 'k-')
  plt.xlim(5,8)
  plt.ylim(-2,2)
  # plt.axis('equal')
  plt.colorbar()
  plt.xlabel('R, Radius [m]')
  plt.ylabel('Z, Vertical Coordinate [m]')
  plt.savefig('particle-iter-RZ-field-wide.png')

  plt.figure(12)
  RR,ZZ = np.meshgrid(R_lim,Z_lim)
  plt.contour(np.transpose(RR),np.transpose(ZZ),psi_viz(R_lim,Z_lim),100,cmap="RdBu_r")
  plt.plot( R_rk4_2, z_rk4_2, 'k-')
  plt.xlim(6,7)
  plt.ylim(-1,1.5)
  # plt.axis('equal')
  plt.colorbar()
  plt.xlabel('R, Radius [m]')
  plt.ylabel('Z, Vertical Coordinate [m]')
  plt.savefig('particle-iter-RZ-field-zoomed.png')

  # plt.figure(13)
  # RR,ZZ = np.meshgrid(R_lim,Z_lim)
  # plt.contour(np.transpose(RR),np.transpose(ZZ),psi_viz(R_lim,Z_lim),100,cmap="RdBu_r")
  # plt.plot( R_rk4_2, z_rk4_2, 'k-')
  # plt.xlim(6.4,6.8)
  # plt.ylim(-0.4,0)
  # # plt.axis('equal')
  # plt.colorbar()
  # plt.xlabel('R, Radius [m]')
  # plt.ylabel('Z, Vertical Coordinate [m]')
  # plt.savefig('particle-iter-RZ-field-morezoomed.png')


if __name__ == '__main__':
   main()

