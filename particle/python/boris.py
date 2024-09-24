import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(1, '../../ode/python/')
import ode

# physical constants
qe = 1.60217662e-19 # C
me = 9.10938356e-31 # kg
mp = 1.6726219e-27 # kg

# define static, uniform E-field
def Efield(x,y,z):
  Ex = 0.0
  Ey = 0.0
  Ez = 0.0
  return Ex, Ey, Ez

# define static, uniform B-field
def Bfield(x,y,z):
  Bx = 0.0
  By = 0.0
  Bz = 0.1
  return Bx, By, Bz

# if user wants frequency correction, calculate and return it
# see slide 17 of lesson 06
# generic input for x, but should be Omega(x_n)*dt/2
def frequency_correction(x, apply_correction=True):
  if x == 0.0:
    alpha = 1.0
  elif apply_correction:
    alpha = np.tan(x)/x
  else:
    alpha = 1.0
  return alpha

# Boris-Bunemann algorithm (Hockney-Bunemann at the very least)
# takes time array, initial values array, and parameter array
def boris_bunemann( time, X0, params ):
  # time = np.linspace( 0.0, 1.0e-6, 100 )
  dt = params[0]
  q = params[1]
  m = params[2]
  apply_correction = params[3]


  qmdt2 = q * dt / (2.0 * m)

  # number of time steps
  N = np.size(time)

  # state vector at every time step
  # assumes 3D cartestian for 6D phase space
  X = np.zeros((N,6))

  # setting the half-step back for velocities
  Ex0, Ey0, Ez0 = Efield(X0[0],X0[1],X0[2])
  Bx0, By0, Bz0 = Bfield(X0[0],X0[1],X0[2])

  #


  # fill the first row with ICs
  X[0,:] = X0

  x  = X[0,0]
  y  = X[0,1]
  z  = X[0,2]
  vx = X[0,3]
  vy = X[0,4]
  vz = X[0,5]

  for n in range(0,N-1): # why not (0,N)?
    Ex, Ey, Ez = EField(x,y,z)
    Bx, By, Bz = BField(x,y,z)

    # Frequency correction factor alpha -> only good for x and y
    alpha = frequency_correction( qm*dt*0.5*Bz, apply_correction )

    # Step 1. Half acceleration (E-field)
    vminus_x = X[n,3] + alpha * qm * dt * 0.5 * Ex
    vminus_y = X[n,4] + alpha * qm * dt * 0.5 * Ey
    vminus_z = X[n,5] + qm * dt * 0.5 * Ez

    # Step 2A. B-field rotation
    tx = alpha * qm * Bx * dt * 0.5
    ty = alpha * qm * By * dt * 0.5
    tz = alpha * qm * Bz * dt * 0.5

    vprime_x += vy*tz - vz*ty
    vprime_y += vz*tx - vx*tz
    vprime_z += vx*ty - vy*tx

    # Step 2B. B-field rotation
    tmag2 = tx*tx + ty*ty + tz*tz
    sx = 2.0*tx/(1 + tmag2)
    sy = 2.0*ty/(1 + tmag2)
    sz = 2.0*tz/(1 + tmag2)

    vpx = vx + vprime_y*sz - vprime_z*sy
    vpy = vy + vprime_z*sx - vprime_x*sz
    vpz = vz + vprime_x*sy - vrpime_y*sx

    # Step 3. Half acceleration (E-field)

    X[n+1,3] = vplus_x + alpha * qm * dt * 0.5 * Ex
    X[n+1,4] = vplus_y + alpha * qm * dt * 0.5 * Ey
    X[n+1,5] = vplus_z + qm * dt * 0.5 * Ez

    # Step 4. Particle push

    X[n+1,0] = X[n,0] + X[n+1,3]*dt
    X[n+1,1] = X[n,1] + X[n+1,4]*dt
    X[n+1,2] = X[n,2] + X[n+1,5]*dt

  return X

# def fun(t,X):
#   x, y, z, vx, vy, vz = X
#   # Charge-to-mass ratio (q/m)
#   qm = qe/me
#   # E-field [V/m]
#   Ex = 0.0
#   Ey = 100.0
#   Ez = 0.0
#   # B-field [T]
#   Bx = 0.0
#   By = 0.0
#   Bz = 1.0e-4
#   # Newton-Lorentz equation in Cartesian coordinates
#   Xdot = np.zeros(6)
#   Xdot[0] = vx
#   Xdot[1] = vy
#   Xdot[2] = vz
#   Xdot[3] = qm * ( Ex + vy*Bz - vz*By )
#   Xdot[4] = qm * ( Ey + vz*Bx - vx*Bz )
#   Xdot[5] = qm * ( Ez + vx*By - vy*Bx )
#   return Xdot


def main():
  # Grid
  time = np.linspace( 0.0, 1.1e-6, 100 )
  # Initial conditions
  X0 = np.array(( 0.0, 0.0, 0.0, 0.0, 1.0e6, 0.0 ))

  # Solve ODE
  X_ef = ode.euler( fun, time, X0 )       # Forward Euler
  X_mp = ode.midpoint( fun, time, X0 )    # Explicit Midpoint
  X_rk = ode.rk4( fun, time, X0 )         # Runge-Kutta 4

  # for i in range(0,xn.size):
  #     print xn[i], y_an[i], y_ef[i,0], y_mp[i,0], y_rk[i,0]

  plt.figure(1)
  plt.plot( X_ef[:,0], X_ef[:,1], 'ro-', label='Forward Euler (1st)' )
  plt.plot( X_mp[:,0], X_mp[:,1], 'go-', label='Explicit Mid-Point (2nd)' )
  plt.plot( X_rk[:,0], X_rk[:,1], 'bx-', label='Runge-Kutta (4th)' )
  plt.xlabel('x [m]')
  plt.ylabel('y [m]')
  plt.axis('equal')
  plt.legend(loc=3)
  plt.savefig('ex01_particle_ExB.png')
  plt.show()

if __name__ == '__main__':
  main()
