import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(1, '../../ode/python/')
import ode

qe = 1.60217662e-19
me = 9.10938356e-31
mp = 1.6726219e-27

# okay, I really need to get GitHub CoPilot... but it is not intelligent
# then why would I get it? Maybe these easy small changes

dt = 1.0e-9
q = qe
m = me
params = np.array(( dt, q, m ))

def Efield(x,y,z):
  Ex = 0.0
  Ey = 0.0
  Ez = 0.0
  return Ex, Ey, Ez

def Bfield(x,y,z):
  Bx = 0.0
  By = 0.0
  Bz = 0.1
  return Bx, By, Bz

def frequency_correction(x, apply_correction=True):
  if apply_correction:
    alpha = np.tan(x)/x
  else:
    alpha = 1.0
  return alpha

def boris_bunemann( time, X0, params ):
  # time = np.linspace( 0.0, 1.0e-6, 100 )
  dt = params[0]
  q = params[1]
  m = params[2]
  apply_correction = params[3]


  qmdt2 = q * dt / (2.0 * m)

  # number of time steps
  N = np.size(time) # why not np.len?

  # state vector at every time step
  X = np.zeros((N,6))

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

    # Frequency correction factor alpha
    alpha_x = frequency_correction( qmdt2*Bx, apply_correction )
    alpha_y = frequency_correction( qmdt2*By, apply_correction )
    alpha_z = frequency_correction( qmdt2*Bz, apply_correction )

    # Step 1. Half acceleration (E-field)
    vx += qmdt2 * Ex * alpha_x
    vy += qmdt2 * Ey * alpha_y
    vz += qmdt2 * Ez * alpha_z

    # for a vector implementation
    # vel += qmdt2 * Evector

    # Step 2A. B-field rotation
    tx = qmdt2 * Bx * alpha_x
    ty = qmdt2 * By * alpha_y
    tz = qmdt2 * Bz * alpha_z

    vprime_x += vy*tz - vz*ty
    vprime_y += vz*tx - vx*tz
    vprime_z += vx*ty - vy*tx

    # Step 2B. B-Field rotation
    tsq = tx*tx + ty*ty + tz*tz

    sx = 2.0 * tx / (1.0 + tsq)
    sy = 2.0 * ty / (1.0 + tsq)
    sz = 2.0 * tz / (1.0 + tsq)

    vpx = vx + vprime_y*sz - vprime_z*sy
    vpy = vy + vprime_z*sx - vprime_x*sz
    vpz = vz + vprime_x*sy - vrpime_y*sx

    # Step 3. Half acceleration (E-field)
    vx = vpx + qmdt2 * Ex * alpha_x
    vy = vpy + qmdt2 * Ey * alpha_y
    vz = vpz + qmdt2 * Ez * alpha_z

    # Step 4. Push position
    x += vx*dt
    y += vy*dt
    z += vz*dt

    # Store the coordinates
    X[n+1,0] = x
    X[n+1,1] = y
    X[n+1,2] = z
    X[n+1,3] = vx
    X[n+1,4] = vy
    X[n+1,5] = vz

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
