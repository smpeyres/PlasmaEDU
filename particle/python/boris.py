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
  # calculate timestep (assumes constant timestep)
  dt = time[1] - time[0]

  # get value of charge, mass, and if frequency correction on or off
  q = params[0]
  m = params[1]
  apply_correction = params[2]

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
  # x, y, z, vx, vy, vz


  Ex0, Ey0, Ez0 = Efield(x,y,z)
  Bx0, By0, Bz0 = Bfield(x,y,z)

  vxm12 = X0[3] - 


  for n in range(0, N-1):
    

  # determine the first 
  x  = X[0,0]
  y  = X[0,1]
  z  = X[0,2]
  vx = X[0,3]
  vy = X[0,4]
  vz = X[0,5]

  # we made the rest of this in class, so I'll trust it for now
  for n in range(0,N-1):
    Ex, Ey, Ez = Efield(x,y,z)
    Bx, By, Bz = Bfield(x,y,z)

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

    vprime_x = vy*tz - vz*ty
    vprime_y = vz*tx - vx*tz
    vprime_z = vx*ty - vy*tx

    # Step 2B. B-Field rotation
    tsq = tx*tx + ty*ty + tz*tz

    sx = 2.0 * tx / (1.0 + tsq)
    sy = 2.0 * ty / (1.0 + tsq)
    sz = 2.0 * tz / (1.0 + tsq)

    vpx = vx + vprime_y*sz - vprime_z*sy
    vpy = vy + vprime_z*sx - vprime_x*sz
    vpz = vz + vprime_x*sy - vprime_y*sx

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

# numerical & physics parameters and no 
dt = 1.0e-9
q = qe
m = me
params = np.array(( q, m, False ))

def main():
  # physics parameters and freq correction
  q = qe
  m = me
  params_nocorr = np.array(( q, m, False ))
  params_corr = np.array(( q, m, True ))

  cyclotron_freq = q*0.1/m # assuming uniform field of 0.1 T
  period = 2*np.pi/cyclotron_freq

  # 2 periods, 15 steps per period
  time_2per = np.linspace(0, 2*period, 2*15)

  # 100 periods, 15 steps per period
  time_100per = np.linspace(0, 100*period, 100*15)

  # Initial conditions - start at origin and positive velocity along y-direction
  X0 = np.array(( 0.0, 0.0, 0.0, 0.0, 1.0e6, 0.0 ))

  # Solve ODE
  X_2per_nocorr = boris_bunemann(time_2per, X0, params_nocorr)
  X_2per_corr = boris_bunemann(time_2per, X0, params_corr)
  X_100per_nocorr = boris_bunemann(time_100per, X0, params_nocorr)
  X_100per_corr = boris_bunemann(time_100per, X0, params_corr)

  plt.figure(1)
  plt.plot( X_2per_nocorr[:,0], X_2per_nocorr[:,1], 'o--', color='r', label='BB w/o Freq. Corr.' )
  plt.plot( X_2per_corr[:,0], X_2per_corr[:,1], 'o--', color='b', label='BB w/ Freq. Corr.' )
  plt.xlabel('x [m]')
  plt.ylabel('y [m]')
  plt.axis('equal')
  plt.legend()
  plt.savefig('boris_01.png',dpi=600)
  plt.show()

if __name__ == '__main__':
  main()
