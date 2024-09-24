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
  qm = q/m
  apply_correction = params[2]

  # number of time steps
  N = np.size(time)

  # state vector at every time step
  # assumes 3D cartestian for 6D phase space
  X = np.zeros((N,6))

  # setting the half-step back for velocities
  Ex0, Ey0, Ez0 = Efield(X0[0],X0[1],X0[2])
  Bx0, By0, Bz0 = Bfield(X0[0],X0[1],X0[2])

  # electric force
  ax0 = qm*Ex0
  ay0 = qm*Ey0
  az0 = qm*Ez0

  # magnetic force
  ax0 += qm*(X0[4]*Bz0 - X0[5]*By0)
  ay0 += qm*(X0[5]*Bx0 - X0[3]*Bz0)
  az0 += qm*(X0[3]*By0 - X0[4]*Bx0)

  # push back - see document
  vxm12 = X0[3] - 0.5*ax0*dt
  vym12 = X0[4] - 0.5*ay0*dt
  vzm12 = X0[5] - 0.5*az0*dt

  # setting first row
  X[0,:] = np.array([X0[0], X0[1], X0[2], vxm12, vym12, vzm12])

  # # determine the first entries
  # x  = X[0,0]
  # y  = X[0,1]
  # z  = X[0,2]
  # vx = X[0,3]
  # vy = X[0,4]
  # vz = X[0,5]

  # we made the rest of this in class, so I'll trust it for now
  for n in range(0,N-1):

    # grab fields
    Ex, Ey, Ez = Efield(X[n,0],X[n,1],X[n,2])
    Bx, By, Bz = Bfield(X[n,0],X[n,1],X[n,2])

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

    vprime_x = vminus_x + (vminus_y*tz - vminus_z*ty)
    vprime_y = vminus_y + (vminus_z*tx - vminus_x*tz)
    vprime_z = vminus_z + (vminus_x*ty - vminus_y*tx)

    # Step 2B. B-field rotation
    tmag2 = tx*tx + ty*ty + tz*tz
    sx = 2.0*tx/(1 + tmag2)
    sy = 2.0*ty/(1 + tmag2)
    sz = 2.0*tz/(1 + tmag2)

    vplus_x = vminus_x + (vprime_y*sz - vprime_z*sy)
    vplus_y = vminus_y + (vprime_z*sx - vprime_x*sz)
    vplus_z = vminus_z + (vprime_x*sy - vprime_y*sx)

    # Step 3. Half acceleration (E-field)

    X[n+1,3] = vplus_x + alpha * qm * dt * 0.5 * Ex
    X[n+1,4] = vplus_y + alpha * qm * dt * 0.5 * Ey
    X[n+1,5] = vplus_z + qm * dt * 0.5 * Ez

    # Step 4. Particle push

    X[n+1,0] = X[n,0] + X[n+1,3]*dt
    X[n+1,1] = X[n,1] + X[n+1,4]*dt
    X[n+1,2] = X[n,2] + X[n+1,5]*dt

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

  # 1 period, 6 steps per period
  time_1per = np.linspace(0, period, 6)

  # 100 periods, 15 steps per period
  time_100per = np.linspace(0, 100*period, 100*15)

  # Initial conditions - start at origin and positive velocity along y-direction
  X0 = np.array(( 0.0, 0.0, 0.0, 0.0, 1.0e6, 0.0 ))

  # Solve ODE
  X_1per_nocorr = boris_bunemann(time_1per, X0, params_nocorr)
  X_1per_corr = boris_bunemann(time_1per, X0, params_corr)
  X_100per_nocorr = boris_bunemann(time_100per, X0, params_nocorr)
  X_100per_corr = boris_bunemann(time_100per, X0, params_corr)

  plt.figure(1)
  plt.plot( X_1per_nocorr[:,0], X_1per_nocorr[:,1], 'o--', color='r', label='BB w/o Freq. Corr.' )
  plt.plot( X_1per_corr[:,0], X_1per_corr[:,1], 'o--', color='b', label='BB w/ Freq. Corr.' )
  plt.xlabel('x [m]')
  plt.ylabel('y [m]')
  plt.axis('equal')
  plt.legend()
  plt.savefig('boris_04.png',dpi=600)
  plt.show()

if __name__ == '__main__':
  main()
