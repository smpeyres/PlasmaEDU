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
  return np.array([Ex, Ey, Ez])

# define static, uniform B-field
def Bfield(x,y,z):
  Bx = 0.0
  By = 0.0
  Bz = 0.1
  return np.array([Bx, By, Bz])

def CrossProd(a,b):
  c = np.zeros(3)
  c[0] = a[1]*b[2] - a[2]*b[1]
  c[1] = a[2]*b[0] - a[0]*b[2]
  c[2] = a[0]*b[1] - a[1]*b[0]
  return c

def DotProd(a,b):
  c = a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
  return c

def Lorentz(E,B,vOld,q,m,dt):
  # inputs are all vectors
  Sig = q*E/m
  Omg = q*B/m
  A = Omg*dt/2
  C = vOld + dt*(Sig + CrossProd(vOld,Omg/2))
  num = C + A*DotProd(A,C) - CrossProd(A,C)
  dom = 1 + DotProd(A,A)
  vNew = num/dom
  return vNew

def Leapfrog(time, X0, params):
  dt = time[1] - time[0]

  q = params[0]
  m = params[1]

  N = np.size(time)

  X = np.zeros((N,6))

  X[0,:] = X0

  for n in range(0,N-1):
    v_old = np.array([X[n,3],X[n,4],X[n,5]])
    E_old = Efield(X[n,0],X[n,1],X[n,2])
    B_old = Bfield(X[n,0],X[n,1],X[n,2])
    v_new = Lorentz(E_old, B_old, v_old, q, m, dt)
    X[n+1,3] = v_new[0]
    X[n+1,4] = v_new[1]
    X[n+1,5] = v_new[2]

    X[n+1,0] = X[n,0] + 0.5*(v_new[0] + v_old[0])*dt
    X[n+1,1] = X[n,1] + 0.5*(v_new[1] + v_old[1])*dt
    X[n+1,2] = X[n,2] + 0.5*(v_new[2] + v_old[2])*dt

  return X

def main():
  # physics parameters and freq correction
  q = qe
  m = me
  params = np.array([q,m])

  cyclotron_freq = q*0.1/m # assuming uniform field of 0.1 T
  period = 2*np.pi/cyclotron_freq

  r_L = 1.0e6/cyclotron_freq

  # 2 period, 15 steps per period
  time_2per = np.linspace(0, 2*period, 2*15)

  # 100 periods, 15 steps per period
  time_100per = np.linspace(0, 100*period, 100*15)

  x_an = r_L*np.cos(cyclotron_freq*time_100per) + r_L

  y_an = r_L*np.sin(cyclotron_freq*time_100per)

  # Initial conditions - start at origin and positive velocity along y-direction
  X0 = np.array(( 0.0, 0.0, 0.0, 0.0, 1.0e6, 0.0 ))

  # Solve ODE
  X_2per = Leapfrog(time_2per, X0, params)
  X_100per = Leapfrog(time_100per, X0, params)

  radius_an = r_L*np.ones(np.size(time_100per))

  radius_100 = np.sqrt(DotProd(X_100per[:,0],X_100per[:,0]) + DotProd(X_100per[:,1],X_100per[:,1]))

  error = ode.error_absolute(radius_an, radius_100)

  plt.figure(1)
  plt.plot( X_2per[:,0]/1000, X_2per[:,1]/1000, 'o--', color='r', label='421 Leapfrog' )
  plt.plot( x_an/1000, y_an/1000, '-', color='k', label="Analytical Solution")
  plt.xlabel('x [mm]')
  plt.ylabel('y [mm]')
  plt.axis('equal')
  plt.legend()
  plt.savefig('421_01.png',dpi=600)
  plt.show()

  plt.figure(2)
  plt.plot( time_100per*cyclotron_freq/(2*np.pi), error, 'o--', color='r', label='421 Leapfrog.')
  plt.xlabel('time, $t\Omega/2\pi$')
  plt.ylabel('absolute error')
  # plt.axis('equal')
  plt.legend()
  plt.savefig('421_02.png',dpi=600)
  plt.show()

if __name__ == '__main__':
  main()
