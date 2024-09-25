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

def verlet(time, X0, params):
  dt = time[1] - time[0]

  q = params[0]
  m = params[1]
  qm = q/m

  N = np.size(time)

  X = np.zeros((N,6))

  X[0,:] = X0

  for n in range(0,N-1):
    Ex, Ey, Ez = Efield(X[n,0],X[n,1],X[n,2])
    Bx, By, Bz = Bfield(X[n,0],X[n,1],X[n,2])

    ax_n = qm*Ex + qm*(X[n,4]*Bz - X[n,5]*By)
    ay_n = qm*Ey + qm*(X[n,5]*Bx - X[n,3]*Bz)
    az_n = qm*Ez + qm*(X[n,3]*By - X[n,4]*Bx0)




  return
