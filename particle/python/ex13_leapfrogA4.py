import sys
import numpy as np
import matplotlib.pyplot as plt

# a 1-D harmonic oscillator problem
# d2x/dt2 + O^2 x = 0
# x(t=0) = x_{n=0} = 1
# v(t=0) = v_{n=0} = 0
# for leapfrog, does v_{n=-1/2} = v_{n=+1/2} = 0?

sys.path.insert(1, '../../ode/python/')
import ode

# dv/dt + O^2 x = 0 => dv/dt = - O^2 x
# RHS of the ODE problem, dy/dx = f(x,y)
def fun(t,X):
  x, v = X
  Xdot = np.zeros(2)
  Xdot[0] = v
  Xdot[1] = - 1 * x # assumes O = 1 for simplicity
  return Xdot

def main():
  x0 = 1
  v0 = 0

  time = np.linspace(0,4*np.pi,2*10)
  time_an = np.linspace(0,4*np.pi,2*10*10)

  X0 = np.array([x0,v0])

  X_rk4 = ode.rk4(fun(freq=1), time, X0)

  # okay, now let's think about the leapfrog...
  # x_n+1 = x_n + v_n+1/2 dt
  # v_n+1/2 = v_n-1/2 + O^2 x_n dt
  # okay, I can't set both v_+1/2 and v_-1/2 equal to zero...
  # I should only set v_-1/2 equal to zero.
  # then it becomes practically is not different from
  # v_{n=0} = 0, v_{n=1} = ...
  # so the real scheme is
  # x_n+1 = x_n + v_n+1 dt
  # v_n+1 = v_n + O^2 x_n dt
  # where we calculate v_n+1 first then x_n+1
  # seems like we have some type of euler scheme then... but not quite...





  plt.figure(1)
  plt.plot(time_an/(2*np.pi), np.cos(time_an), '-', color='k', label="Analytical Solution")
  plt.plot(time/(2*np.pi), X_rk4[:,0], 'o', color='b', label="Runge-Kutta 4")
  plt.xlabel('time, $t\Omega/2\pi$')
  plt.ylabel('position, $x$')
  plt.legend()
  plt.show()


if __name__ == '__main__':
   main()
