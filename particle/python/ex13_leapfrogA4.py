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

# okay, now let's think about the leapfrog...
# x_n+1 = x_n + v_n+1/2 dt
# v_n+1/2 = v_n-1/2 + O^2 x_n dt
# Use the def v_n = (v_n+1/2 - v_n-1/2)/2 to obtain
# v_n = v_n-1/2 + 1/2 a dt
# then, v_-1/2 = v_0 - 1/2 a_0 dt
# where a = - O^2 x or Xdot[1]

def leapfrog(fun, x, y0, correction=True):
  # following the basic structure of the functions in ode.py
  N = np.size(x) # get number of timesteps
  h = x[1] - x[0] # get size of timestep
  I = np.size(y0) # number of initial values -> number of positions and velocities
  y = np.zeros((N,I)) # make a matrix that contains position and velocity values for each timestep
  # okay, let's make it clear what the first entries are:
  # for position : n = 0
  # for velocity : n = -1/2
  # okay, we need to update y0
  # I'll keep it as simple 1D problem for now
  xn0, vn0 = y0
  an0 = fun(0, [xn0, vn0])[1]
  freq = np.sqrt(-1*an0/xn0)
  ynm12 = vn0 - 0.5*an0*h
  y[0,:] = np.array([xn0,ynm12])

  if correction == False:
    for n in range(0, N-1):
      # update velocity first:
      y[n+1,1] = y[n,1] + h*fun( x[n], y[n,:])[1]
      # update position next:
      y[n+1,0] = y[n,0] + h*y[n+1,1]
  else:
    freq_corr = np.sin(0.5*freq*h)/(0.5*freq*h)
    for n in range(0, N-1):
      # update velocity first:
      y[n+1,1] = y[n,1] + h*(freq_corr**2)*fun( x[n], y[n,:])[1]
      # update position next:
      y[n+1,0] = y[n,0] + h*y[n+1,1]


  return y




def main():
  x0 = 1
  v0 = 0

  time = np.linspace(0,4*np.pi,2*10)
  time_an = np.linspace(0,4*np.pi,2*10*10)

  X0 = np.array([x0,v0])

  X_rk4 = ode.rk4(fun, time, X0)

  X_leap_no_corr = leapfrog(fun, time, X0, correction=False)

  X_leap_corr = leapfrog(fun, time, X0, correction=True)

  X_an = np.cos(time)

  error_rk4 = ode.error_absolute(X_an, X_rk4[:,0])
  error_leap_no_corr = ode.error_absolute(X_an, X_leap_no_corr[:,0])
  error_leap_corr = ode.error_absolute(X_an, X_leap_corr[:,0])


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
  plt.plot(time/(2*np.pi), X_rk4[:,0], 'o--', color='b', label="Runge-Kutta 4")
  plt.plot(time/(2*np.pi), X_leap_no_corr[:,0], 'o--', color='r', label="Leapfrog w/o Freq. Corr.")
  plt.plot(time/(2*np.pi), X_leap_corr[:,0], 'o--', color='g', label="Leapfrog w/ Freq. Corr.")
  plt.xlabel('time, $t\Omega/2\pi$')
  plt.ylabel('position, $x$')
  plt.legend(loc=3)
  plt.draw()
  plt.savefig("ex13_verification.png",dpi=600)
  plt.show()

  plt.figure(2)
  plt.plot(time/(2*np.pi), error_rk4, 'o--', color='b', label="Runge-Kutta 4")
  plt.plot(time/(2*np.pi), error_leap_no_corr, 'o--', color='r', label="Leapfrog w/o Freq. Corr.")
  plt.plot(time/(2*np.pi), error_leap_corr, 'o--', color='g', label="Leapfrog w/ Freq. Corr.")
  plt.xlabel('time, $t\Omega/2\pi$')
  plt.ylabel('absolute error')
  plt.legend(loc=0)
  plt.draw()
  plt.savefig("ex13_error.png",dpi=600)
  plt.show()


if __name__ == '__main__':
   main()
