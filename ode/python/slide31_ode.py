import ode
import numpy as np
import matplotlib.pyplot as plt

def fun(x,y):
    ydot = y - x*x + 1.0
    return ydot

def main():
    xn   = np.linspace( 0.0, 5.0, 20 )     # Grid
    y0   = np.array( [ 0.5 ] )            # Initial condition
    y_ef = ode.euler( fun, xn, y0 )       # Forward Euler
    y_mp = ode.midpoint( fun, xn, y0 )    # Explicit Midpoint
    y_rk = ode.rk4( fun, xn, y0 )         # Runge-Kutta 4
    y_an = xn**2 + 2.0*xn + 1.0 - 0.5*np.exp(xn) # Analytical

    for i in range(0,xn.size):
        print(xn[i], y_an[i], y_ef[i,0], y_mp[i,0], y_rk[i,0])

    print(np.shape(xn))
    print(np.shape(y_ef))
    print(np.shape(y_mp))
    print(np.shape(y_rk))
    print(np.shape(y_an))

    plt.figure(1)
    plt.plot( xn, y_ef, 'ro-', label='Forward Euler (1st)' )
    plt.plot( xn, y_mp, 'go-', label='Explicit Mid-Point (2nd)' )
    plt.plot( xn, y_rk, 'bx-', label='Runge-Kutta (4th)' )
    plt.plot( xn, y_an, 'k-',  label='Analytical Solution' )
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc=3)
    plt.savefig('slide31_ode_solution.png')
    plt.show()

    plt.figure(2)
    plt.plot( xn, ode.error_absolute(y_an, y_ef[:,0]), 'ro-', label='Forward Euler (1st)' )
    plt.plot( xn, ode.error_absolute(y_an, y_mp[:,0]), 'go-', label='Explicit Mid-Point (2nd)' )
    plt.plot( xn, ode.error_absolute(y_an, y_rk[:,0]), 'bx-', label='Runge-Kutta (4th)' )
    plt.xlabel('x')
    plt.ylabel('absolute error')
    plt.legend()
    plt.savefig('slide31_ode_abs_error.png')

    plt.figure(3)
    plt.plot( xn, ode.error_relative(y_an, y_ef[:,0]), 'ro-', label='Forward Euler (1st)' )
    plt.plot( xn, ode.error_relative(y_an, y_mp[:,0]), 'go-', label='Explicit Mid-Point (2nd)' )
    plt.plot( xn, ode.error_relative(y_an, y_rk[:,0]), 'bx-', label='Runge-Kutta (4th)' )
    plt.xlabel('x')
    plt.ylabel('relative error')
    plt.legend()
    plt.savefig('slide31_ode_rel_error.png')

    plt.figure(4)
    plt.plot( xn, ode.error_percent(y_an, y_ef[:,0]), 'ro-', label='Forward Euler (1st)' )
    plt.plot( xn, ode.error_percent(y_an, y_mp[:,0]), 'go-', label='Explicit Mid-Point (2nd)' )
    plt.plot( xn, ode.error_percent(y_an, y_rk[:,0]), 'bx-', label='Runge-Kutta (4th)' )
    plt.xlabel('x')
    plt.ylabel('percent error')
    plt.legend()
    plt.savefig('slide31_ode_perc_error.png')

if __name__ == '__main__':
   main()