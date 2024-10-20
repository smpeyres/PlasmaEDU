import numpy as np
from pylab import plot, axis, show, xlim, ylim, savefig
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from distribution_sampler import generate_maxwellian_velocity

# Physical Constants (SI units, 2019 redefinition)
qe   = -1.602176634e-19      # charge of electron [C]
qp   =  1.602176634e-19      # charge of proton [C]
me   = 9.109383701528e-31    # electron rest mass [kg]
mp   = 1.6726219236951e-27   # proton rest mass [kg]
lux  = 299792458.0           # speed of light [m/s]
hp   = 6.62607015e-34        # Planck constant [Js]
muref= 1.0000000005415e-7    # Reference measure of mu0
mu0  = 4.0*np.pi*muref       # Vacuum permeability [H/m]
eps0 = 1/lux/lux/mu0         # Vacuum permittivity [F/m]
fine = qe*qe*lux*mu0/2.0/hp  # Fine structure
kc   = 1.0/4.0/np.pi/eps0    # Coulomb constant
hbar = hp/2.0/np.pi          # h-bar
epsilon = 1.0e-15            # Small number (nucleus size) [m]

# Bohr model (SI units)
a0   = hbar/me/lux/fine             # Bohr radius
mk   = kc*qe*qe/me
vb   = np.sqrt(mk/a0)               # Bohr speed
tb   = 2.0*np.pi*np.sqrt(a0**3/mk)  # Bohr period

# Number of particles
Np = 20

# Charge and Mass -> protons and electrons
q = np.concatenate( (qp*np.ones(Np//2), -qe*np.ones(Np//2) ) )
m = np.concatenate( (mp*np.ones(Np//2),  me*np.ones(Np//2) ) )

# Characteristic time [s]
T = 200.0*tb #20.0*tb
print('T=',T,' [s]')

# Characteristic size of the domain [m]
L = 100.0*a0
print('L=',L,' [m]')

Rx = np.random.rand(Np)*L
Ry = np.random.rand(Np)*L
Rz = np.random.rand(Np)*L

Tp = 11604 # Plasma temperature in K (1.0 eV)

# Initialize initial velocities for protons and electrons
vx_protons = generate_maxwellian_velocity(mp, Tp, Np//2)
vy_protons = generate_maxwellian_velocity(mp, Tp, Np//2)
vz_protons = generate_maxwellian_velocity(mp, Tp, Np//2)

vx_electrons = generate_maxwellian_velocity(me, Tp, Np//2)
vy_electrons = generate_maxwellian_velocity(me, Tp, Np//2)
vz_electrons = generate_maxwellian_velocity(me, Tp, Np//2)

# Concatenate velocities
Vx = np.concatenate( ( vx_protons, vx_electrons ) )
Vy = np.concatenate( ( vy_protons, vy_electrons ) )
Vz = np.concatenate( ( vz_protons, vz_electrons ) )

# Dynamic function, Newton-Lorentz Equation
def dynamics(time,y):

   rx = y[0*Np:1*Np]
   ry = y[1*Np:2*Np]
   rz = y[2*Np:3*Np]

   vx = y[3*Np:4*Np]
   vy = y[4*Np:5*Np]
   vz = y[5*Np:6*Np]

   # Electric field
   Ex = 0.0
   Ey = 0.0
   Ez = 0.0

   # Magnetic field
   Bx = 0.0
   By = 0.0
   Bz = 0.0

   ax = np.zeros(Np)
   ay = np.zeros(Np)
   az = np.zeros(Np)

   for i in range(Np):
      for j in range(Np):
         if (j!=i):

            rx_ij = rx[i] - rx[j]
            ry_ij = ry[i] - ry[j]
            rz_ij = rz[i] - rz[j]

            # Apply minimum image convention for periodic boundary conditions
            rx_ij -= L * np.round(rx_ij / L)
            ry_ij -= L * np.round(ry_ij / L)
            rz_ij -= L * np.round(rz_ij / L)

            r_ij = np.sqrt( rx_ij**2 + ry_ij**2 + rz_ij**2 ) + epsilon

            Fx_ij = kc * q[i]*q[j] * rx_ij / (r_ij**3)
            Fy_ij = kc * q[i]*q[j] * ry_ij / (r_ij**3)
            Fz_ij = kc * q[i]*q[j] * rz_ij / (r_ij**3)

            ax[i] += Fx_ij/m[i]
            ay[i] += Fy_ij/m[i]
            az[i] += Fz_ij/m[i]

   return np.concatenate( (vx, vy, vz, ax, ay, az) )


def ode4( f, y0, x ):
   '''
    Runge-Kutta 4th order
    -----------------------------
    Butcher Table:

    0   | 0     0     0     0
    1/2 | 1/2   0     0     0
    1/2 | 0     1/2   0     0
    1   | 0     0     1     0
    -----------------------------
        | 1/6   1/3   1/3   1/6
   '''
   N = np.size( x )
   h = x[1] - x[0]
   I = np.size( y0 )
   y = np.zeros((N,I))
   y[0,:] = y0
   for n in range(0, N-1):
      k1 = h * f( x[n]       , y[n,:]        )
      k2 = h * f( x[n]+h/2.0 , y[n,:]+k1/2.0 )
      k3 = h * f( x[n]+h/2.0 , y[n,:]+k2/2.0 )
      k4 = h * f( x[n]+h     , y[n,:]+k3     )
      y[n+1,:] = y[n,:] + k1/3.0 + k2/6.0 + k3/6.0 + k4/3.0

      # Apply periodic boundary conditions
      y[n+1, 0*Np:1*Np] = y[n+1, 0*Np:1*Np] % L
      y[n+1, 1*Np:2*Np] = y[n+1, 1*Np:2*Np] % L
      y[n+1, 2*Np:3*Np] = y[n+1, 2*Np:3*Np] % L

   return y


# Time interval
tspan = np.linspace(0.0, T, 200)

# Initial conditions
Y0 = np.zeros( (6*Np) )
Y0 = np.concatenate( ( Rx, Ry, Rz, Vx, Vy, Vz ) )

# Solve the ODE
Y = ode4( dynamics, Y0, tspan )

Rx = Y[ :, 0*Np:1*Np ]
Ry = Y[ :, 1*Np:2*Np ]
Rz = Y[ :, 2*Np:3*Np ]

Vx = Y[ :, 3*Np:4*Np ]
Vy = Y[ :, 4*Np:5*Np ]
Vz = Y[ :, 5*Np:6*Np ]

# Function to wrap coordinates around the boundaries
def wrap_coordinates(arr, L):
    return arr % L

# Function to handle boundary crossings and plot trajectories
def plot_trajectories(Rx, Ry, L, Np):
    for i in range(Np):
        x = Rx[:, i]
        y = Ry[:, i]

        # Detect boundary crossings
        x_diff = np.diff(x)
        y_diff = np.diff(y)

        # Find indices where boundary crossing occurs
        x_crossings = np.where(np.abs(x_diff) > L / 2)[0]
        y_crossings = np.where(np.abs(y_diff) > L / 2)[0]

        # Combine and sort unique crossing indices
        crossings = np.unique(np.concatenate((x_crossings, y_crossings)))

        # Plot segments between crossings
        start_idx = 0
        for idx in crossings:
            plt.plot(x[start_idx:idx+1], y[start_idx:idx+1], 'r-' if i < Np//2 else 'b-')
            start_idx = idx + 1

        # Plot the last segment
        plt.plot(x[start_idx:], y[start_idx:], 'r-' if i < Np//2 else 'b-')

# Wrap the coordinates
Rx_wrapped = wrap_coordinates(Rx, L)
Ry_wrapped = wrap_coordinates(Ry, L)

# Plot trajectories
plot_trajectories(Rx_wrapped, Ry_wrapped, L, Np)

# Plot initial positions
plt.plot(Rx_wrapped[0, 0:Np//2], Ry_wrapped[0, 0:Np//2], 'ro', label='Initial Protons')  # Initial protons
plt.plot(Rx_wrapped[0, Np//2:Np], Ry_wrapped[0, Np//2:Np], 'bo', label='Initial Electrons')  # Initial electrons

# Plot final positions
plt.plot(Rx_wrapped[-1, 0:Np//2], Ry_wrapped[-1, 0:Np//2], 'rs', label='Final Protons')  # Final protons
plt.plot(Rx_wrapped[-1, Np//2:Np], Ry_wrapped[-1, Np//2:Np], 'bs', label='Final Electrons')  # Final electrons

plt.legend()
plt.xlim([0, L])
plt.ylim([0, L])
plt.savefig('ex09B_nbody_20body.png', dpi=200)
plt.show()
