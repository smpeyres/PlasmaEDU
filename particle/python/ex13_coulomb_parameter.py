import numpy as np
from pylab import plot, axis, show, xlim, ylim, savefig
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Physical Constants (SI units, 2019 redefinition)
qe   = -1.602176634e-19      # electron charge [C]
qp   =  1.602176634e-19      # proton charge [C]
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
kb = 1.380649e-23            # Boltzmann constant [J/K]

def mean_speed(T,m):
  """
  Calculate the mean speed of particles in a plasma.

  Parameters:
  T (float): Temperature of the plasma in Kelvin.
  m (float): Mass of a single particle in kilograms.

  Returns:
  float: The mean speed of the particles in meters per second.
  """
  return np.sqrt(8 * kb * T / (np.pi * m))

def generate_random_velocities(T,m):
  """
  Generate a random velocity vector based on the given temperature and mass.

  This function generates three random numbers, normalizes them to create a unit vector,
  and then scales this unit vector by the mean speed calculated from the given temperature (T)
  and mass (m) to produce a velocity vector.

  Args:
    T (float): Temperature in Kelvin.
    m (float): Mass of the particle in kilograms.

  Returns:
    tuple: A tuple containing three components of the velocity vector (vx, vy, vz).
  """
  random_numbers = 2 * np.random.rand(3) - 1

  # Normalize the random numbers
  random_numbers = random_numbers / np.linalg.norm(random_numbers)

  magnitude = np.linalg.norm(random_numbers)

  # Calculate the mean speed
  mean_v = mean_speed(T,m)

  # Scale the random numbers by the mean speed
  velocity_components = random_numbers * mean_v

  magnitude = np.linalg.norm(velocity_components)

  return velocity_components

# Begin selecting operating parameters

# Number of total particles -> 20 protons and 20 electrons
Np = 40

def target_density(G, Tp):
  """
  Calculate the target plasma density from Coulomb coupling parameter and plasma temperature.

  Inputs:
  G (float): Coulomb coupling parameter.
  Tp (float): Plasma temperature in Kelvin.

  Returns:
  float: Target plasma density in particles per cu
  """

  # Wigner-Seitz radius
  a = qe**2 / (G * kb * Tp)

  return 3*(a**-3)/4*np.pi

Tp = 1.0e4 # Plasma temperature [K] -> Jupiter interior

G = 175  # Coulomb coupling parameter

target_density = target_density(G, Tp)

# Calculate the characteristic size of the domain [m] to achieve the target density
L = 5e-9
print('L=', L, '[m]')

# Calculate the proton plasma frequency [Hz] based on the target density

def plasma_frequency(n):
  """
  Calculate the plasma frequency based on the target density.

  Inputs:
  n (float): Target plasma density in particles per cubic meter.

  Returns:
  float: Proton plasma frequency in Hertz.
  """
  return np.sqrt(n * qe**2 / (mp * eps0))

frequency = plasma_frequency(target_density)
period = 2*np.pi / frequency
timestep = period / 10

# Characteristic time as 150 periods of the plasma frequency
T = 150 * period
print('T=', T, '[s]')

# Charge and mass -> protons and electrons
q = np.concatenate( (qp*np.ones(Np//2),  qe*np.ones(Np//2) ) )
m = np.concatenate( (mp*np.ones(Np//2),  me*np.ones(Np//2) ) )

# Random initial positions
Rx = np.random.rand(Np)*L
Ry = np.random.rand(Np)*L
Rz = np.random.rand(Np)*L

# Initialize velocities
Vx_protons = np.zeros(Np//2)
Vy_protons = np.zeros(Np//2)
Vz_protons = np.zeros(Np//2)

Vx_electrons = np.zeros(Np//2)
Vy_electrons = np.zeros(Np//2)
Vz_electrons = np.zeros(Np//2)

# Generate random velocities for protons and electrons
for i in range(Np//2):
  velocities = generate_random_velocities(Tp, mp)
  Vx_protons[i], Vy_protons[i], Vz_protons[i] = velocities
  velocities = generate_random_velocities(Tp, me)
  Vx_electrons[i], Vy_electrons[i], Vz_electrons[i] = velocities

# Concatenate velocities
Vx = np.concatenate((Vx_protons, Vx_electrons))
Vy = np.concatenate((Vy_protons, Vy_electrons))
Vz = np.concatenate((Vz_protons, Vz_electrons))


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
    y[n+1, 0*Np:1*Np] = np.mod(y[n+1, 0*Np:1*Np], L)
    y[n+1, 1*Np:2*Np] = np.mod(y[n+1, 1*Np:2*Np], L)
    y[n+1, 2*Np:3*Np] = np.mod(y[n+1, 2*Np:3*Np], L)

   return y


# Time interval
tspan = np.arange(0.0, T, timestep)

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

# def calculate_collision_timescale(Vx, Vy, Vz, tspan):
#   """
#   Calculate the collision timescale based on a 90-degree change in velocity vectors.

#   Parameters:
#   Vx, Vy, Vz (ndarray): Arrays of velocities for each particle over time
#   tspan (ndarray): Array of time points

#   Returns:
#   float: Mean collision timescale
#   float: Standard deviation of the collision timescale
#   """
#   Np = np.size(Vx, 1)
#   Nsteps = np.size(Vx, 0)
#   collision_times = []

#   dt = tspan[1] - tspan[0]

#   for i in range(Np):
#     for t in range(Nsteps - 1):
#       v1 = np.array([Vx[t, i], Vy[t, i], Vz[t, i]])
#       norm_v1 = np.linalg.norm(v1)
#       if norm_v1 == 0:
#         continue

#       for t2 in range(t + 1, Nsteps):
#         v2 = np.array([Vx[t2, i], Vy[t2, i], Vz[t2, i]])
#         norm_v2 = np.linalg.norm(v2)
#         if norm_v2 == 0:
#           continue

#         dot_product = np.dot(v1, v2) / (norm_v1 * norm_v2)
#         if np.abs(dot_product) < np.cos(np.deg2rad(90)):
#           collision_times.append(t2 * dt)

#   if collision_times:
#     mean_collision_time = np.mean(collision_times)
#     std_collision_time = np.std(collision_times)
#   else:
#     mean_collision_time = float('nan')
#     std_collision_time = float('nan')

#   return mean_collision_time, std_collision_time

# # Calculate the collision timescale
# mean_collision_time, std_collision_time = calculate_collision_timescale(Vx, Vy, Vz, tspan)
# print(f"Mean collision timescale: {mean_collision_time} s")
# print(f"Standard deviation of collision timescale: {std_collision_time} s")

# # Plot Vx for each particle as a function of time
# plt.figure()
# for i in range(Np):
#   plt.plot(tspan, Vx[:, i], label=f'Particle {i+1}')
# plt.xlabel('Time [s]')
# plt.ylabel('Vx [m/s]')
# plt.title('Vx for each particle as a function of time')
# plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1))
# plt.grid(True)
# plt.show()

# # plot trajectories
# plot( Rx[:,0:Np//2], Ry[:,0:Np//2], 'ro-') # protons
# plot( Rx[:,Np//2:Np], Ry[:,Np//2:Np], 'b-') # electrons

# Plot initial positions
plot(Rx[0, 0:Np//2], Ry[0, 0:Np//2], 'ro', label='Initial Protons')  # Initial protons
plot(Rx[0, Np//2:Np], Ry[0, Np//2:Np], 'bo', label='Initial Electrons')  # Initial electrons

# Plot final positions
plot(Rx[-1, 0:Np//2], Ry[-1, 0:Np//2], 'rs', label='Final Protons')  # Final protons
plot(Rx[-1, Np//2:Np], Ry[-1, Np//2:Np], 'bs', label='Final Electrons')  # Final electrons


plt.legend()
xlim([0, L])
ylim([0, L])
savefig('nbody_1.png', dpi=200)
show()
