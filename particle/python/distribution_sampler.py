import numpy as np
import matplotlib.pyplot as plt

kb = 1.380649e-23 # Boltzmann constant [J/K]

def v_mag_avg(m, T):
  """
  Calculate the average magnitude of velocity for a 1D Maxwellian distribution.

  Parameters:
  m (float): Particle mass in kg.
  T (float): Temperature of the gas in K.

  Returns:
  float: Average magnitude of velocity.
  """
  return np.sqrt(2 * kb * T / (np.pi * m))

def maxwellian_1D(vi, m, T):
  """
  One-dimensional Maxwellian distribution probability density function for velocity.

  Parameters:
  vi (float): Velocity at which to evaluate the PDF within a single dimension (vx, vy, or vz)
  m (float): Particle mass in kg
  T (float): Temperature of the gas in K

  Returns:
  float: Probability density at vi.

  Note:
  The 3D Maxwellian velocity distribution is simply the product of three 1D Maxwellian distributions,
  one for each velocity component.
  """
  return np.sqrt(m/(2*np.pi*kb*T)) * np.exp(-m*vi**2/(2*kb*T))

def generate_maxwellian_velocity(m, T, num_samples):
  """
  Generate samples from a Maxwellian distribution probability density function (PDF).

  Parameters:
  m (float): Particle mass in kg.
  T (float): Temperature of the gas in K.
  num_samples (int): Number of samples to generate.

  Returns:
  np.ndarray: Array of generated samples.
  """

  # Obtain the average magnitude of velocity
  v_avg = v_mag_avg(m, T)

  # Set boundaries for sampling
  x_min = -5.0 * v_avg
  x_max = 5.0 * v_avg

  samples = []
  max_pdf = max(maxwellian_1D(x, m, T) for x in np.linspace(x_min, x_max, 1000))  # Find the maximum value of the PDF in the range

  while len(samples) < num_samples:
    x = np.random.uniform(x_min, x_max)
    y = np.random.uniform(0, max_pdf)
    if y < maxwellian_1D(x, m, T):
      samples.append(x)
  return np.array(samples)

# num_samples = 10000
# m = 40*1.67e-27  # roughly the mass of argon in kg
# T = 300  # temperature in K

# # Generate samples
# samples_maxwell = generate_maxwellian_velocity(m, T, num_samples)

# # Check that the integral over all samples approximates 1 for the Maxwellian distribution
# hist, bin_edges = np.histogram(samples_maxwell, bins=50, density=True)
# bin_width = bin_edges[1] - bin_edges[0]
# integral = np.sum(hist * bin_width)
# print(f"Integral over all samples (Maxwellian): {integral}")

# fig1 = plt.figure(figsize=(10, 5))
# # Plot histogram of samples
# plt.hist(samples_maxwell, bins=50, density=True, alpha=0.6, color='b')

# # Plot the Maxwellian PDF for comparison
# v = np.linspace(-5.0*v_mag_avg(m,T), 5.0*v_mag_avg(m,T), 1000)
# pdf_values_maxwell = maxwellian_1D(v, m, T)
# plt.plot(v, pdf_values_maxwell, 'r', linewidth=2)

# plt.title(f'Histogram of {num_samples} Samples and 1D Maxwellian PDF for Ar at {T} K')
# plt.xlabel('v (m/s)')
# plt.ylabel('Density (s/m)')
# plt.savefig(f'1D_maxwellian_{num_samples}samples.png',dpi=600)
# plt.show()

