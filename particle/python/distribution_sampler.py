import numpy as np
import matplotlib.pyplot as plt

kb = 1.380649e-23 # Boltzmann constant [J/K]

def gaussian_pdf(x, mean, std_dev):
    """
    Gaussian distribution probability density function.

    Parameters:
    x (float): Value at which to evaluate the PDF.
    mean (float): Mean of the Gaussian distribution.
    std_dev (float): Standard deviation of the Gaussian distribution.

    Returns:
    float: Probability density at x.
    """
    return (1 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std_dev) ** 2)

def maxwellian_pdf(v, m, T):
    """
    Maxwellian distribution probability density function.

    Parameters:
    v (float): Value at which to evaluate the PDF.
    m (float): Particle mass in kg
    T (float): Temperature of the gas in K

    Returns:
    float: Probability density at v.
    """
    return 4 * np.pi * (m / (2 * np.pi * kb * T)) ** (3 / 2) * (v ** 2) * np.exp(-m * v ** 2 / (2 * kb * T))

def generate_samples(pdf, num_samples, x_range):
    """
    Generate samples from a given probability density function (PDF).

    Parameters:
    pdf (function): Probability density function to sample from.
    num_samples (int): Number of samples to generate.
    x_range (tuple): Range of x values to consider for sampling (min, max).

    Returns:
    np.ndarray: Array of generated samples.
    """
    x_min, x_max = x_range
    samples = []
    max_pdf = max(pdf(x) for x in np.linspace(x_min, x_max, 1000))  # Find the maximum value of the PDF in the range

    while len(samples) < num_samples:
        x = np.random.uniform(x_min, x_max)
        y = np.random.uniform(0, max_pdf)
        if y < pdf(x):
            samples.append(x)
    return np.array(samples)

# Example usage:
mean = 0
std_dev = 1
num_samples = 10000
x_range = (-5, 5)

# Generate samples
samples_gauss = generate_samples(lambda x: gaussian_pdf(x, mean, std_dev), num_samples, x_range)

fig1 = plt.figure(figsize=(10, 5))
# Plot histogram of samples
plt.hist(samples_gauss, bins=50, density=True, alpha=0.6, color='g')

# Plot the Gaussian PDF for comparison
x = np.linspace(x_range[0], x_range[1], 1000)
pdf_values = gaussian_pdf(x, mean, std_dev)
plt.plot(x, pdf_values, 'r', linewidth=2)

plt.title('Histogram of Samples and Gaussian PDF')
plt.xlabel('x')
plt.ylabel('Density')
plt.show()

# Parameters for Maxwellian distribution
m = 40*1.67e-27  # roughly the mass of argon in kg
T = 300  # temperature in K
v_range = (0, 3e3)  # velocity range in m/s

# Generate samples
samples_maxwell = generate_samples(lambda v: maxwellian_pdf(v, m, T), num_samples, v_range)

# Check that the integral over all samples approximates 1 for the Maxwellian distribution
hist, bin_edges = np.histogram(samples_maxwell, bins=50, density=True)
bin_width = bin_edges[1] - bin_edges[0]
integral = np.sum(hist * bin_width)
print(f"Integral over all samples (Maxwellian): {integral}")


fig2 = plt.figure(figsize=(10, 5))
# Plot histogram of samples
plt.hist(samples_maxwell, bins=50, density=True, alpha=0.6, color='b')

# Plot the Maxwellian PDF for comparison
v = np.linspace(v_range[0], v_range[1], 1000)
pdf_values_maxwell = maxwellian_pdf(v, m, T)
plt.plot(v, pdf_values_maxwell, 'r', linewidth=2)

plt.title('Histogram of Samples and Maxwellian PDF')
plt.xlabel('v (m/s)')
plt.ylabel('Density')
plt.show()
