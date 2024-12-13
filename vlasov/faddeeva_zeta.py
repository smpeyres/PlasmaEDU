import numpy as np
from scipy.special import wofz  # Faddeeva function
import scipy.optimize as opt
import matplotlib.pyplot as plt

def Z(zeta):
    """
    Plasma dispersion function Z(zeta) = i*sqrt(pi) * w(zeta)
    where w(zeta) is the Faddeeva function
    """
    return 1j * np.sqrt(np.pi) * wofz(zeta)

def Zprime(zeta):
    """
    Derivative of plasma dispersion function
    Z'(zeta) = -2[1 + zeta*Z(zeta)]
    """
    return -2.0 * (1.0 + zeta * Z(zeta))

def root_equation(zeta, x):
    """
    Equation to find roots of: 1 + zeta*Z(zeta) + x^2 = 0
    where x = k*lambda_D
    """
    return 1.0 + zeta*Z(zeta) + x**2

def find_root(x, zeta_guess=1.0+0.0j):
    """
    Find root of the dispersion relation for given x = k*lambda_D
    """
    root = opt.fsolve(lambda z: [np.real(root_equation(z[0] + 1j*z[1], x)),
                                np.imag(root_equation(z[0] + 1j*z[1], x))],
                      [np.real(zeta_guess), np.imag(zeta_guess)])
    return root[0] + 1j*root[1]

def calc_omega(zeta, k, lambda_D):
    """
    Calculate omega/omega_pe from zeta
    omega/omega_pe = zeta * k * v_th/omega_pe = zeta * sqrt(2) * k*lambda_D
    """
    return zeta * np.sqrt(2) * k * lambda_D

# Calculate roots for a range of k*lambda_D values
x_values = np.linspace(0.1, 2.0, 100)
zeta_values = []
omega_values = []

# Use previous root as initial guess for next calculation
zeta_guess = 1.0 + 0.0j

for x in x_values:
    zeta = find_root(x, zeta_guess)
    zeta_values.append(zeta)
    omega = calc_omega(zeta, 1.0, x)  # Note: k*lambda_D = x
    omega_values.append(omega)
    zeta_guess = zeta

# Convert to numpy arrays
zeta_values = np.array(zeta_values)
omega_values = np.array(omega_values)

# Plot results
plt.plot(x_values, np.real(omega_values)/np.sqrt(2), 'k-', label='Re($ω/ω_{pe}$)')
plt.plot(x_values, np.imag(omega_values)/np.sqrt(2), 'b-', label='Im($ω/ω_{pe}$)')
plt.xlabel('$kλ_D$')
plt.ylabel('$ω_{r,i}/ω_{pe}$')
plt.grid(True)
plt.legend()
plt.savefig('landau_damping_dispersion.png',dpi=600)
plt.show()

# Print some specific values for verification
test_points = [0.25, 0.50, 0.75, 1.00, 2.00]
print("\nVerification points:")
print("kλD      ωr/ωpe    γi/ωpe")
print("--------------------------")
for x in test_points:
    idx = np.abs(x_values - x).argmin()
    print(f"{x:.2f}    {np.real(omega_values[idx]/np.sqrt(2)):8.6f}  {np.imag(omega_values[idx]/np.sqrt(2)):10.6f}")
