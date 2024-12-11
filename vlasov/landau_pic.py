import numpy as np
import matplotlib.pyplot as plt

# Fundamental constants
QE    = 1.602176634e-19     # Elementary charge [C]
MP    = 1.67262192369e-27   # Proton mass [kg]
ME    =  9.10938356e-31     # Electron mass [kg]
KB    = 1.38064852e-23      # Boltzmann constant [J/K]
EPS0  = 8.854187817620e-12  # Vacuum Permittivity [F/m]
eV2K  = QE/KB               # Conversion from eV to kelvin

class Constants:
    def __init__(self):
        self.QE    = 1.602176634e-19     # [C]     Elementary charge
        self.MP    = 1.67262192369e-27   # [kg]    Proton mass
        self.ME    =  9.10938356e-31     # [kg]    Electron mass
        self.KB    = 1.38064852e-23      # [J/K]   Boltzmann constant
        self.EPS0  = 8.854187817620e-12  # [F/m]   Vacuum Permittivity
        self.eV2K  = self.QE/self.KB     # [eV->K] Conversion from eV to kelvin

class Particle:
    def __init__(self, x, vx, Ms, Zs):
        self.x  = x
        self.vx = vx
        self.Ms = Ms
        self.Zs = Zs
        self.Qs = Zs*QE
    def move(self, dt):
        self.vx += 2.0 * qmdt2 * Ex
        self.x  += self.vx*dt
    def push(self, dt):
        self.vx += qmdt2 * Ex
        self.x  += self.vx*dt
    def pull(self, dt):
        self.vx += qmdt2 * Ex
        self.x  += self.vx*dt

class MaxwellianSpecies:
    def __init__(self, N, x0, vx0, Ms, Zs):
        self.N   = N
        self.x   = np.zeros(N)
        self.vx  = np.zeros(N)
        self.Ms  = Ms
        self.Zs  = Zs
        self.Qs  = Zs*QE
        self.vth = self.get_vth(Te)
        self.x   = np.random.normal(x0, 0.1, N)
        self.vx  = np.random.normal(vx0, self.vth, N)
        self.particle = Particle(self.x, self.vx, self.Ms, self.Zs)
    def get_vth(self, Te):
        return np.sqrt(2.0*KB*Te/self.Ms)

def weights(xp,dx):
    icell = (int)(np.absolute( np.floor(xp/dx) ))
    x_node_left = icell*dx
    w0 = (xp-x_node_left)/dx
    w1 = 1.0 - w0
    return w0, w1, icell

def push_particles_dt(x, vx, dt, L, dx, Efield, Qfield, Qp, Ze):
    N_part = np.size(x)
    N_nodes = np.size(Qfield)
    qmdt2 = 0.5 * dt  # Normalized q/m = 1

    Qfield.fill(0.0)

    for i in range(N_part):
        w0, w1, icell = weights(x[i], dx)
        Ex = Efield[icell]*w1 + Efield[(icell+1) % N_nodes]*w0
        vx[i] += 2.0 * qmdt2 * Ex  # Normalized push
        x[i] += vx[i] * dt
        x[i] = np.mod(x[i], L)

        w0, w1, icell = weights(x[i], dx)
        Qfield[icell] += (Qp*w1)
        Qfield[(icell+1) % N_nodes] += (Qp*w0)

def poisson_1d_periodic(rhs):
    N_nodes = np.size(rhs)
    phi = np.zeros(N_nodes)
    phi[0] = 0.0
    for i in range(N_nodes):
        phi[0] += ((i+1)*rhs[i])
    phi[0] = phi[0]/N_nodes
    phi[1] = rhs[0] + 2.0*phi[0]
    for i in range(2,N_nodes):
        phi[i] = rhs[i-1] + 2.0*phi[i-1] - phi[i-2]
    return phi

def gradphi(phi,dx):
    N_nodes = np.size(phi)
    Efield  = np.zeros(N_nodes)
    for i in range(1,N_nodes-1):
        Efield[i] = -(phi[i+1] - phi[i-1])/2.0/dx
    Efield[0] = -(phi[1] - phi[0])/2.0/dx
    Efield[N_nodes-1] = -(phi[N_nodes-1] - phi[N_nodes-2])/2.0/dx
    return Efield

def efield(Q_nodes, L, dx):
    N_nodes = np.size(Q_nodes)
    rho_e = np.zeros(N_nodes)

    rho_e[0] = Q_nodes[0]/(0.5*dx)
    rho_e[N_nodes-1] = Q_nodes[N_nodes-1]/(0.5*dx)
    for i in range(1, N_nodes-1):
        rho_e[i] = Q_nodes[i] / dx

    rho_i = np.sum(Q_nodes)/L

    # Normalized Poisson equation (-∇²φ = ρ)
    rhs = -(rho_i - rho_e) * dx*dx
    phi = poisson_1d_periodic(rhs)
    E_nodes = gradphi(phi, dx)
    return E_nodes, phi, rho_e

def main():
    # Domain
    k_target = 0.5
    L       = 2.0*np.pi/k_target
    N_nodes = 128
    dx      = L / (N_nodes-1)
    grid    = np.linspace(0.0, L, N_nodes)

    # Time
    dt      = 0.02   # From inputdata.py, normalized units
    N_steps = 1000

    # Particle parameters (all normalized)
    Ze      = -1.0
    n0      = 1.0
    N_part  = 1000000 # 1e6 particles
    p2c     = n0 * L / N_part
    Qp      = Ze * p2c

    x_uniform = np.linspace(0, L, N_part, endpoint=False)
    perturb = 0.1  # Same as vlasov.py: nx = 1.0 - perturb*np.cos(k*X)
    x = x_uniform.copy()
    density = 1.0 - perturb * np.cos(k_target * x)  # Match vlasov.py exactly
    F = np.cumsum(density) * L/np.sum(density)
    x = np.interp(x_uniform, F, x)

    # Check density perturbation
    hist, bins = np.histogram(x, bins=N_nodes, range=(0,L))
    print(f"Density perturbation check:")
    print(f"Min/Max density: {np.min(hist)/np.mean(hist):.3f}, {np.max(hist)/np.mean(hist):.3f}")
    print(f"Should be close to: {1-perturb:.3f}, {1+perturb:.3f}")

    # Thermal velocities with sqrt(2) correction
    vx = np.random.normal(0, np.sqrt(0.5), N_part)  # Matches vlasov.py normalization

    # Normalized field arrays
    Q_nodes = np.zeros(N_nodes)
    E_nodes = np.zeros(N_nodes)
    Esquare = np.zeros(N_steps)

    # Time loop
    plt.figure(1)
    for n in range(N_steps):
        push_particles_dt(x, vx, dt, L, dx, E_nodes, Q_nodes, Qp, Ze)
        E_nodes, phi, rho_e = efield(Q_nodes, L, dx)

        # Energy diagnostic matching vlasov.py
        Esquare[n] = 0.5*np.trapz(E_nodes**2, x=grid)
        Enorm = np.sqrt(Esquare)  # This is key - we need to sqrt after integration

        # Phase space plot
        if n % 20 == 0:  # Match vlasov.py plotting frequency
            plt.clf()
            plt.plot(x, vx, 'b.', markersize=1)
            plt.xlim([0, L])
            plt.ylim([-6, 6])
            plt.xlabel('x')
            plt.ylabel('v')
            plt.title(f'Phase Space')
            plt.draw()
            plt.pause(0.01)

                    # Plot phase space
        if n == N_steps-1:  # At final timestep
            plt.figure(1)
            plt.clf()
            plt.plot(x, vx, 'b.', markersize=1)
            plt.xlim([0, L])
            plt.ylim([-6, 6])
            plt.xlabel('x')
            plt.ylabel('v')
            plt.title('Phase Space')
            plt.savefig('phasespace_final_pic_1e6.png', dpi=300, bbox_inches='tight')

    # Energy plot matching vlasov.py format
    plt.figure(2)
    t = np.arange(N_steps)*dt
    plt.semilogy(t, Enorm)  # Note sqrt to match vlasov normalization
    plt.xlabel('Time')
    plt.ylabel('L2 Norm, Electric Field')
    plt.grid(True)
    plt.ylim([1e-4, 10])
    plt.show()

    # After time loop, save energy data
    np.savez('Enorm_pic_1e6.npz',
            Enorm=Enorm,
            time=np.arange(N_steps)*dt)  # Save time array too

if __name__ == '__main__':
   main()
