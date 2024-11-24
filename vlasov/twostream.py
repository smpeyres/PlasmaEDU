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

def push_particles_dt( x,vx, dt, L,dx, Efield, Qfield, Qp, Ze ):
    N_part  = np.size(x)
    N_nodes = np.size(Qfield)
    qmdt2  = Ze * QE / ME * dt / 2.0
    # Reset Charge on nodes
    for i in range(N_nodes):
        Qfield[i] = 0.0
    # Push particles and weight their charge to the nodes
    for i in range(N_part):
        # Step 1: Find weights and cell number
        w0, w1, icell = weights(x[i],dx)
        # Step 2: Interpolate E-field at particle location
        Ex = Efield[icell]*w1 + Efield[icell+1]*w0
        # Step 3A: Boris-Bunemann, push velocity (E-field only)
        vx[i] += 2.0 * qmdt2 * Ex
        # Step 3B: Boris-Bunemann, push positions
        x[i] += vx[i] * dt
        # Periodicity on x-coordinate of the particle
        x[i] = np.mod(x[i],L)
        # Find new weights and cell number
        w0, w1, icell = weights(x[i],dx)
        # Add charge to grid nodes
        Qfield[icell  ] += (Qp*w1)
        Qfield[icell+1] += (Qp*w0)

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

def efield(Q_nodes,L,dx):
    N_nodes = np.size(Q_nodes)
    rho_e = np.zeros(N_nodes) # [C/m^3] charge density
    # Find electron charge density from charge dividing by covolume
    rho_e[0] = Q_nodes[0]/(0.5*dx)
    rho_e[N_nodes-1] = Q_nodes[N_nodes-1]/(0.5*dx)
    for i in range(1,N_nodes-1):
        rho_e[i] = Q_nodes[i] / dx
    # Average ion charge density
    rho_i = np.sum(Q_nodes)/L
    # Assemble R.H.S. of Poisson Equation
    rhs = (rho_i - rho_e)/EPS0 * dx*dx
    # Solve Poisson Equation and find the electric potential [V]
    phi = poisson_1d_periodic(rhs)
    # Find E-field [V/m]
    E_nodes = gradphi(phi,dx)
    return E_nodes, phi, rho_e

def main():

    # Domain
    L       = 0.01                           # [m] Domain size
    N_nodes = 500                            # [#] Number of nodes
    dx      = L / (N_nodes-1)                # [m] Cell size
    grid    = np.linspace( 0.0, L, N_nodes ) # [m] Grid

    # Time
    N_steps = 1000        # [#] Number of time steps
    dt      = 2.5e-11   # [s] Time step

    # Particle Distribution Function
    Ze      = -1.0                           # [#]     Electron charge number
    n0      = 1e17                           # [m^-3]  Electron beam density
    Te      = 1.0 * eV2K                     # [eV->K] Electron beam temperature
    Vthe    = np.sqrt(2.0*KB*Te/ME)          # [m/s]   Electron thermal speed
    Ue      = 5.0*Vthe                       # [m/s]   Electron beam velocity
    N_part  = 20000                          # [#]     Number of computational particles
    p2c     = n0 * L / N_part                # [#]     Physical-to-Computational ratio
    Qp      = (QE*Ze) * p2c                  # [C]     Charge of one macroparticle

    # Particle list, made of 2 electron beams
    x  = np.random.uniform(0.0, L, N_part)    # [m]     Inital position of the particles
    Npart_beam1 = (int)(np.floor(N_part/2.0)) # [#]     Number of computational particles in Beam 1
    Npart_beam2 = N_part - Npart_beam1        # [#]     Number of computational particles in Beam 2
    beam1 = np.random.normal(  Ue, Vthe, Npart_beam1 )  # [m/s]   Initial velocity of Beam 1
    beam2 = np.random.normal( -Ue, Vthe, Npart_beam2 )  # [m/s]   Initial velocity of Beam 2
    vx = np.concatenate((beam1, beam2))       # [m/s]   Initial velocity of the particles

    # Fields
    Q_nodes = np.zeros(N_nodes)
    E_nodes = np.zeros(N_nodes)

    # Time series
    Esquare = np.zeros(N_steps)

    # Time loop
    plt.figure(1)
    for n in range(N_steps):

        # Step 1. Push particles in time ("PARTICLE STEP")
        push_particles_dt( x,vx, dt, L, dx, E_nodes,Q_nodes,Qp,Ze)

        # Step 2. Solve for the Electric field ("MESH STEP")
        E_nodes, phi, rho_e = efield(Q_nodes,L,dx)

        # Integral of electrostatic energy in the domain
        Esquare[n] = 0.5*EPS0*np.sum(E_nodes**2*dx)

        # Plot Phase Space (x,vx) of Particles
        plt.plot( x[0:Npart_beam1],vx[0:Npart_beam1]/Vthe, 'r.' )
        plt.plot( x[Npart_beam1:], vx[Npart_beam1: ]/Vthe, 'b.' )
        plt.xlim([0,L])
        plt.ylim([-12,12])
        plt.xlabel('x [m]')
        plt.ylabel('vx / vth')
        plt.title('Electron Beam Phase Space')
        plt.draw()
        plt.pause(.0001)
        plt.clf()
        # plt.savefig('phasespace_t'+str(n)+'.png')

        # plt.plot(E_nodes)
        # plt.xlim([0,L])
        # plt.ylim([-10000,10000])
        # plt.draw()
        # plt.pause(.0001)
        # plt.clf()

    plt.figure(2)
    plt.plot(Esquare)
    plt.xlabel('time steps')
    plt.ylabel('Electrostatic Energy, (1/2) EPS0 E^2')
    plt.savefig('esenergy.png')
    plt.show()

if __name__ == '__main__':
   main()
