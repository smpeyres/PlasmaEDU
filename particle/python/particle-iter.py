import sys
import numpy as np
import matplotlib.pyplot as plt

# point-by-point input; does not accept vector/array inputs.
def psi_iter_like(R,Z):

        R0   = 6.2
        A    = -0.155
        Psi0 = 202.92

        # Normalized coordinates w.r.t. major radius
        x = R/R0
        y = Z/R0

        # Powers of x and y and log
        x2 = x*x
        x4 = x2*x2
        y2 = y*y
        y4 = y2*y2
        lnx = np.log(x)

        # Single-null Grad Shafranov functions
        psi_i = np.zeros(12)
        coeff = np.zeros(12)

        psi_i[0] = 1.0
        psi_i[1] = x2
        psi_i[2] = y2 - x2*lnx
        psi_i[3] = x4 - 4.0*x2*y2
        psi_i[4] = 2.0*y4 - 9.0*y2*x2 + 3.0*x4*lnx - 12.0*x2*y2*lnx
        psi_i[5] = x4*x2 - 12.0*x4*y2 + 8.0*x2*y4
        psi_i[6] = 8.0*y4*y2 - 140.0*y4*x2 +75.0*y2*x4 -15.0*x4*x2*lnx + 180.0*x4*y2*lnx - 120.0*x2*y4*lnx
        psi_i[7] = y
        psi_i[8] = y*x2
        psi_i[9] = y*y2 - 3.0*y*x2*lnx
        psi_i[10] = 3.0*y*x4 - 4.0*x2*y2*y
        psi_i[11] = 8.0*y4*y - 45.0*y*x4 - 80.0*y2*y*x2*lnx + 60.0*y*x4*lnx

        # Coefficients for ITER-like magnetic equilibrium
        coeff[0] =  1.00687012e-1
        coeff[1] =  4.16274456e-1
        coeff[2] = -6.53880989e-1
        coeff[3] = -2.95392244e-1
        coeff[4] =  4.40037966e-1
        coeff[5] = -4.01807386e-1
        coeff[6] = -1.66351381e-2
        coeff[7] =  1.92944621e-1
        coeff[8] =  8.36039453e-1
        coeff[9] = -5.30670714e-1
        coeff[10]= -1.26671504e-1
        coeff[11]=  1.47140977e-2

        psi = np.dot(coeff, psi_i) + x4/8.0 + A * (0.5*x2*lnx - x4/8.0)

        Psi = Psi0*psi

        return Psi



























def psi_matrix(R,Z):

        """
        psi = psi_iter_like(R,Z)

        Note that this usage is innappropriate,
        as psi_iter_like does not accept vector inputs.
        Psi is evaluated at which point R,Z provided individually.

        Let's try a point-wise approach instead.
        """
        N_r = len(R)
        N_z = len(Z)

        # create empty matrix for psi
        psi = np.zeros((N_r,N_z))

        # loop over all points and evaluate Psi at each point
        for i in range(0,len(R)):
                for j in range(0,len(Z)):
                        psi[i,j] = psi_iter_like(R[i],Z[j])

        return psi

# generate the polodial field in radial coordinates
def B_pol_rad(R,Z):

        # number of steps
        N_r = len(R)
        N_z = len(Z)

        # make empty matrices for each component
        Br_pol = np.zeros((N_r,N_z))
        Bz_pol = np.zeros((N_r,N_z))

        # calculate the uniform step size
        h_R = R[1] - R[0]
        h_Z = Z[1] - Z[0]

        psi = psi_matrix(R,Z)

        # create empty matrix for psi
        psi = np.zeros((N_r,N_z))

        # loop over all points and evaluate Psi at each point
        for i in range(0,len(R)):
                for j in range(0,len(Z)):
                        psi[i,j] = psi_iter_like(R[i],Z[j])

        # bottom and top boundaries
        for i in range(0,len(R)):
                Br_pol[i, 0]  = (1/R[i])*(-3*psi[i,0] + 4*psi[i,1] - psi[i,2])/(2*h_Z)
                Br_pol[i,-1] = (1/R[i])*(3*psi[i,-1] - 4*psi[i,-2] + psi[i,-3])/(2*h_Z)

        # rest of domain
        for i in range(0,len(R)):
                for j in range(1,len(Z)-1):
                        Br_pol[i, j]  = (1/R[i])*(psi[i,j+1] - psi[i,j-1])/(2*h_Z)


        # left and right boundaries
        for i in range(0,len(Z)):
                Bz_pol[0, i]  = -(1/R[0])*(-3*psi[0,i] + 4*psi[1,i] - psi[2,i])/(2*h_R)
                Bz_pol[-1,i] = -(1/R[-1])*(3*psi[-1,i] - 4*psi[-2,i] + psi[-3,i])/(2*h_R)

        # rest of domain
        for i in range(1,len(R)-1):
                for j in range(0,len(Z)):
                        Bz_pol[i, j]  = -(1/R[i])*(psi[i+1,j] - psi[i-1,j])/(2*h_R)

        return Br_pol, Bz_pol














R0   = 6.2

R_vals = np.linspace(  0.6*R0, 1.4*R0, 100 )
Z_vals = np.linspace( -0.8*R0, 0.8*R0, 100 )

plt.figure(1)
RR,ZZ = np.meshgrid(R_vals/R0,Z_vals/R0)
plt.contourf(np.transpose(RR),np.transpose(ZZ),psi_matrix(R_vals,Z_vals),30,cmap="RdBu_r")
plt.colorbar()
plt.xlabel('R/R0')
plt.ylabel('Z/R0')
plt.title('ITER Polodial Flux Function [Tm${}^2$]')
plt.savefig('iter_flux_function_test.png')

# generate the polodial field in radial coordinates
def B_pol_rad(R,Z):

        N_r = len(R)
        N_z = len(Z)

        R0 = 6.2

        # make empty matrices for each component
        Br_pol = np.zeros((N_r,N_z))
        Bz_pol = np.zeros((N_r,N_z))

        # uniform step sizes
        h_R = (R[-1] - R[0])/(N_r-1)
        h_Z = (Z[-1] - Z[0])/(N_z-1)

        psi = psi_matrix(R,Z)

        # bottom and top boundaries
        for i in range(0,len(R)):
                Br_pol[i, 0]  = (1/R[i])*(-3*psi[i,0] + 4*psi[i,1] - psi[i,2])/(2*h_Z)
                Br_pol[i,-1] = (1/R[i])*(3*psi[i,-1] - 4*psi[i,-2] + psi[i,-3])/(2*h_Z)

        # rest of domain
        for i in range(0,len(R)):
                for j in range(1,len(Z)-1):
                        Br_pol[i, j]  = (1/R[i])*(psi[i,j+1] - psi[i,j-1])/(2*h_Z)


        # left and right boundaries
        for i in range(0,len(Z)):
                Bz_pol[0, i]  = -(1/R[0])*(-3*psi[0,i] + 4*psi[1,i] - psi[2,i])/(2*h_R)
                Bz_pol[-1,i] = -(1/R[-1])*(3*psi[-1,i] - 4*psi[-2,i] + psi[-3,i])/(2*h_R)

        # rest of domain
        for i in range(1,len(R)-1):
                for j in range(0,len(Z)):
                        Bz_pol[i, j]  = -(1/R[i])*(psi[i+1,j] - psi[i-1,j])/(2*h_R)

        return Br_pol, Bz_pol

plt.figure(2)
RR,ZZ = np.meshgrid(R_vals/R0,Z_vals/R0)
plt.contourf(np.transpose(RR),np.transpose(ZZ),B_pol_rad(R_vals,Z_vals)[0],30,cmap="RdBu_r")
plt.colorbar()
plt.xlabel('R/R0')
plt.ylabel('Z/R0')
plt.title('Radial Component of Polodial Field [T]')
plt.savefig('BR_polodial_iter_test.png')

plt.figure(3)
RR,ZZ = np.meshgrid(R_vals/R0,Z_vals/R0)
plt.contourf(np.transpose(RR),np.transpose(ZZ),B_pol_rad(R_vals,Z_vals)[1],30,cmap="RdBu_r")
plt.colorbar()
plt.xlabel('R/R0')
plt.ylabel('Z/R0')
plt.title('Axial Component of Polodial Field [T]')
plt.savefig('BZ_polodial_iter_test.png')

# generate the magnitude of the polodial field
def B_pol_mag(R,Z):
        N_r = len(R)
        N_z = len(Z)

        Bpol_mag = np.zeros((N_r,N_z))
        Br_pol = B_pol_rad(R,Z)[0]
        Bz_pol = B_pol_rad(R,Z)[1]

        for i in range(0,len(R)):
                for j in range(0,len(Z)):
                        Bpol_mag[i,j] = np.sqrt(Br_pol[i,j]**2 + Bz_pol[i,j]**2)

        return Bpol_mag

plt.figure(4)
RR,ZZ = np.meshgrid(R_vals/R0,Z_vals/R0)
plt.contour(np.transpose(RR),np.transpose(ZZ),B_pol_mag(R_vals,Z_vals),100,cmap="RdBu_r")
plt.colorbar()
plt.xlabel('R/R0')
plt.ylabel('Z/R0')
plt.title('Magnitude of Polodial Field [T]')
plt.savefig('Bmag_polodial_iter_test.png')

R_limited = np.linspace(  4, 8, 100 )
Z_limited = np.linspace( -4, 4, 100 )

plt.figure(5)
RR,ZZ = np.meshgrid(R_limited,Z_limited)
plt.contour(np.transpose(RR),np.transpose(ZZ),B_pol_mag(R_limited,Z_limited),100,cmap="RdBu_r")
plt.colorbar()
plt.xlabel('R [m]')
plt.ylabel('Z [m]')
plt.title('Magnitude of Polodial Field [T]')
plt.savefig('Bmag_polodial_iter_limited_test.png')

"""

# polodial field in cartestian
# need to re-write the entire thing in order to have the proper # of indices
def B_pol_cart(x,y,z):

        R0 = 6.2

        N_x = len(x)
        N_y = len(y)
        N_z = len(z)

        R = np.sqrt(x*x + y*y)

        # make empty matrices for each component
        Bx_pol = np.zeros((N_x,N_y,N_z))
        By_pol = np.zeros((N_x,N_y,N_z))
        Bz_pol = np.zeros((N_x,N_y,N_z))

        # uniform step sizes
        h_x = (x[-1] - x[0])/(N_x-1)
        h_y = (y[-1] - y[0])/(N_y-1)
        h_Z = (z[-1] - z[0])/(N_z-1)

        # this is only 2D, but let's see if we can make it work
        psi = psi_matrix(R,Z)

        # okay, we note that
        # x = R cos(theta) and y = R sin(theta)
        # then df/dr = df/dx cos(theta) + df/dy sin(theta)
        # cos(theta) = x/R and sin(theta) = y/R
        # so df/dr = df/dx (x/R) + df/dy (y/R)

        # boundaries of x :
        for i in range(0,N_y):
                for j in range(0,N_z):
                        Bx_pol

####



                Br_pol[i, 0]  = (1/R[i])*(-3*psi[i,0] + 4*psi[i,1] - psi[i,2])/(2*h_Z)
                Br_pol[i,-1] = (1/R[i])*(3*psi[i,-1] - 4*psi[i,-2] + psi[i,-3])/(2*h_Z)

        # rest of domain
        for i in range(0,len(R)):
                for j in range(1,len(Z)-1):
                        Br_pol[i, j]  = (1/R[i])*(psi[i,j+1] - psi[i,j-1])/(2*h_Z)


        # left and right boundaries
        for i in range(0,len(Z)):
                Bz_pol[0, i]  = -(1/R[0])*(-3*psi[0,i] + 4*psi[1,i] - psi[2,i])/(2*h_R)
                Bz_pol[-1,i] = -(1/R[-1])*(3*psi[-1,i] - 4*psi[-2,i] + psi[-3,i])/(2*h_R)

        # rest of domain
        for i in range(1,len(R)-1):
                for j in range(0,len(Z)):
                        Bz_pol[i, j]  = -(1/R[i])*(psi[i+1,j] - psi[i-1,j])/(2*h_R)



        # Major radius (projection)
        R = np.sqrt(x*x + y*y)
        B_pol_start = B_pol_rad(R,z)
        # easy to assign z component
        Bz_pol = B_pol_start[1]
        # Okay, let's try the x and y components now
        # Sin, Cos of particle position on (x,y) plane
        ca = x/R
        sa = y/R
        Bx_pol = -1*B_pol_rad(R,z)[0] * sa
        By_pol = B_pol_rad(R,z)[0] * ca
        return Bx_pol, By_pol, Bz_pol

X_limited = np.linspace(  4, 8, 100 )
Y_limited = np.linspace(  4, 8, 100 )

print(B_pol_cart(X_limited,Y_limited,Z_limited))

print(np.shape(B_pol_cart(X_limited,Y_limited,Z_limited)))


def B_tor(x,y,z):
    B0 = 5.3 # on-axis according to wikipedia
    R0 = 6.2
    # Major radius (projection)
    R = np.sqrt( x*x + y*y )
    # Sin, Cos of particle position on (x,y) plane
    ca = x/R
    sa = y/R
    # Toroidal component [T]
    Bphi = B0 * R0 / R
    # B-field [T]
    Bx = -Bphi * sa
    By =  Bphi * ca
    Bz =  0.0
    return Bx, By, Bz

def B_total(x,y,z):
        R = np.sqrt(x*x + y*y)
        # I wish I had a better name than this
        B_Pol = B_pol(R,z)
        # Sin, Cos of particle position on (x,y) plane
        ca = x/R
        sa = y/R
        Bx_pol = -B_Pol[0]*sa
        By_pol =  B_Pol[0]*ca
        Bz_pol =  B_Pol[1]
        Bx = Bx_pol + B_tor(x,y,z)[0]
        By = By_pol + B_tor(x,y,z)[1]
        Bz = Bz_pol + B_tor(x,y,z)[2]
        return Bx, By, Bz

# making the values of X and Y, which are the same as R_35
X_35 = np.linspace(  0.6*R0, 1.4*R0, 100 )
Y_35 = np.linspace(  0.6*R0, 1.4*R0, 100 )

print(B_total(X_35,Y_35,Z_35)[0])
print(np.shape(B_total(X_35,Y_35,Z_35)[0]))



plt.figure(4)
XX,ZZ = np.meshgrid(X_35/R0,Z_35/R0)
plt.contourf(np.transpose(XX),np.transpose(ZZ),B_total(X_35,0,Z_35)[0],30,cmap="RdBu_r")
plt.colorbar()
plt.xlabel('X/R0')
plt.ylabel('Z/R0')
plt.title('x Component of Polodial Field [T] (y = 0)')
plt.savefig('Bx_iter_test.png')


sys.path.insert(1, '../../ode/python/')
import ode

# physical constants
qe = -1.60217662e-19
qp = 1.60217662e-19
me = 9.10938356e-31
mp = 1.6726219e-27
kB = 1.38064852e-23


# ion mass
Mi = 2*mp

# Charge-to-mass ratio (q/m)
qm = qp/Mi

def Efield(x,y,z):
    Ex = 0.0
    Ey = 0.0
    Ez = 0.0
    return Ex, Ey, Ez

# split Br_pol into x and y components
Bx_pol = Br_pol/np.sqrt(2)
By_pol = Br_pol/np.sqrt(2)

# making the values of X and Y, which are the same as R_35
X_35 = np.linspace(  0.6*R0, 1.4*R0, 100 )
Y_35 = np.linspace(  0.6*R0, 1.4*R0, 100 )

#uniform step sizes
h_X = (0.6*R0 + 0.6*R0)/(100-1)
h_Y = (0.6*R0 + 0.6*R0)/(100-1)

def B_tor(x,y,z):
    B0 = 5.3 #on-axis according to wikipedia
    # Major radius (projection)
    R = np.sqrt( x*x + y*y )
    # Sin, Cos of particle position on (x,y) plane
    ca = x/R
    sa = y/R
    # Toroidal component [T]
    Bphi = B0 * 6.2 / R
    # B-field [T]
    Bx = -Bphi * sa
    By =  Bphi * ca
    Bz =  0.0
    return Bx, By, Bz

# constructing the torodial components
Bx_tor = B_tor(X_35,Y_35,Z_35)[0]
By_tor = B_tor(X_35,Y_35,Z_35)[1]
Bz_tor = B_tor(X_35,Y_35,Z_35)[2]

Bx = Bx_pol + Bx_tor
By = By_pol + By_tor
Bz = Bz_pol + Bz_tor

plt.figure(4)
RR,ZZ = np.meshgrid(R_35/R0, Z_35/R0)
plt.contourf(np.transpose(RR),np.transpose(ZZ),Bx,30,cmap="RdBu_r")
plt.colorbar()
plt.xlabel('R/R0')
plt.ylabel('Z/R0')
plt.title('x Component of Polodial Field [T]')
plt.savefig('Bx_iter.png')

plt.figure(5)
RR,ZZ = np.meshgrid(R_35/R0, Z_35/R0)
plt.contourf(np.transpose(RR),np.transpose(ZZ),By,30,cmap="RdBu_r")
plt.colorbar()
plt.xlabel('R/R0')
plt.ylabel('Z/R0')
plt.title('y Component of Polodial Field [T]')
plt.savefig('By_iter.png')

plt.figure(6)
RR,ZZ = np.meshgrid(R_35/R0, Z_35/R0)
plt.contourf(np.transpose(RR),np.transpose(ZZ),Bz,30,cmap="RdBu_r")
plt.colorbar()
plt.xlabel('R/R0')
plt.ylabel('Z/R0')
plt.title('z Component of Polodial Field [T]')
plt.savefig('Bz_iter.png')



def fun(t,X):
   x, y, z, vx, vy, vz = X
   # E-field [V/m]
   Ex, Ey, Ez = Efield(x,y,z)
   # Newton-Lorentz equation in Cartesian coordinates
   Xdot = np.zeros(6)
   Xdot[0] = vx
   Xdot[1] = vy
   Xdot[2] = vz
   Xdot[3] = qm * ( Ex + vy*Bz - vz*By )
   Xdot[4] = qm * ( Ey + vz*Bx - vx*Bz )
   Xdot[5] = qm * ( Ez + vx*By - vy*Bx )
   return Xdot

# this is where I need to use functions...

def main():

    # Thermal speed
    T_eV = 10.0e3
    v_th = np.sqrt(2*kB*T_eV*11600/Mi)

    # Initial velocity [m/s]
    vx0 = 0.0
    vy0 = 0.0
    vz0 = v_th

    # Reference Magnetic Field
    Bx,By,Bz = Bfield(0.72,0,0)
    B0 = np.sqrt(Bx*Bx + By*By + Bz*Bz)

    # Larmor pulsation [rad/s]
    w_L = np.abs(qm * B0)

    # Larmor period [s]
    tau_L = 2.0*np.pi / w_L

    # Larmor radius [m]
    r_L = vy0 / w_L

    # Initial conditions
    X0 = np.array( [ 0.72, 0.0, 0.0, 0.0, vy0, vz0 ] )

    # Number of Larmor gyrations
    N_gyro = 200

    # Number of points per gyration
    N_points_per_gyration = 100

    # Time grid
    time = np.linspace( 0.0, tau_L*N_gyro, N_gyro*N_points_per_gyration )

    # Solve ODE (Runge-Kutta 4)
    X = ode.rk4( fun, time, X0 )

    # Get components of the state vector
    x  = X[:,0]
    y  = X[:,1]
    z  = X[:,2]
    vx = X[:,3]
    vy = X[:,4]
    vz = X[:,5]

    R = np.sqrt(x*x + y*y)

    plt.figure(4)
    plt.plot( x, y, 'b-', label='Runge-Kutta (4th)' )
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.axis('equal')
    plt.legend(loc=3)
    plt.savefig('ex02_drift_grad_B_trajectory.png')
    plt.show()

    plt.figure(5)
    plt.plot( R, z, 'b-')
    plt.xlabel('R, Radius [m]')
    plt.ylabel('Z, Vertical Coordinate [m]')
    plt.axis('equal')
    plt.savefig('ex02_drift_grad_B_vertical_drift.png')
    plt.show()


if __name__ == '__main__':
   main()


"""