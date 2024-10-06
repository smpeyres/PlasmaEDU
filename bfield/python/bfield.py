import numpy as np
import scipy.special as special


def loopbrz( Ra, I0, Nturns, R, Z ):
    # Input
    #     Ra [m]      Loop radius
    #     I0 [A]      Loop current
    #     Nturns      Loop number of turns (windings)
    #     R  [m]      Radial coordinate of the point
    #     Z  [m]      Axial  coordinate of the point
    # Output
    #     Br, Bz [T]  Radial and Axial components of B-field at (R,Z)
    #
    # (Note that singularities are not handled here)
    mu0   = 4.0e-7 * np.pi
    B0    = mu0/2.0/Ra * I0 * Nturns
    alfa  = np.absolute(R)/Ra
    beta  = Z/Ra
    gamma = (Z+1.0e-10)/(R+1.0e-10)
    Q     = (1+alfa)**2 + beta**2
    ksq   = 4.0 * alfa / Q
    asq   = alfa * alfa
    bsq   = beta * beta
    Qsp   = 1.0/np.pi/np.sqrt(Q)
    K     = special.ellipk(ksq)
    E     = special.ellipe(ksq)
    Br    = gamma * B0*Qsp * ( E * (1+asq+bsq)/(Q-4.0*alfa) - K )
    Bz    =         B0*Qsp * ( E * (1-asq-bsq)/(Q-4.0*alfa) + K )
    return Br, Bz


def roto(EulerAngles):
    # Classic (proper) Euler Angles (p,t,f)
    # with Z-X-Z rotation sequence:
    # (psi,z), (theta,x), (phi,z)
    # p=psi, t=theta, f=phi angles in [rad]
    p=EulerAngles[0]
    t=EulerAngles[1]
    f=EulerAngles[2]
    sp=np.sin(p)
    st=np.sin(t)
    sf=np.sin(f)
    cp=np.cos(p)
    ct=np.cos(t)
    cf=np.cos(f)
    R=np.array([[cf*cp-sf*ct*sp,-sf*cp-sp*ct*cf,st*sp], \
                [cf*sp+sf*ct*cp,-sf*sp+cf*ct*cp,-st*cp],\
                [st*sf,st*cf,ct]])
    return R

def loopxyz( Ra, I0, Nturns, Center, EulerAngles, Point ):
    # This function returns the 3 Cartesian components B=(Bx,By,Bz) of the
    # magnetic field at the point Point(Xp,Yp,Zp) generated by a current loop
    # arbitrarily oriented in the 3D space.
    #
    # Input
    #           Ra  :  Loop Radius [m]
    #           I0  :  Loop Current [A]
    #       Nturns  :  Loop number of turns
    #
    # Center = (Xc, Yc, Zc)  :  Position [m] of the Center of the Current Loop,
    #                            expressed in the LAB Reference Frame
    #
    # EulerAngles = (p,t,f)   :  Orientation of the Current Loop, given by the
    #                            three Euler Angles phi, theta, phi
    #                            expressed w.r.t. the LAB Reference Frame;
    #
    # Point = (Xp, Yp, Zp)   : Position of interest, defined as the
    #                           segment OP, where O is the origin of the
    #                           LAB Reference Frame, and P the point where
    #                           the magnetic field has to be found
    # Output
    #
    #     Magnetic field vector expressed in the LAB Reference Frame
    #
    #     Bx       X component of the B-field generated by the Current Loop
    #     By       Y component of the B-field generated by the Current Loop
    #     Bz       Z component of the B-field generated by the Current Loop
    #
    # Rotation matrix from LAB Reference Frame to LOOP Reference Frame
    ROT_LAB_LOOP = roto(EulerAngles)
    # Roto-traslation of the point P into the LOOP reference frame
    P_LOOP = ROT_LAB_LOOP.dot( Point - Center )
    R = P_LOOP[0]
    Z = P_LOOP[1]
    # Magnetic field in the LOOP Reference Frame
    Br, Bz = loopbrz( Ra, I0, Nturns, R, Z )
    B_LOOP = np.array([Br,Bz,0])
    # Rotate the magnetic field from LOOP Reference Frame to LAB Reference Frame
    ROT_LOOP_LAB = np.transpose(ROT_LAB_LOOP)
    B_LAB  = ROT_LOOP_LAB.dot(B_LOOP)
    Bx = B_LAB[0]
    By = B_LAB[1]
    Bz = B_LAB[2]
    return Bx, By, Bz


def makeloop( Ra, Center, EulerAngles, Npoints ):
    # Construct the geometrical points of a loop
    #
    # Input
    #
    # Ra  :  Loop Radius [m]
    #
    # Center = (Xc, Yc, Zc)  :  Position [m] of the Center of the Current Loop,
    #                            expressed in the LAB Reference Frame
    #
    # EulerAngles = (p,t,f)   :  Orientation of the Current Loop, given by the
    #                            three Euler Angles phi, theta, phi
    #                            expressed w.r.t. the LAB Reference Frame;
    #
    # Npoint : Number of discrete points
    #
    # Output
    #
    # CurrentFilament   :  (3 x Npoint) Array containing the 3D coordinates of
    #                      the points of the current loop
    CurrentFilament = np.zeros((3,Npoints))
    # Rotation matrix from LAB Reference Frame to LOOP Reference Frame
    ROT_LAB_LOOP = roto(EulerAngles)
    # Rotation matrix from LOOP Reference Frame to LAB Reference Frame
    ROT_LOOP_LAB = np.transpose(ROT_LAB_LOOP)
    # Construct the coordinates of the Loop
    P_LOOP   = np.zeros((3,1))
    phi = np.linspace(0.0, 2.0*np.pi, Npoints)
    for i in range(0,Npoints):
        P_LOOP[0] =  Ra * np.cos( phi[i] )
        P_LOOP[1] =  0.0
        P_LOOP[2] = -Ra * np.sin( phi[i] )
        P_LAB = ROT_LOOP_LAB.dot( P_LOOP )
        CurrentFilament[0][i] = P_LAB[0] + Center[0]
        CurrentFilament[1][i] = P_LAB[1] + Center[1]
        CurrentFilament[2][i] = P_LAB[2] + Center[2]
    return CurrentFilament


def biotsavart( filament, current, point ):
    Npoints = np.size(filament,1)
    B = np.zeros((3,1))
    for i in range(Npoints-1):
        P1 = filament[:,i  ]
        P2 = filament[:,i+1]
        dl = P2 - P1
        midpoint = 0.5 * (P1 + P2)
        R  = np.transpose(point) - midpoint
        Rm = np.sqrt( R[0,0]*R[0,0] + R[0,1]*R[0,1] + R[0,2]*R[0,2] )
        R3 = Rm * Rm * Rm + 1.0e-12
        dI = current * dl
        dB = 1.0e-7 * np.cross(dI,R) / R3
        B[0] += dB[0,0]
        B[1] += dB[0,1]
        B[2] += dB[0,2]
    return B[0], B[1], B[2]


def blines(y,x, filament, current):
    X=y[0]
    Y=y[1]
    Z=y[2]
    direction=y[3]
    point = np.array([ [X], [Y], [Z] ])
    B     = biotsavart( filament, current, point )
    Bnorm = np.sqrt(B[0]*B[0] + B[1]*B[1] + B[2]*B[2])
    dY    = np.zeros(4)
    dY[0] = direction * B[0]/Bnorm
    dY[1] = direction * B[1]/Bnorm
    dY[2] = direction * B[2]/Bnorm
    dY[3] = 0.0
    return dY

def helix(x, y, z, Ra, La, Nturns, Npoints, phi0=0.0, Center=np.array([0,0,0]), EulerAngles=np.array([0,0,0])):
    """
    Generate a helical filament representing a finite solenoid.

    Ra = radius of helix
    La = length of helix
    Nturns = number of turns of the helix
    Npoints = number of points resolved
    phi0 = azimuthal angle at which helix intersects XY plane
    """

    # generate primary parameter - azimuthal angle
    phi = np.linspace(0, 2*np.pi*Nturns, Npoints)

    # parametric equations
    X = Ra * np.cos(phi)
    Y = Ra * np.sin(phi)
    Z = ((La/Nturns)/(2*np.pi))*(phi - phi0)

    # create filament
    filament_local = np.vstack((X, Y, Z))

    # rotate and translate filament, if necessary
    R = bfield.roto(EulerAngles)
    filament_rotated = R @ filament_local
    filament = filament_rotated + Center[:, np.newaxis]

    # Initialize the B-field components
    Bx = np.zeros((X_grid.size, Y_grid.size, Z_grid.size))
    By = np.zeros((X_grid.size, Y_grid.size, Z_grid.size))
    Bz = np.zeros((X_grid.size, Y_grid.size, Z_grid.size))

    # compute the magnetic field at each grid point
    for i in range(len(x)):
        for j in range(len(y)):
            for k in range(len(z)):
                point = np.array(x[i], y[j], z[k])
                Bfield = bfield.biotsavart(filament, I0, point)
                Bx[i, j, k] = Bfield[0]
                By[i, j, k] = Bfield[1]
                Bz[i, j, k] = Bfield[2]

    return Bx, By, Bz


def inf_helix_Hagel(Ra,La,I0, phi0, x, y, z):
    """
    Computes the magnetic field at a given point generated by an infinite helical wire
    Equations 21-28 in Hagel et al (1994, IEE Trans Magnetics) "On the Magnetic Field of an Infinitely Long Helical Line Current"

    Parameters:
    - Ra   : Loop radius [m] (a in the paper)
    - La   : Axial separation between turns/pitch distance of the helix [m] (p in the paper)
    - I0   : Current in the wire [A] (I in the paper)
    - phi0 : phi coordinate of the point where the helix intersets the plane z = 0 [rad]
    - x    : x-coordinate [m]
    - y    : y-coordinate [m]
    - z    : z-coordinate [m]

    Note: There seem to be some inconsistencies within the paper, particularly some potential sign errors, that make replication difficult.
    I am particularly stuck on replicating Figure 7.
    """

    # convert cartesian to cylindrical coordinates
    r = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    # z = z stays the same

    # Constants and other parameters
    mu0 = 4 * np.pi * 1e-7  # Magnetic permeability of free space
    Omega = 2 * np.pi / La

    # Equation 21 - first three terms
    # Note : potential sign error in paper, switched sign in front
    Br_outside = (mu0*I0/np.pi)*Ra*(Omega**2)*( special.ivp(1, Ra*Omega)*special.kvp(1, r*Omega)*np.sin(phi0 - phi + Omega*z)
                                              + 2*special.ivp(2, 2*Ra*Omega)*special.kvp(2, 2*r*Omega)*np.sin(2*(phi0 - phi + Omega*z))
                                              + 3*special.ivp(3, 3*Ra*Omega)*special.kvp(3, 3*r*Omega)*np.sin(3*(phi0 - phi + Omega*z)))

    # Equation 22 - first three terms
    # Note : potential sign error in paper, switched sign of second term
    Bphi_outside = mu0*I0/(2*np.pi*r) - (mu0*I0/np.pi)*(Ra/r)*Omega*( special.ivp(1, Ra*Omega)*special.kv(1, r*Omega)*np.cos(phi0 - phi + Omega*z)
                                                              + 2*special.ivp(2, 2*Ra*Omega)*special.kv(2, 2*r*Omega)*np.cos(2*(phi0 - phi + Omega*z))
                                                              + 3*special.ivp(3, 3*Ra*Omega)*special.kv(3, 3*r*Omega)*np.cos(3*(phi0 - phi + Omega*z)))


    # Equation 23
    # bote : potential sign error in paper, switched sign in front
    Bz_outside = (mu0*I0*Ra*(Omega**2)/np.pi)*( special.ivp(1, Ra*Omega)*special.kv(1, r*Omega)*np.cos(phi0 - phi + Omega*z)
                                                              + 2*special.ivp(2, 2*Ra*Omega)*special.kv(2, 2*r*Omega)*np.cos(2*(phi0 - phi + Omega*z))
                                                              + 3*special.ivp(3, 3*Ra*Omega)*special.kv(3, 3*r*Omega)*np.cos(3*(phi0 - phi + Omega*z)))


    # Equation 25 - first two terms
    # Note : potential sign error in paper, switched sign in front
    Br_inside = (mu0*I0/np.pi)*Ra*(Omega**2)*( special.ivp(1, r*Omega)*special.kvp(1, Ra*Omega)*np.sin(phi0 - phi + Omega*z)
                                              + 2*special.ivp(2, 2*r*Omega)*special.kvp(2, 2*Ra*Omega)*np.sin(2*(phi0 - phi + Omega*z))
                                              + 3*special.ivp(3, 3*r*Omega)*special.kvp(3, 3*Ra*Omega)*np.sin(3*(phi0 - phi + Omega*z)))

    # Equation 26 - first two terms
    Bphi_inside = (mu0*I0/np.pi)*(Ra/r)*Omega*( special.iv(1, r*Omega)*special.kvp(1, Ra*Omega)*np.cos(phi0 - phi + Omega*z)
                                                + 2*special.iv(2, 2*r*Omega)*special.kvp(2, 2*Ra*Omega)*np.cos(2*(phi0 - phi + Omega*z))
                                                + 3*special.iv(3, 3*r*Omega)*special.kvp(3, 3*Ra*Omega)*np.cos(3*(phi0 - phi + Omega*z)))

    # Equation 27
    # Note : sign errors?
    Bz_inside = (mu0*I0*Omega/(2*np.pi)) - (mu0*I0*Ra*(Omega**2)/np.pi)*( special.iv(1, r*Omega)*special.kvp(1, Ra*Omega)*np.cos(phi0 - phi + Omega*z)
                                                              + 2*special.iv(2, 2*r*Omega)*special.kvp(2, 2*Ra*Omega)*np.cos(2*(phi0 - phi + Omega*z))
                                                              + 3*special.iv(3, 3*r*Omega)*special.kvp(3, 3*Ra*Omega)*np.cos(3*(phi0 - phi + Omega*z)))

    B_outside = np.array([Br_outside, Bphi_outside, Bz_outside])
    B_inside = np.array([Br_inside, Bphi_inside, Bz_inside])

    return np.where(r > Ra, B_outside, B_inside)

def inf_helix_Tominaka(Ra,La,I0, phi0, x, y, z):
    """
    Computes the magnetic field at a given point generated by an infinite helical wire
    Tominaka (2006, Eur J Phys) "Magnetic field calculation of an infinitely long solenoid"
    Tominaka (2014, J Cryo Supercon Soc Japan) "Magnetic Field, Vector Potential and Inductances of Long Helical Conductors"

    Parameters:
    - Ra   : Loop radius [m] (a in the paper)
    - La   : Axial separation between turns/pitch distance of the helix [m] (Lp in the paper)
    - I0   : Current in the wire [A] (I in the paper)
    - phi0 : phi coordinate of the point where the helix intersets the plane z = 0 [rad]
    - x    : x-coordinate [m]
    - y    : y-coordinate [m]
    - z    : z-coordinate [m]

    Note : I am using phi instead of theta

    First two terms of each component only

    """
    # convert cartesian to cylindrical coordinates
    r = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    # z = z stays the same

    # Constants and other parameters
    mu0 = 4 * np.pi * 1e-7  # Magnetic permeability of free space
    k = 2 * np.pi / La

    # Eq 16 - a
    Br_inside = (mu0*I0/np.pi)*(k*k*Ra)*( 1*special.kvp(1, 1*k*Ra)*special.ivp(1, 1*k*r)*np.sin(1*(phi - phi0 - k*z))
                                         +  2*special.kvp(2, 2*k*Ra)*special.ivp(2, 2*k*r)*np.sin(2*(phi - phi0 - k*z))
                                          +  3*special.kvp(3, 3*k*Ra)*special.ivp(3, 3*k*r)*np.sin(3*(phi - phi0 - k*z))   )


    # Eq 16 - b
    Bphi_inside = (mu0*I0/np.pi)*(k*Ra/r)*( 1*special.kvp(1, 1*k*Ra)*special.iv(1,1*k*r)*np.cos(1*(phi - phi0 - k*z))
                                            + 2*special.kvp(2, 2*k*Ra)*special.iv(2,2*k*r)*np.cos(2*(phi - phi0 - k*z))
                                             + 3*special.kvp(3, 3*k*Ra)*special.iv(3,3*k*r)*np.cos(3*(phi - phi0 - k*z))      )

    # Eq 16 -c
    Bz_inside = (mu0*I0/(2*np.pi))*k - (mu0*I0/np.pi)*(k*k*Ra)*( 1*special.kvp(1, 1*k*Ra)*special.iv(1,1*k*r)*np.cos(1*(phi - phi0 - k*z))
                                            + 2*special.kvp(2, 2*k*Ra)*special.iv(2,2*k*r)*np.cos(2*(phi - phi0 - k*z))
                                            + 3*special.kvp(3, 3*k*Ra)*special.iv(3,3*k*r)*np.cos(3*(phi - phi0 - k*z))    )
    # Eq 17 - a
    Br_outside = (mu0*I0/np.pi)*(k*k*Ra)*( 1*special.ivp(1, 1*k*Ra)*special.kvp(1, 1*k*r)*np.sin(1*(phi - phi0 - k*z))
                                         +  2*special.ivp(2, 2*k*Ra)*special.kvp(2, 2*k*r)*np.sin(2*(phi - phi0 - k*z))
                                        + 3*special.ivp(3, 3*k*Ra)*special.kvp(3, 3*k*r)*np.sin(3*(phi - phi0 - k*z)) )
    # Eq 17 - b
    Bphi_outside = (mu0*I0/(2*np.pi*r)) + (mu0*I0/np.pi)*(k*Ra/r)*( 1*special.ivp(1, 1*k*Ra)*special.kv(1,1*k*r)*np.cos(1*(phi - phi0 - k*z))
                                            + 2*special.ivp(2, 2*k*Ra)*special.kv(2,2*k*r)*np.cos(2*(phi - phi0 - k*z))
                                                + 3*special.ivp(3, 3*k*Ra)*special.kv(3,3*k*r)*np.cos(3*(phi - phi0 - k*z))    )
    # Eq 17 - c
    Bz_outside = -(mu0*I0/np.pi)*(k*k*Ra)*( 1*special.ivp(1, 1*k*Ra)*special.kv(1,1*k*r)*np.cos(1*(phi - phi0 - k*z))
                                            + 2*special.ivp(2, 2*k*Ra)*special.kv(2,2*k*r)*np.cos(2*(phi - phi0 - k*z))
                                        + 3*special.ivp(3, 3*k*Ra)*special.kv(3,3*k*r)*np.cos(3*(phi - phi0 - k*z))    )

    B_inside_cyl = np.array([Br_inside, Bphi_inside, Bz_inside])
    B_outside_cyl = np.array([Br_outside, Bphi_outside, Bz_outside])

    # convert to cartesian

    # \hat{x} = cos(phi) \hat{r} - sin(phi) \hat{phi}
    # \hat{y} = sin(phi) \hat{r} + cos(phi) \hat{phi}
    # \hat{z} = \hat{z}

    B_inside_cart = B_inside_cyl
    B_outside_cart = B_inside_cyl

    B_inside_cart[0] = np.cos(phi)*B_inside_cyl[0] - np.sin(phi)*B_inside_cyl[1]
    B_inside_cart[1] = np.sin(phi)*B_inside_cyl[0] + np.cos(phi)*B_inside_cyl[1]
    B_inside_cart[2] = B_inside_cyl[2]

    B_outside_cart[0] = np.cos(phi)*B_outside_cyl[0] - np.sin(phi)*B_outside_cyl[1]
    B_outside_cart[1] = np.sin(phi)*B_outside_cyl[0] + np.cos(phi)*B_outside_cyl[1]
    B_outside_cart[2] = B_outside_cyl[2]

    return np.where(r < Ra, B_inside_cart, B_outside_cart)

