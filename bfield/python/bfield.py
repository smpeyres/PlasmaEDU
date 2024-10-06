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


# def biotsavart( filament, current, point ):
#     Npoints = np.size(filament,1)
#     B = np.zeros((3,1))
#     for i in range(Npoints-1):
#         P1 = filament[:,i  ]
#         P2 = filament[:,i+1]
#         dl = P2 - P1
#         midpoint = 0.5 * (P1 + P2)
#         R  = np.transpose(point) - midpoint
#         Rm = np.sqrt( R[0,0]*R[0,0] + R[0,1]*R[0,1] + R[0,2]*R[0,2] )
#         R3 = Rm * Rm * Rm + 1.0e-12
#         dI = current * dl
#         dB = 1.0e-7 * np.cross(dI,R) / R3
#         B[0] += dB[0,0]
#         B[1] += dB[0,1]
#         B[2] += dB[0,2]
#     return B[0], B[1], B[2]

# def biotsavart(filament, I0, point):
#     """
#     Compute the B-field at a given point due to a current filament using the Biot-Savart law.
#     """
#     mu0 = 4 * np.pi * 1e-7  # Vacuum permeability
#     B = np.zeros((3, 1))     # Initialize B-field as a 3D vector

#     # Loop over each segment of the filament
#     for i in range(filament.shape[1] - 1):
#         # Segment of filament
#         dL = filament[:, i+1] - filament[:, i]

#         # Position vector from the filament to the point of interest
#         R = point[:, 0] - filament[:, i]

#         # Compute the cross product dL x R
#         dL_cross_R = np.cross(dL, R)

#         # Distance from the filament segment to the point
#         Rm = np.linalg.norm(R)  # Correct handling of R as a 1D vector

#         # Apply Biot-Savart law to compute dB
#         if Rm != 0:
#             dB = (mu0 * I0 / (4 * np.pi * Rm**3)) * dL_cross_R
#             B += dB[:, np.newaxis]  # Add dB to the total B-field

#     return B[0], B[1], B[2]

def biotsavart(filament, I0, point):
    """
    Compute the B-field at a given point due to a current filament using the Biot-Savart law.
    """
    mu0 = 4 * np.pi * 1e-7  # Vacuum permeability
    B = np.zeros((3, 1))     # Initialize B-field as a 3D vector

    # Ensure point is a 2D array with shape (3, 1)
    if point.ndim == 1:
        point = point[:, np.newaxis]  # Reshape to (3, 1)

    # Loop over each segment of the filament
    for i in range(filament.shape[1] - 1):
        # Segment of filament
        dL = filament[:, i+1] - filament[:, i]

        # Position vector from the filament to the point of interest
        R = point[:, 0] - filament[:, i]

        # Compute the cross product dL x R
        dL_cross_R = np.cross(dL, R)

        # Distance from the filament segment to the point
        Rm = np.linalg.norm(R)  # Correct handling of R as a 1D vector

        # Apply Biot-Savart law to compute dB
        if Rm != 0:
            dB = (mu0 * I0 / (4 * np.pi * Rm**3)) * dL_cross_R
            B += dB[:, np.newaxis]  # Add dB to the total B-field

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

def helix(x, y, z, I0, Ra, La, Nturns, Npoints, phi0=0.0, Center=np.array([0,0,0]), EulerAngles=np.array([0,0,0])):
    """
    Generate a helical filament representing a finite solenoid.

    Ra = radius of helix
    La = length of helix
    Nturns = number of turns of the helix
    Npoints = number of points resolved
    phi0 = azimuthal angle at which helix intersects XY plane
    """

    # generate primary parameter - azimuthal angle
    phi = np.linspace(-2*np.pi, 2*np.pi*(Nturns+1), Npoints)

    # parametric equations
    X = Ra * np.cos(phi - phi0)
    Y = Ra * np.sin(phi - phi0)
    Z = ((La/Nturns)/(2*np.pi))*(phi)

    # create filament
    filament_local = np.vstack((X, Y, Z))

    # rotate and translate filament, if necessary
    R = roto(EulerAngles)
    filament_rotated = R @ filament_local
    filament = filament_rotated + Center[:, np.newaxis]

    # Initialize the B-field components
    Bx = np.zeros((x.size, y.size, z.size))
    By = np.zeros((x.size, y.size, z.size))
    Bz = np.zeros((x.size, y.size, z.size))

    # compute the magnetic field at each grid point
    for i in range(len(x)):
        for j in range(len(y)):
            for k in range(len(z)):
                point = np.array([x[i], y[j], z[k]])
                Bx[i, j, k], By[i, j, k], Bz[i, j, k] = biotsavart(filament, I0, point)

    return Bx, By, Bz, filament

def undulator(x, y, z, I0, Ra, La, Nturns, Npoints, phi0=0.0, phase_shift=np.pi, Center1=np.array([0,0,0]), Center2=np.array([0,0,0]), EulerAngles1=np.array([0,0,0]), EulerAngles2=np.array([0,0,0])):
    """
    Generate a magnetic field for a helical undulator by combining two helical filaments.
    The function will call `helix` twice to simulate two oppositely directed helices and sum their magnetic fields.

    Parameters:
    - x, y, z: grid points
    - I0: current
    - Ra: radius of the helix
    - La: length of the helix
    - Nturns: number of turns of the helix
    - Npoints: number of points resolved
    - phi0: azimuthal angle
    - Center1, Center2: centers of the two helices
    - EulerAngles1, EulerAngles2: orientations (Euler angles) of the two helices
    """
    # First helix
    Bx1, By1, Bz1, filament1 = helix(x, y, z, I0, Ra, La, Nturns, Npoints, phi0, Center1, EulerAngles1)

    # Second helix -> flipped current
    Bx2, By2, Bz2, filament2 = helix(x, y, z, -1*I0, Ra, La, Nturns, Npoints, phi0 + phase_shift, Center2, EulerAngles2)

    # Sum the magnetic fields of both helices
    Bx = Bx1 + Bx2
    By = By1 + By2
    Bz = Bz1 + Bz2

    return Bx, By, Bz, filament1, filament2
