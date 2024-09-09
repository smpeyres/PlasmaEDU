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

R0   = 6.2
# R, Z Grid -> estimating using graph on slide 35
R_35 = np.linspace(  0.6*R0, 1.4*R0, 100 )
Z_35 = np.linspace( -0.8*R0, 0.8*R0, 100 )

"""
psi_35 = psi_iter_like(R_35,Z_35)

Note that this usage is innappropriate,
as psi_iter_like does not accept vector inputs. 
Psi is evaluated at which point R,Z provided individually.

Let's try a point-wise approach instead.
"""

# create empty matrix that is size of R x Z
psi_35 = np.zeros((100,100))

# loop over all points and evaluate Psi at each point
for i in range(0,len(R_35)):
        for j in range(0,len(Z_35)):
                psi_35[i,j] = psi_iter_like(R_35[i],Z_35[j])

plt.figure(1)
RR,ZZ = np.meshgrid(R_35/R0,Z_35/R0)
plt.contourf(np.transpose(RR),np.transpose(ZZ),psi_35,30,cmap="RdBu_r")
plt.colorbar()
plt.xlabel('R/R0')
plt.ylabel('Z/R0')
plt.title('ITER Polodial Flux Function [Tm${}^2$]')
plt.savefig('iter_flux_function.png')

# create empty matrices, size of R x Z, for the components of B
B_R = np.zeros((100,100))
B_Z = np.zeros((100,100))

# uniform step sizes
h_Z = (0.8*R0 + 0.8*R0)/(100-1)
h_R = (0.6*R0 + 0.6*R0)/(100-1)


# bottom and top boundaries
for i in range(0,len(R_35)):
        B_R[i, 0]  = (1/R_35[i])*(-3*psi_35[i,0] + 4*psi_35[i,1] - psi_35[i,2])/(2*h_Z)
        B_R[i,-1] = (1/R_35[i])*(3*psi_35[i,-1] - 4*psi_35[i,-2] + psi_35[i,-3])/(2*h_Z)

# rest of domain
for i in range(0,len(R_35)):
        for j in range(1,len(Z_35)-1):
                B_R[i, j]  = (1/R_35[i])*(psi_35[i,j+1] - psi_35[i,j-1])/(2*h_Z)


# left and right boundaries

# left and right boundaries
for i in range(0,len(Z_35)):
        B_Z[0, i]  = -(1/R_35[0])*(-3*psi_35[0,i] + 4*psi_35[1,i] - psi_35[2,i])/(2*h_R)
        B_Z[-1,i] = -(1/R_35[-1])*(3*psi_35[-1,i] - 4*psi_35[-2,i] + psi_35[-3,i])/(2*h_R)

# rest of domain
for i in range(1,len(R_35)-1):
        for j in range(0,len(Z_35)):
                B_Z[i, j]  = -(1/R_35[i])*(psi_35[i+1,j] - psi_35[i-1,j])/(2*h_R)


plt.figure(2)
RR,ZZ = np.meshgrid(R_35/R0,Z_35/R0)
plt.contourf(np.transpose(RR),np.transpose(ZZ),B_R,30,cmap="RdBu_r")
plt.colorbar()
plt.xlabel('R/R0')
plt.ylabel('Z/R0')
plt.title('Radial Component of Polodial Field [T]')
plt.savefig('BR_polodial_iter.png')

plt.figure(3)
RR,ZZ = np.meshgrid(R_35/R0,Z_35/R0)
plt.contourf(np.transpose(RR),np.transpose(ZZ),B_Z,30,cmap="RdBu_r")
plt.colorbar()
plt.xlabel('R/R0')
plt.ylabel('Z/R0')
plt.title('Axial Component of Polodial Field [T]')
plt.savefig('BZ_polodial_iter.png')
