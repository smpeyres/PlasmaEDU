import numpy as np
import matplotlib.pylab as plt
import math
from inputdata import *
from scipy.optimize import curve_fit

# Variables stored from imput_params

# Load both Eulerian and pic_1e4 data
Enorm_eul = np.load('Enorm_linear_1.npy')
Enorm_eul = np.sqrt(Enorm_eul)

pic_1e4_data = np.load('Enorm_pic_1e4.npz')
Enorm_pic_1e4 = np.sqrt(pic_1e4_data['Enorm'])  # Apply same sqrt as Eulerian
time_pic_1e4 = pic_1e4_data['time']

pic_1e5_data = np.load('Enorm_pic_1e5.npz')
Enorm_pic_1e5 = np.sqrt(pic_1e5_data['Enorm'])  # Apply same sqrt as Eulerian
time_pic_1e5 = pic_1e5_data['time']

pic_1e6_data = np.load('Enorm_pic_1e6.npz')
Enorm_pic_1e6 = np.sqrt(pic_1e6_data['Enorm'])  # Apply same sqrt as Eulerian
time_pic_1e6 = pic_1e6_data['time']

### RECURRENCE TIME
tr = 2.0*np.pi / 0.5 / dV
tr2 = tr / 1.5
tr3 = tr / 2.0
print(tr)

xtr = np.zeros((100)) + tr
ytr = np.linspace(np.min(Enorm_eul),np.max(Enorm_eul),100)

xtr2 = np.zeros((100)) + tr2
ytr2 = np.linspace(np.min(Enorm_eul),np.max(Enorm_eul),100)

xtr3 = np.zeros((100)) + tr3
ytr3 = np.linspace(np.min(Enorm_eul),np.max(Enorm_eul),100)

### THEORETICAL ELECTRIC FIELD
Et = np.zeros((Nx,len(time_vector)))
Et_norm = np.zeros((len(time_vector)))
for i in range(Nx):
    Et[i,:] = 4.0*0.001 * 0.3677 * np.exp(-0.1533*time_vector)*np.sin(0.5*X[i])*np.cos(1.4156*time_vector - 0.5326245)

for n in range(len(time_vector)):
    Et_norm[n] = np.sqrt(np.trapz(Et[:,n]*Et[:,n],x=X))

### DAMPING RATE FITTING
# Take derivative of Enorm
dedx = np.zeros((len(Enorm_eul),1))
for i in range(1,len(Enorm_eul)-1):
    dedx[i] = -(Enorm_eul[i+1] - Enorm_eul[i-1]) / 2.0 / dt
dedx[0] = ( 3*Enorm_eul[0]   - 4*Enorm_eul[1]     + Enorm_eul[2]     ) / 2.0 / dt
dedx[-1]  = - ( 3*Enorm_eul[-1]  - 4*Enorm_eul[-2]    + Enorm_eul[-3]    ) / 2.0 / dt

# Find location of zeros (only the local maxima, not minima)
dedx_temp = np.zeros((len(Enorm_eul),1))
for i in range(1,len(dedx)-1):
    if (dedx[i]*dedx[i-1] < 0 and dedx[i-1]<0):
        dedx_temp[i] = i
dedx_0 = dedx_temp[dedx_temp != 0]



# Find values of maxima for least-squares fitting
x_max = np.zeros((len(dedx_0)))
Enorm_eul_max = np.zeros((len(dedx_0)))
for i in range(len(dedx_0)):
    # x values (time 'locations')
    x_max[i] = time_vector[int(dedx_0[i])]

    # y values (maxima)
    Enorm_eul_max[i] = Enorm_eul[int(dedx_0[i])]


# Theoretical Damping Rate (for linear Landau damping)
imw = np.sqrt(np.pi/8.0) * (1.0/(k**3.0)) * np.exp(-1.0/(2.0*(k**2.0)) - (3.0/2.0))
print('Theoretical damping rate = '+str(imw))


### LEAST SQUARES FIT TO MAXIMA :  y = a*exp(-c*t)
def func(x, a, c):
    return (a*np.exp(-c*x))

# c is the damping rate, a is the amplitude
popt, pcov = curve_fit(func, x_max[1:10], Enorm_eul_max[1:10])
print('Numerical damping rate, eulerian = '+str(popt))

# Separate maxima finding for PIC data
dedx_pic = np.zeros((len(Enorm_pic_1e6),1))
for i in range(1,len(Enorm_pic_1e6)-1):
    dedx_pic[i] = -(Enorm_pic_1e6[i+1] - Enorm_pic_1e6[i-1]) / 2.0 / dt
dedx_pic[0] = (3*Enorm_pic_1e6[0] - 4*Enorm_pic_1e6[1] + Enorm_pic_1e6[2]) / 2.0 / dt
dedx_pic[-1] = -(3*Enorm_pic_1e6[-1] - 4*Enorm_pic_1e6[-2] + Enorm_pic_1e6[-3]) / 2.0 / dt

# Find PIC maxima
dedx_pic_temp = np.zeros((len(Enorm_pic_1e6),1))
for i in range(1,len(dedx_pic)-1):
    if (dedx_pic[i]*dedx_pic[i-1] < 0 and dedx_pic[i-1]<0):
        dedx_pic_temp[i] = i
dedx_pic_0 = dedx_pic_temp[dedx_pic_temp != 0]

# Find values of PIC maxima for fitting
x_pic_max = np.zeros((len(dedx_pic_0)))
Enorm_pic_max = np.zeros((len(dedx_pic_0)))
for i in range(len(dedx_pic_0)):
    x_pic_max[i] = time_pic_1e6[int(dedx_pic_0[i])]
    Enorm_pic_max[i] = Enorm_pic_1e6[int(dedx_pic_0[i])]

# Fit PIC damping rate with its own maxima
popt_pic, pcov_pic = curve_fit(func, x_pic_max[1:10], Enorm_pic_max[1:10])
print('Numerical damping rate, pic = '+str(popt_pic))


xx = np.linspace(x_max[0],x_max[-1],100)
# xx = np.linspace(time_vector[1000],time_vector[3000],100)
yy = func(xx, *popt)



# E-norm needs epsilon_0 to be the correct units

# plt.figure()
# plt.semilogy(time_vector, Enorm_eul, linewidth=2, label='Eulerian')
# # plt.semilogy(time_pic_1e4, Enorm_pic_1e4, linewidth=2, label='PIC, $10^4$ Particles')
# # plt.semilogy(time_pic_1e5, Enorm_pic_1e5, linewidth=2, label='PIC, $10^5$ Particles')
# plt.semilogy(time_pic_1e6, Enorm_pic_1e6, linewidth=2, label='PIC, $10^6$ Particles')
# plt.semilogy(time_vector, func(time_vector, *popt), '--', linewidth=2, label='Numerical Damping Rate, Eulerian')
# plt.semilogy(time_vector, func(time_vector, *popt_pic), '--', linewidth=2, label='Numerical Damping Rate, PIC')
# plt.ylabel('Electric Field Energy')
# plt.xlabel('Time')
# plt.legend()
# plt.savefig('energy_comparison.png',dpi=600)
# plt.show()


# k = 0.5
# theoretical damping rate = 0.151387230212514
# eulerian damping rate = 0.16700997
# pic damping rate = 0.02438793

# k = 0.4
# theoretical damping rate = 0.09599265198249836
# eulerian = 0.0794255
# pic = 0.00755725

# k = 0.3
# theoretical damping rate = 0.02002061131206703
# eulerian = 0.02114066
# pic = 7.46597851e-04

k_array = np.array([0.3, 0.4, 0.5])
theoretical_damping_rate = np.array([0.02002061131206703, 0.09599265198249836, 0.151387230212514])
eulerian_damping_rate = np.array([0.02114066, 0.0794255, 0.16700997])
pic_damping_rate = np.array([7.46597851e-04, 0.00755725, 0.02438793])

plt.figure()
plt.plot(k_array, theoretical_damping_rate, marker='o', linestyle='-', label='Theoretical')
plt.plot(k_array, eulerian_damping_rate, marker='s', linestyle='-', label='Eulerian')
plt.plot(k_array, pic_damping_rate, marker='^', linestyle='-', label='PIC')
plt.xlabel('k')
plt.ylabel('Damping Rate')
plt.yscale('log')
plt.legend()
plt.savefig('damping_rate_comparison.png', dpi=600)
plt.show()

