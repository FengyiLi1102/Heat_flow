import numpy as np
import imageio
import math as math
import scipy.special as special
import matplotlib.pylab as plb
import matplotlib.pyplot as plt
import matplotlib.image as mimage
from dynamic_plot import dyna_plot as dp
import gc

##############################################################################
############################## ALL IN SI UNIT ################################
##############################################################################
# Set globel parameters for drawing
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (10, 8),
          'axes.labelsize': 20,
          'axes.titlesize': 20,
          'xtick.labelsize':'x-large',
          'ytick.labelsize':'x-large'}
plb.rcParams.update(params)
# Globel constants
h_g, h_A = 2, 2
k_c = 2.5
k_s = 16.3
r_s_i, r_c_o = 0.027, 0.027
r_s_o = 0.028
r_c_i = 0.026
L1 = 0.09       # 1 for the section before the catalyst inserted
L2 = 0.01       # 2 for the section of the pipe with the catalyst
L = L1 + L2
Q_out_1 = 47.5  # W / m
Q_out_2 = 28.04
Q_abs_c = Q_out_1 - Q_out_2
Q_tot = 13919.7
rho_c = 2300
rho_s = 8030
C_c = 900
# Heat capacity of the steel is temperature dependent
# C_s = A + B * T
A = 450
B = 0.28
v_g = 21.8
t_tot_1 = 4.122 * (10**(-3))
t_tot_2 = 4.587 * (10**(-4))
t_tot = L / v_g
T_g = 573.15
T_A = 298.15
m_s_1 = rho_s * L1 * math.pi * (r_s_o**2 - r_s_i**2)
V_c = 6598.06 * (10**(-9))
m_c = rho_c * V_c
# Thermal diffusivity of the steel is temperature dependent
# alpha_s = k_s / (rho_s*C_s)
alpha_c = k_c / (rho_c*C_c)         # Thermal diffusivity
r_c_e = math.sqrt(V_c / (math.pi * L2))

def main(TYPE):
    """
    Coordinates: (x, y, z) in 3D; (x, z) in 2D
    Z axis: along the length (0: Gas entrance, 0.1: Gas exit)
    X axis: along the radius (0: center of the cylinder)
    Y axis: along the change of the angle from the center 
            of the cylindrical coordinate (phi)

    Settings:
    Set an infinitesimal time dt to obtain the respective displacement dz 
    along the Z direction. Due to the high symmetry of the pipe shape, we
    can only analyse the model in the Z and X axis without considering Y
    axis.

    Assumptions:
    >> Ignore the heat transfer along the Z axis from one block to its 
       neighbours when the time interval is small enough during the 
       iteration.

    >> Assume the heat transfer along the X axis can be obtained by 1D 
       transient conduction.

    >> Assume an equivalent thickness generated from the catalyst grids,
       contributed to the catalyst sleeve.
    """ 
    Nt1 = Nz                      # Number of time intervals
    Nt_critial = int((L1/L)*Nt1)  # When the gas reaches the catalyst
    dz = L / Nz                  # Real block length
    dt = dz / v_g                # Real time interval to pass one disk
    Nt2 = int((t_total - t_tot) / dt)
    
    Q_abs_1 = Q_tot - Q_out_1    # Absorbed heat flux for the steel before
                                 # reaching the cordierite

    Q_abs_2 = Q_tot - Q_out_2    # Absorbed heat flux for the steel and the
                                 # catalyst

    dQ = dz * Q_abs_1 / Nrec     # Power for a unit block

    block_t = np.zeros((2, 1))  # The 0th row repersents the unit block at 
                                 # the z = 0. The value means the total time 
                                 # that have passed at this unit block, which 
                                 # is the local time for heating the block.
                                 # Here only Nz number of rows are created due
                                 # to the symmetry through the pipe.
    
    t_constant = np.ones((1, 1))      # For constant iteration
    block_T_s = np.full((1, 1), T_A)  # Inital temperature for the pipe
    block_T_c = np.full((1, 1), T_A)

    block_T_s_mr = []            # Store temperature of each block during the
                                 # same iteration
    
    block_T_c_mr = []
    Nz_c = int(Nz-Nt_critial)
    m_c_unit = m_c / Nz_c

##############################################################################
    ######### Simulate the model in 2D based on the symmetry #############
    ########################## 0 <= z < 0.09 #############################
    # Analyse the temperature change along the Z axis at a given thickness 
    # to the pipe from inner to the outermost surface.
    if TYPE == 'Z': 
        # Until the gas reaches the end 
        for i in np.arange(Nt1):
            # When the gas has not reached the catalyst
            # Timer for timing the time
            t_increment = np.concatenate(
                (np.ones((int(i), 1)), np.zeros((int(Nz-i), 1))), axis=0)
            block_t += t_constant * dt         # Real time passed on each block

            C_s = A + B * (block_T_s + 273.15)    # Heat capacity
            alpha_s = k_s / (rho_s*C_s)         # Thermal diffusivity

            # Apply 1D transient conduction to the block along X axis 
            block_T_s = special.erf(thickness / (2*np.sqrt(alpha_s*block_t[0]))) * (T_A - T_g) + T_g
            block_T_s_mr.append(block_T_s)

            if i >= Nt_critial:
                # Catalyst starts to be heated
                block_T_c = special.erf(0.5*r_c_e / (2*np.sqrt(alpha_c*block_t[1]))) * (T_A - T_g) + T_g
                block_T_c_mr.append(block_T_c)
            
            else:
                block_T_c_mr.append(block_T_c)


        # Fully fill the pipe
        for i in np.arange(Nt2):
            block_t += t_constant * dt

            C_s = A + B * (block_T_s + 273.15)    # Heat capacity
            alpha_s = k_s / (rho_s*C_s)         # Thermal diffusivity

            # Apply 1D transient conduction to the block along X axis 
            block_T_s = special.erf(thickness / (2*np.sqrt(alpha_s*block_t[0]))) * (T_A - T_g) + T_g
            block_T_s_mr.append(block_T_s)

            # Catalyst starts to be heated
            block_T_c = special.erf(0.5*r_c_e / (2*np.sqrt(alpha_c*block_t[1]))) * (T_A - T_g) + T_g
            block_T_c_mr.append(block_T_c)


    # For plot the dynamic change of the temperature along the Z axis at a given
    # thickness. This is plotted for a temperature in 1D, but it can be rotated
    # for 360 degree for forming a lateral face temperature change plot, which will
    # be easier to achieve in SOLIDWORKS.
    name = 'Outermost surface'
    fig = plt.figure()
    ax = fig.add_subplot(111)
    block_T_s_mr = np.asarray(block_T_s_mr)
    ax.plot([i*dt for i in np.arange(len(block_T_s_mr))], block_T_s_mr[:, 0], 'r-')
    fig.savefig('dd.png')
    plt.close(fig)
    gc.collect()

    name = 'Catalyst'
    fig = plt.figure()
    ax = fig.add_subplot(111)
    block_T_c_mr = np.asarray(block_T_c_mr)
    ax.plot([i*dt for i in np.arange(len(block_T_c_mr))], block_T_c_mr[:, 0], 'r-')
    fig.savefig('ddd.png')
    plt.close(fig)
    gc.collect()
    # dp(Nz, dz, dt, block_T_s_mr, name, thickness)

    # Plot the catalyst temperature along the Z axis
    # name = 'Catalyst'
    # dp(Nz_c, dz, dt, block_T_c_mr, name)


if __name__ == '__main__':
    t_total = 50                  # Total time for the simulation
    thickness = 0.001            # Thickness from the inner surface
    Nz = 200                      # Number of disks aligned vertically
    Nrec = 1000                  # Number of rectanglar blocks for each disk

    main('Z')