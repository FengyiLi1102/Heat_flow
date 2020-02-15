import numpy as np
import tqdm as tqdm
import imageio
import math as math
import scipy.special as special
import matplotlib.pylab as plb
import matplotlib.pyplot as plt
import matplotlib.image as mimage
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


# Set globel parameters for drawing
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (10, 8),
          'axes.labelsize': 20,
          'axes.titlesize': 20,
          'xtick.labelsize':'x-large',
          'ytick.labelsize':'x-large'}
plb.rcParams.update(params)
##############################################################################
############################## ALL IN SI UNIT ################################
##############################################################################

# Globel constants
h_g, h_A = 2, 2
k_c = 2.5
k_s = 16.3
r_s_i, r_c_o = 0.027, 0.027
r_s_o = 0.028
r_c_i = 0.026
L1 = 0.09       # 1 for the section before the catalyst inserted
L2 = 0.01       # 2 for the section of the pipe with the catalyst
Q_out_1 = 47.5  # W / m
Q_out_2 = 46.7
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
T_g = 573.15
T_A = 298.15
m_s_1 = rho_s * L1 * math.pi * (r_s_o**2 - r_s_i**2)
# Thermal diffusivity of the steel is temperature dependent
# alpha_s = k_s / (rho_s*C_s)


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

    >> 
    """
##############################################################################
    ######### Simulate the model in 2D based on the symmetry #############
    ########################## 0 <= z < 0.09 #############################
    Nz = 10                    # Number of disks aligned vertical
    Nrec = 1000                  # Number of rectanglar blocks for each disk
    thickness = 0.001            # Thickness from the inner surface 
    Nt = Nz                      # Number of time intervals
    dz = L1 / Nz                 # Real block length
    dt = dz / v_g                # Real time interval to pass one disk
    
    Q_abs_1 = Q_tot - Q_out_1    # Absorbed heat flux for the steel before
                                 # reaching the cordierite

    dQ = dz * Q_abs_1 / Nrec     # Power for a unit block

    block_t = np.zeros((Nz, 1))  # The 0th row repersents the unit block at 
                                 # the z = 0. The value means the total time 
                                 # that have passed at this unit block, which 
                                 # is the local time for heating the block.
                                 # Here only Nz number of rows are created due
                                 # to the symmetry through the pipe.
    
    block_T = np.full((Nz, 1), T_A)  # Inital temperature for the pipe

    temp_time = []               # For dynamic plotting the temperature change
                                 # along the x axis

    ti = 0                       # Initial time
    block_T_mr = []              # Store temperature of each block during the
                                 # same iteration

    # Analyse the temperature change along the Z axis at a given thickness 
    # to the pipe from inner to the outermost surface.
    if TYPE == 'Z': 
        # Iteration 
        for i in np.arange(Nt):
            # Timer for timing the time
            t_increment = np.concatenate(
                (np.ones((int(i), 1)), np.zeros((int(Nz-i), 1))), axis=0)
            block_t += t_increment * dt         # Real time passed on each block

            C_s = A + B * (block_T + 273.15)    # Heat capacity
            alpha_s = k_s / (rho_s*C_s)         # Thermal diffusivity

            # Apply 1D transient conduction to the block along X axis 
            block_T = special.erf(thickness / (2*np.sqrt(alpha_s*block_t))) * (T_A - T_g) + T_g
            block_T_mr.append(block_T)

##############################################################################
     ######### Simulate the model in 2D based on the symmetry #############
     ######################## 0.09 < z <= 0.1 #############################
    

    # For plot the dynamic change of the temperature along the Z axis at a given
    # thickness. This is plotted for a temperature in 1D, but it can be rotated
    # for 360 degree for forming a lateral face temperature change plot, which will
    # be easier to achieve in SOLIDWORKS.
    images = []

    
    xZ = np.array([i for i in np.arange(Nz)]) * dz * 1000
    plt.ion()

    i = 0
    for tem in block_T_mr:
        print("-------{}-------".format(i))
        fig = plt.figure(dpi=300)
        ax = fig.add_subplot(111)
        ax.plot(xZ, tem, 'r-', linewidth=1.2)
        ax.set_xlabel("Length along the Z axis (mm)")
        ax.set_ylabel("Temperature (K)")
        ax.set_ylim(273.15, 600)
        fig.savefig('Present.png')
        images.append(imageio.imread('Present.png'))
        i += 1
    imageio.mimsave('T_zX_{}.gif'.format(thickness), images, duration=0.5)


if __name__ == '__main__':
    main('Z')