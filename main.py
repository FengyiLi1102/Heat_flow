import numpy as np
import matplotlib.pylab as plb
import matplotlib.pyplot as plt

##############################################################################
# Globel constants
# ALL IN SI UNIT
h_g, h_A = 2, 2
k_c = 2.5
k_s = 16.3
r_s_i, r_c_o = 0.027, 0.027
r_s_o = 0.028
r_c_i = 0.026
L1 = 0.09
L2 = 0.01
Q_out_1 = 47.5  # W / m
Q_out_2 = 46.7
Q_tot = 13919.7
rho_c = 2300
rho_s = 8030
C_c = 900
# C_s is temperature dependent
# C_s = 450 + 0.28 * T
v_g = 21.8
t_tot_1 = 4.122 * (10**(-3))
t_tot_2 = 4.587 * (10**(-4))
T_g = 573.15
T_A = 298.15


def main():
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
    >> Ignore the heat transfer along the Z axis when the time interval is
       small enough during the iteration.

    >> Assume the heat transfer along the X axis can be obtained by 1D 
       transient conduction.

    >> 
    """
    ######### Simulate the model in 2D based on the symmetry #############
    ########################## 0 <= z < 0.09 #############################
    Nz = 1000                   # Number of rectangular blocks aligned vertical                     
    dz = L1 / Nz                # Real block length
    dt = dz / v_g               n# Real time interval
    
    Q_abs_1 = Q_tot - Q_out_1   # Absorbed heat flux for the steel before
                                # reaching the cordierite
    dQ = dz * Q_abs_1           # Power for a unit disk

    block_t = np.zeros((Nz, 1))  # The 0th row repersents the unit block at 
                                 # the z = 0. The value means the time passed
                                 # at this unit block, which is the local time
                                 # to heat the material.
    # Iteration 
    for i in np.arange(Nt):
         t_increment = np.concatenate(np.ones((i)), np.zeros((Nz)))

    
