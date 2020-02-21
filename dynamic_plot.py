import numpy as np
import imageio
import matplotlib.image as mimage
import matplotlib.pylab as plb
import matplotlib.pyplot as plt
import gc
from tqdm import trange

# Set globel parameters for drawing
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (10, 8),
          'axes.labelsize': 20,
          'axes.titlesize': 20,
          'xtick.labelsize':'x-large',
          'ytick.labelsize':'x-large'}
plb.rcParams.update(params)


def dyna_plot(Nz, dz, dt, block_mr, name, thickness='None'):
    """

    """

    images = []

    xZ = np.array([i for i in np.arange(Nz)]) * dz * 1000
    #plt.ion()

    for tem in np.arange(0, len(block_mr), 250):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(xZ, block_mr[tem], 'r-', linewidth=1.2)
        ax.set_title("Time = {:.3f}".format(tem*dt), loc='left')
        ax.set_xlabel("Length along the Z axis (mm)")
        ax.set_ylabel("Temperature (K)")
        ax.set_ylim(273.15, 600)
        fig.savefig('Present.png')
        plt.close(fig)
        gc.collect()
        images.append(imageio.imread('Present.png'))
    imageio.mimsave('T_zX_{}_{}.gif'.format(thickness, name), images, duration=0.2)

    return None