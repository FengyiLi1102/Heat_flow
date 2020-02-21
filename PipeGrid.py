import numpy as np

# Global parameters
T_a = 298.15
T_g = 573.15
thickness = 0.001
L = 0.1


def PipeGrid(dh):

    grid = np.full((int(L/dh)+2, int(thickness/dh)+2), T_a)
    grid[0, :], grid[-1, :], grid[:, -1] = T_a, T_a, T_a
    grid[:, 0] = T_g

    return grid