import numpy as np
from getNeighbours import getNeighbours


def run(grid, T_g, k_s, rho_s, T_Tot, dh, dt):

    for x in np.arange(1, grid.shape[0]-1):
            for y in np.arange(1, grid.shape[1]-1):
                if grid[x, y] == T_g:
                    pass
                else:
                    grid_last = T_Tot[-1]
                    c_s = 450 + 0.28 * (grid[x, y]-273.15)
                    alpha = k_s / (rho_s*c_s)
                    neighbours = getNeighbours((x, y))
                    coeff = alpha / (dh**2)
                    sum_T = 0
                    for pair in neighbours:
                        sum_T += grid_last[pair[0], pair[1]]

                    grid[x, y] = dt * (coeff * (sum_T - 6*grid_last[x, y])) + grid_last[x, y]
    
    return