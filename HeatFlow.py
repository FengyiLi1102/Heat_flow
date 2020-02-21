import numpy as np
from PipeGrid import PipeGrid
from getNeighbours import getNeighbours
import matplotlib.pyplot as plt

# Global parameters
v = 21.8
L = 0.1
t_Tot = L / v
T_g = 573.15
k_s = 16.3
rho_s = 8030
reco_step = 100


def main():
    dh = 0.0001
    dt = 0.00001
    time = 0.1
    t_finish = int(time/dt)
    grid = PipeGrid(dh)
    T_Tot = []

    T_Tot.append(grid)
    for t in np.arange(1, t_finish):
        z = v * dt * t

        if z < L:
            if t % reco_step == 0:
                T_Tot.append(grid)

            heated = int(z/dh)
            grid[:heated, 1] = T_g
        else:
            grid[:, 1] = T_g
            t_start = t
            break
        
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
    
    for t in np.arange(t_start, t_finish):
        if t % reco_step == 0:
            T_Tot.append(grid)

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
    
    fig, ax = plt.subplots(111, dpi=300)
    t_finish = np.linspace(0, t_finish, t_finish+1) * dt
    T_plot = []
    for T in T_Tot:
        T_plot.append(T[int(grid.shape[0]/2), grid.shape[1]-2])
    ax.plot(t_finish, T_plot, 'r-')
    fig.savefig('image.png')
    plt.close(fig)


if __name__ == '__main__':
    main()