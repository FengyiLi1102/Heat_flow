import numpy as np
from PipeGrid import PipeGrid
from getNeighbours import getNeighbours
import matplotlib.pyplot as plt
from simulate import run


# Global parameters
v = 21.8
L = 0.1
t_Tot = L / v
T_g = 573.15
k_s = 16.3
rho_s = 8030
reco_step = 1


def body():
    dh = 0.0001
    dt = 0.0001
    time = 100
    t_finish = int(time/dt)
    grid = PipeGrid(dh)
    X = int(grid.shape[0]/2)
    Y = grid.shape[1] - 2
    T_Tot = []
    file = open('stats.csv', 'w')
    file.write('Time (s), Temperature (K)\n')

    T_Tot.append(grid)
    file.write('{0}, {1:.2f}\n'.format(0, grid[X, Y]))
    for t in np.arange(1, t_finish):
        print("----{}----".format(t*dt))
        z = v * dt * t

        if z < L:
            heated = int(z/dh)
            grid[:heated, 1] = T_g

            if t % reco_step == 0:
                T_Tot.append(grid)
                file.write('{0:.2f}, {1:.2f}\n'.format(t*dt, grid[X, Y]))
        else:
            grid[:, 1] = T_g
            break
        
        run(grid, T_g, k_s, rho_s, T_Tot, dh, dt)
    
    t_start = t

    for t in np.arange(t_start, t_finish):
        print("----{}----".format(t*dt))
        if t % reco_step == 0:
            T_Tot.append(grid)
            file.write('{0:.2f}, {1:.2f}\n'.format(t*dt, grid[X, Y]))

        run(grid, T_g, k_s, rho_s, T_Tot, dh, dt)
    
    t_list = np.arange(len(T_Tot)) * dt
    T_plot = []
    for T in T_Tot:
        T_plot.append(T[X, Y])

    file.close()
    return t_list, T_plot


if __name__ == '__main__':
    body()