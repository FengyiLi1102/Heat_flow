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
            heated = int(z/dh)
            grid[:heated, 1] = T_g

            if t % reco_step == 0:
                T_Tot.append(grid)
        else:
            grid[:, 1] = T_g
            t_start = t
            break
        
        run(grid, T_g, k_s, rho_s, T_Tot, dh, dt)
    

    for t in np.arange(t_start, t_finish):
        if t % reco_step == 0:
            T_Tot.append(grid)

        run(grid, T_g, k_s, rho_s, T_Tot, dh, dt)
    
    fig, ax = plt.subplots(111, dpi=300)
    t_list = np.arange(len(T_Tot)) * dt
    T_plot = []
    for T in T_Tot:
        T_plot.append(T[int(grid.shape[0]/2), grid.shape[1]-2])
    ax.plot(t_list, T_plot, 'r-')
    fig.savefig('image.png')
    plt.close(fig)


if __name__ == '__main__':
    main()