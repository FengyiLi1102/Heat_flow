import matplotlib.pyplot as plt
import matplotlib.pylab as plb
import pandas as pd
import scipy.constants as constant
import pandas as pd
import numpy as np
import gc
import math as ma


# Global variables
C = 4 * 10**27
E = 3 * 10**4
R = constant.R
T_eff = 273.15 +300
N = 1.173 * 10**24 * 0.1
f = 0.01 * 10**-2
A = 0.01351

# Globle setting for the figure drawing
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (10, 8),
          'axes.labelsize': 20,
          'axes.titlesize': 20,
          'xtick.labelsize':'x-large',
          'ytick.labelsize':'x-large'}
plb.rcParams.update(params)


def main():

    data = pd.read_csv('data.csv')
    time = data.iloc[:, 0].to_numpy()
    temp = data.iloc[:, 1].to_numpy()

    dt = (np.roll(time, -1, axis=0) - time)[: -1]
    Tbar = ((np.roll(temp, -1, axis=0) + temp) / 2)[: -1]
    Ti, = np.where(Tbar >= T_eff)

    N_target = N * (1 - f)
    N_transmitted = 0
    N_left = []

    for i in Ti:
        N_transmitted = C * ma.exp(-E / (R*i)) * A * dt[i]
        N_left.append(N - N_transmitted)

        if N_transmitted >= N_target:
            print('Time required is: {:.2f}'.format(dt[i]))
    
    N_left = np.concatenate((np.full((1, Ti[0]+1), N), N_transmitted), axis=0)
    fig = plt.figure(dpi=300)
    ax = fig.add_subplots(111)
    ax.plot(time, N_left, 'r-', linewidth=1.2)
    ax.set_xlabel('Number of unprocessed CO molecules (unit)')
    ax.set_ylabel('Temperature of the catalyst (K)')
    ax.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
    ax.xaxis.get_offset_text().set_fontsize(15)
    ax.xaxis.major.formatter._useMathText = True
    fig.savefig('N_T_plot.png')
    plt.close(fig)
    gc.collect()


if __name__ == '__main__':
    main()




