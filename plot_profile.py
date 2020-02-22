import matplotlib.pyplot as plt
import matplotlib.pylab as plb
import pandas as pd


# Globle setting for the figure drawing
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (10, 8),
          'axes.labelsize': 20,
          'axes.titlesize': 20,
          'xtick.labelsize':'x-large',
          'ytick.labelsize':'x-large'}
plb.rcParams.update(params)


def main():

    data = pd.read_csv('data3.csv')
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111)

    ax.plot(data.iloc[:, 2], data.iloc[:, 3], 'r-', linewidth=1.5)
    ax.set_ylabel('Temperature (K)')
    ax.set_xlabel('Time (s)')
    fig.savefig('T_t_catalyst.png')


if __name__ == "__main__":
    main()