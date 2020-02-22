import multiprocessing as mp
from HeatFlow import body
import matplotlib.pyplot as plt
import matplotlib.pylab as plb
import gc

paras = {
    'figure.figsize': (10, 8),
    'axes.labelsize': 'x-large',
    'axes.titlesize': 'x-large',
    'xtick.labelsize': 'x-large',
    'ytick.labelsize': 'x-large',
}
plb.rcParams.update(paras)


def main():

    pool = mp.Pool(mp.cpu_count())
    time, temp = pool.apply(body)
    pool.close()

    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111)
    ax.plot(time, temp,'r-')
    fig.savefig('image.png')
    plt.close(fig)
    gc.collect()


if __name__ == '__main__':
    main()