import numpy as np
import matplotlib.pyplot as plt


def plotprob(eigv, para):

    N = len(eigv[0])
    
    for eig in range(N):
        if eig == 1:
            plt.title('Wavefunction Plot')
        plt.subplot(N, 1, eig + 1)
        plt.plot(list(range(len(eigv))), eigv[:,eig] )
    
    plt.show()

def internal_plot(fig, ax, tcd, gpi, balance, energy, dis, title):
    
    L = len(tcd[0])
    N = len(balance)

    for i in range(4):
        ax[i].clear()

    gpi = np.abs(gpi[1:])
    gs = energy[0]
    plasmon = energy[1] - gs

    energy = energy[1:]

    idx = np.argsort( gpi)[::-1][:6]


    ax[0].plot( np.arange(1, L + 1), tcd[1, :])
    ax[0].plot( np.arange(0.5, L + 1.5), [tcd[1, 0] / 2] + [ (tcd[1, j] + tcd[1, j - 1])/2 for j in range(1, L)] + [tcd[1, -1]/2])
    ax[1].scatter ( [ ( ene - gs) / (plasmon) for ene in np.delete(energy, idx)], np.delete(gpi, idx))
    ax[1].scatter ( [ ( ene - gs) / (plasmon) for ene in energy[idx]], gpi[idx], c = 'red')
    ax[1].set_xlim(0, 20)
    ax[2].scatter( np.arange(N), balance, s = 3)
    ax[3].scatter( np.arange(1, L + 1) + dis, np.ones(L))
    ax[3].set_ylim(0, 2)
    ax[3].set_xlim(0, L + 1)
    ax[3].set_axis_off()

    ax[0].set_title('TCD vs site')
    ax[1].set_title('GPI vs excitation energy/ plasmon energy')
    ax[2].set_title('Balance vs eigenstate')
    ax[3].set_title('Visualization of disorder')

    ax[0].set_xlabel('site number')
    ax[1].set_xlabel('Excitation energy/ plasmon energy')
    ax[2].set_xlabel('eigenstate number')

    fig.suptitle(title)
    #plt.show()

