import numpy as np
from utils import *
from solve import *
from set import *
from init import *
from copy import deepcopy
import scipy.linalg as dense_linalg
import scipy.sparse.linalg as sparse_linalg
from scipy.linalg import eigh
import os
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
import matplotlib.pyplot as plt


def TE_GS(para, ic):

    # force set QE=0 to get chain GS
    # we only need GS
    
    new = deepcopy(para)

    qe = para['qe']
    if qe != len(ic):
        raise ValueError('wrong initial condition ')

    new['qe'] = 0
    new['Nth eig'] = [0, 0]
    S = init(new)
    # generate dictionary for book keeping
    sdict, _, _ = initdict(S, new)
    L = new['L']
    # for the moment, disorder is set to 0
    dis = np.zeros((2, L))
    #print('x Disorder: {}\n y Disorder: {}'.format(dis[0], dis[1]))
    # total number of states
    N = len(S)
    M = setMatrix(S, N, dis, sdict,  new)
    _, eigv = solve(M, new)

    # set the correct index for outer product
    ind = np.sum( np.power(2, np.arange( qe) ) * ic)

    pos = np.zeros( 2 ** qe)
    pos[ind] = 1
    eigv = np.outer(eigv, pos).flatten('F')

    return eigv

def time_evolve(para):

    method = para['method']
    start = para['start']
    end = para['end']
    timestep = para['timestep']
    num_e = para['num_e']
    ic = [1, 0]
    gs = para['gs']

    L = para['L']
    qe = para['qe']
    S = init(para)
    # generate dictionary for book keeping
    sdict, occdict, _ = initdict(S, para)
    psi = TE_GS(para, ic)
    np.savetxt('te_gs', psi) 

    if gs:

        gs_cd = charge_density(psi, occdict, para)
        gs_cd[L:] = np.zeros( gs_cd.shape[0] - L)

    else:
        gs_cd = np.zeros( L + qe)
    
    # for the moment, disorder is set to 0
    dis = np.zeros((2, L))

    # get the full hamiltonian M
    N = len(S)
    
    # if sparse:
    #     expM = sparse_linalg.expm(- 1j * timestep * M )

    # else:
    #     expM = dense_linalg.expm(- 1j * timestep * M )
    # psi2 = psi
    
    res = []
    raw_comp = []
    if method == "direct":

        M = setMatrix(S, N, dis, sdict,  para)

        for t in np.arange(start, end, timestep):
            print(t)

            print( 'direct sum:', np.sum( psi.conj().dot(psi)))
            cur_cd = charge_density(psi, occdict, para)
        #     #psi2 = expM * psi2
        #     #print(  psi2.conj().transpose().dot(psi2))
        #     #charge_density(psi2, occdict)

            raw_comp.append(cur_cd)
            res.append(cur_cd - gs_cd)
            psi = sparse_linalg.expm_multiply( -1j * timestep * M, psi)

            

    # in this case we want the full spectrum
    elif method == "eigen":
        
        # override sparse to false

        if os.path.exists('full_eigen'):

            energies = np.loadtxt('full_energy')
            v = np.loadtxt('full_eigen')

        else:
            para["sparse"] = 0
            M = setMatrix(S, N, dis, sdict,  para)
            # rememer the structure of the output eigv is (M x N), where N <= M
            energies, v = eigh(M)

            np.savetxt('full_eigen', v)
            np.savetxt('full_energy', energies)

        # calculate the overlap vector between all eigenv and initial state
        overlap = psi.dot( v)

        # calculate the cd for each energy eigenstate
        cd = v.transpose().dot(occdict)

        #normalization = np.repeat( np.sqrt(  np.sum( np.abs(cd) **2 , axis=1) / num_e ), cd.shape[-1]).reshape( cd.shape)
        #cd = cd / normalization

        #print( np.sum( np.abs(cd) **2 , axis=1))

        # calculate the phased overlap
        for t in np.arange(start, end, timestep):

            #print(t)
            ph_overlap = np.exp(-1j * energies * t) * overlap
            cur_cd = np.absolute( ph_overlap.dot(cd) ) ** 2 
            #print( cur_cd)

            raw_comp.append(cur_cd)
            res.append( cur_cd - gs_cd)
            #print( 'phased overlap inner:', np.sum( np.absolute( ph_overlap) ** 2))
            #charge_density(ph_overlap, occdict)

    res= np.array(res)
    raw_comp = np.array(raw_comp)

    mps_consist(raw_comp)
    np.savetxt('cd', res, fmt='%.4f')
    #print(gs_cd)
    timeplot(res, para)

def charge_density(psi, occdict, para):
    
    L = para['L']
    cd = (np.abs(psi) ** 2).dot(occdict)

    #cd[:L] = cd[:L] / np.sum( cd [:L])
    #cd[L:] = cd[L:]/ np.sum( cd[L:])
    #print(np.sum(cd[:-2]))
    # print(np.sum(cd))
    # print(cd)

    return cd

def mps_consist(raw):

    zeros = np.zeros( (raw.shape[0], 1))

    perm = list(range(1, 13)) + [0] + [13]
    idx = np.empty_like(perm)
    idx[perm] = np.arange(len(perm))

    raw[:] = raw[:, idx]
    raw = np.concatenate( (zeros, raw, zeros), axis= 1)

    np.savetxt('expN', raw)

def timeplot(res, para):

    def animate(i):

        print("frame {}".format(i))
        ax[0].clear()
        ax[1].clear()

        ref = np.arange( chain[i].shape[0])
        ax[0].scatter( ref, chain[i])
        ax[0].set_ylim(lo, hi)
        ax[0].plot( ref, chain[i])

        qeref = np.array([0, 1])
        ax[1].scatter( qeref, qe[i])
        ax[1].set_ylim(0, 1)

    L = para['L']
    fig, ax = plt.subplots(2)
    frame = res.shape[0]

    chain = res[:, :L]
    qe = res[:,L:]

    padding = np.zeros( (frame, 1))
    chain = (np.concatenate( (padding, chain), axis=1) + np.concatenate((chain, padding) , axis=1) ) / 2
    hi= np.amax(chain)
    lo = np.amin(chain)


    anim = FuncAnimation(fig, animate, frames= frame, interval=100, repeat=False)
    #plt.show()

    #mpl.rcParams['animation.ffmpeg_path'] = os.getcwd() + '/ffmpeg'

    html = False
    if not html:
        writervideo = animation.FFMpegWriter(fps=15)
        anim.save( 'cd.mp4', writer=writervideo)

    else:
        writervideo = animation.HTMLWriter()
        anim.save( 'cd.html', writer=writervideo)