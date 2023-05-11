import numpy as np
from gen import *


def initParameters():

    inputs = np.loadtxt('inp')

    if inputs.shape[0] == 7:
        Lx, num_e, maxcase, lo, hi, mode, sparse = inputs
        Ly = 1

    elif inputs.shape[0] == 8:
        Lx, Ly, num_e, maxcase, lo, hi, mode, sparse = inputs 
        
    else:
        print(' Wrong number of overall inputs')
        exit()

    Lx = int(Lx)
    Ly = int(Ly)
    L = Lx * Ly

    sparse = int(sparse)
    mode = int(mode)
    num_site = 0
    maxlen = 0

    # mode 0 is the universal x, y disorder generation, used in testing a new t, lambda combination
    # mode 1 is select site generation, only on x direction, and the rest of the site have minimal disorder. 
    if mode == 0:

        para_dis = np.loadtxt('para_dis', dtype='str')

        if para_dis.shape[0] == 8:
            tun, cou, a, b,  readdisorder, seed, decay, distype = para_dis

        else:
            print(' Wrong number of inputs. Mode 0. check: tun, cou, a, b,  readdisorder, seed, decay, distype')
            exit()

    elif mode == 1:
        # the a, b now refers to the lower and upper limit of the site disorder, a sign is assigned randomly.

        para_dis = np.loadtxt('para_dis', dtype='str')

        if para_dis.shape[0] == 10:
            tun, cou, a, b,  readdisorder, seed, decay, distype, num_site, maxlen = para_dis

        else:
            print(' Wrong number of inputs. Mode 1. check: tun, cou, a, b,  readdisorder, seed, decay, distype, num_site, maxlen')
            exit()

    # mode 2 generates all maxlen cases
    # mode 3 is the interactive mode
    else:

        para_dis = np.loadtxt('para_dis', dtype='str')

        if para_dis.shape[0] == 9:
            tun, cou, a, b,  readdisorder, seed, decay, distype, num_site  = para_dis

        else:
            print(' Wrong number of inputs. Mode 1+. check: tun, cou, a, b,  readdisorder, seed, decay, distype, num_site')
            exit()

    ham = np.loadtxt('hamiltonian')

    if ham.shape[0] == 10:
        t, int_ee, int_ne, z, zeta, ex, selfnuc, hopmode, int_range, alltoall = ham

    else:
        print(' Wrong number of inputs to ham: t, int_ee, int_ne, z, zeta, ex, selfnuc, hopmode, int_range, alltoall')
        exit()


    para = {
    'L' : L,
    'Lx': Lx,
    'Ly': Ly,
    'num_e' : int(num_e),
    't': t,
    'int_ee': int_ee,
    'int_ne': int_ne,
    'int_range': int_range,
    'z': z,
    'zeta':zeta,
    'ex': ex,
    # if-include-nuc-self-int switch, 1 means include
    'selfnuc': int(selfnuc),
    'alltoall':int(alltoall),
    'tun': int(tun),
    'cou': int(cou),
    'a': float(a),
    'b': float(b),
    'seed': int(seed),
    'readdisorder': int(readdisorder),
    'decay': float(decay),
    'maxcase': int(maxcase),
    'distype': distype,
    'Nth eig': [int(lo), int(hi)],
    # mode controls the generation type. 0 is generation on all sites, 1 is controlled generation on fixed number of sites on singular maxlen, 2 is 1 but on all possible maxlens
    'mode': mode,
    'num_site': int(num_site),
    'maxlen':int(maxlen),
    'hopmode': int(hopmode),
    'sparse':sparse}
    
    if mode == 0:
        para['batch'] = para['maxcase']

    elif mode == 1:
        para['batch'] = para['maxcase'] * (para['L'] - para['maxlen'] + 1)

    else:
        para['batch'] = para['maxcase'] * (para['L'] - para['num_site'] + 1)

    print('Simulation parameters: {}'.format(para))
    return para

def init(para):
    L = para['L']
    num_e = para['num_e']
    states = generateState(L, num_e)
    return states

def initdict(S):
    num_state = len(S)
    L = len(S[0])

    occdict = np.zeros( (num_state, L) )
    balancestate = np.zeros( num_state)
    sdict = {}

    for i, state in enumerate(S):
        sdict[str(state)] = i

        for j, occ in enumerate(state):
            #print(i, j)
            occdict[ i, j ] = occ

        if sum( state[:L//2]) == 3:
            balancestate[i] = 1

    #print(balancestate)
    return sdict, occdict, balancestate
