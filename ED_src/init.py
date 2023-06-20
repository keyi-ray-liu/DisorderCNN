import numpy as np
from gen import *
import json


def initParameters():

    with open('inp.json', 'r') as f:
        input = json.load(f)


    Lx = int(input['Lx'])
    num_e = input['num_e']
    sparse = int(input['sparse'])
    mode = int(input['mode'])
    maxcase = input['maxcase']
    Ly = int(input['Ly']) if 'Ly' in input else 1

    lo = input['lo']
    hi = input['hi']

    L = Lx * Ly


    # mode -1 is time evolution, separate
    # mode 0 is the universal x, y disorder generation, used in testing a new t, lambda combination
    # mode 1 is select site generation, only on x direction, and the rest of the site have minimal disorder. 

    with open('para_dis.json', 'r') as f:
        dis = json.load(f)

    tun = dis['tun']
    cou = dis['cou']
    a = dis['a']
    b = dis['b']
    readdisorder = dis['readdisorder']
    seed = dis['seed']
    decay = dis['decay']
    distype = dis['distype']


    with open('hamiltonian.json', 'r') as f:
        ham = json.load(ham)

    t = ham['t']
    int_ee = ham['int_ee']
    int_ne = ham['int_ne']
    int_range = ham['int_range']
    z = ham['z']
    zeta = ham['zeta']
    ex = ham['ex']
    selfnuc = ham['selfnuc']
    alltoall = ham['alltoall']
    hopmode = ham['hopmode']


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
        'num_site': int(dis['num_site']) if 'num_site' in dis else 0,
        'maxlen':int(dis['maxlen']) if 'maxlen' in dis else 0,
        'hopmode': int(hopmode),
        'sparse':sparse
    }
    

    try:
        with open('dyna.json', 'r') as f:
            dyna = json.load(ham)

        para['timestep'] = dyna['timestep']
        para['start'] = dyna['start']
        para['end'] = dyna['end']


    except FileNotFoundError:
        pass


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
