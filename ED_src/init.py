import numpy as np
from gen import *
import json
import collections
from hopping_gen import gen_hopping


def initParameters():

    with open('inp.json', 'r') as f:
        input = json.load(f)


    Lx = int(input['Lx'])
    num_e = input['num_e']
    sparse = int(input['sparse'])
    mode = int(input['mode'])
    maxcase = input['maxcase']
    readM = int(input['readM'])
    hop_default = int(input["hop_default"])
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
        ham = json.load(f)

    U = ham['U']
    int_ee = ham['int_ee']
    int_ne = ham['int_ne']
    int_range = ham['int_range']
    z = ham['z']
    t = ham['t']
    zeta = ham['zeta']
    ex = ham['ex']
    selfnuc = ham['selfnuc']
    hopmode = ham['hopmode']

    if hop_default:
        gen_hopping(Lx, Ly, t)


    try:
        with open('nn.json', 'r') as f:
            nn_raw = json.load(f)

            nn = collections.defaultdict(list)
            for key in nn_raw:
                nn[ int(key) ] = nn_raw[key]
        
    except FileNotFoundError:
        raise(ValueError('NN not found!'))

    para = {
        'L' : L,
        'Lx': Lx,
        'Ly': Ly,
        'num_e' : num_e,
        'int_ee': int_ee,
        'int_ne': int_ne,
        'U': U,
        'int_range': int_range,
        'z': z,
        'zeta':zeta,
        'ex': ex,
        # if-include-nuc-self-int switch, 1 means include
        'selfnuc': int(selfnuc),
        'nn' : nn,
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
        'sparse':sparse,
        'readM' : readM
    }
    

    try:
        with open('dyna.json', 'r') as f:
            dyna = json.load(f)

        para['gs'] = int(dyna['gs'])
        para['timestep'] = dyna['timestep']
        para['start'] = dyna['start']
        para['end'] = dyna['end']
        para['method'] = dyna['method']

    except FileNotFoundError:

        if mode <0:
            raise ValueError('no dyna para found!')
        
        else:
            pass

    try:
        with open('para_qe.json', 'r') as f:
            para_qe = json.load(f)

        para['qe'] = int(para_qe['qe'])
        para['dp'] = para_qe['dp']
        para['qe_energy'] = para_qe['qe_energy']
        para['qe_dis'] = para_qe['qe_dis']

    except FileNotFoundError:
        para['qe'] = 0

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
    qe = para['qe']
    states = generateState(L, num_e, qe)
    return states

def initdict(S, para):
    num_state = len(S)
    L = para['L']
    qe = para['qe']
    num_e = para['num_e']

    occdict = np.zeros( (num_state, L + qe) )
    balancestate = np.zeros( num_state)
    sdict = {}

    for i, state in enumerate(S):
        sdict[str(state)] = i

        for j, occ in enumerate(state):
            #print(i, j)
            occdict[ i, j ] = occ

        if sum( state[:L//2]) == sum(num_e)/2:
            balancestate[i] = 1

    np.savetxt('occdict', occdict, fmt='%i')
    #print(balancestate)
    return sdict, occdict, balancestate
