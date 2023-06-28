import time
import multiprocessing as mp

#from numpy.linalg import eigh

from init import *
from gen import *
from set import *
from single_run import *
from hamiltonian import *
from solve import *
from utils import *
from interactive import *
from plots import *


if __name__ == '__main__':

    para = initParameters()
    num_e = para['num_e']
    # Start the iterative Monte Carlo updates
    energy = []

    # Here we try using a randomly generated set of occupation configuration
    S = init(para)

    # generate dictionary for book keeping
    sdict, occdict, balancestate = initdict(S, para)

    disx, disy, sites = generateDisorder(para) 
    energy = np.loadtxt('energy')

    if len(disx) != len(energy):
        raise ValueError("test files has wrong dimensions")
    
    inds = np.random.choice( energy.shape[0], 20, replace=False)
    L = para['L']
    lo, hi = para['Nth eig']
    fullrange = hi - lo + 1
    
    for ind in inds:
     
        _, result, _, _ = single_iteration(ind, disx, disy, sites, S,  sdict, occdict, balancestate, para )
        newenergy = result[2 * L : 2 * L + fullrange]
        print(np.allclose(newenergy, energy[ind]))