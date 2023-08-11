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
from time_evolve import *


if __name__ == '__main__':

    para = initParameters()
    num_e = para['num_e']
    readdisorder = para['readdisorder']
    mode = para['mode']
    # Start the iterative Monte Carlo updates
    energy = []

    # Here we try using a randomly generated set of occupation configuration
    S = init(para)
    # generate dictionary for book keeping
    sdict, occdict, balancestate = initdict(S, para)

    #print(occdict.shape)

    if mode == -1:
        time_evolve(para)

    elif mode < 3:
        disx, disy, sites = generateDisorder(para) 
        
        #cases = 1
        cases = len(disx)

        para["readM"] = 0 if cases > 1 else 1

        allres = []
        eigvs = []
        allmany_res = []
        site_info = []

        start_time = time.time()
        pool = mp.Pool(mp.cpu_count())

        def collect_result( result):

            eigv, result, many_result, site_res = result

            if num_e > 1:
                allmany_res.append(many_result)

            eigvs.append( eigv.flatten())
            allres.append(result)
            site_info.append(site_res)


        for case in range(cases):
            
            x = pool.apply_async(single_iteration, args=(case, disx, disy, sites, S, sdict, occdict, balancestate, para), callback=collect_result)
        
        x.get()
        pool.close()
        pool.join()
            
        # the order is disx, disy, energy, ipr
        saveresults(eigvs, allres, allmany_res, site_info, para)
        print('finish time: {}'.format(time.time() - start_time))
            #calEnergy(S, A, para)
            #S = initSpin(rdim, cdim)
            #print(hamiltonian(S, rdim, cdim, t, int_ee, int_ne, Z, zeta, ex))

    # interactive mode
    else:
        interactive(S, sdict, occdict, balancestate, para)
        
