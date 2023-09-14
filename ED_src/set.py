from collections import defaultdict
from hamiltonian import *
from scipy.sparse import csr_matrix, load_npz, save_npz
import os
from utils import *


def setMatrix(S, N, dis, sdict, para, tag=''):

    sparse = para['sparse'] 
    readM = para["readM"]
    def checksparce():
        
        cnt = defaultdict(int)
        for row in M:
            cnt[len(np.argwhere(row))] += 1

        #print(cnt)

    frac = N // 100 if N >= 100 else 1
    cnt = 0

    sparse_M = 'M{}.npz'.format(tag)
    dense_M = 'M{}'.format(tag)

    if sparse:

        if os.path.exists(sparse_M) and readM:
            M = load_npz(sparse_M)

        else:
            row = []
            col = []
            val = []

            for i, state in enumerate(S):

                if i % frac == 0:
                    print('setup complete {}%'.format(cnt))
                    cnt += 1

                newstates = hamiltonian(state, dis, para)
                for j, newstate  in enumerate(newstates[1]):
                    row += [i]
                    col += [sdict[str(newstate)]]
                    val += [newstates[0][j]]

            M = csr_matrix((val, (row, col)), shape=(N, N))
            save_npz(sparse_M, M)

    else:

        if os.path.exists(dense_M) and readM:
            M = np.loadtxt(dense_M)

        else:
            M = np.zeros((N, N))

            for i, state in enumerate(S):

                if i % frac == 0:
                    print('setup complete {}%'.format(cnt))
                    cnt += 1
            
                newstates = hamiltonian(state, dis, para)
                for j, newstate  in enumerate(newstates[1]):
                    M[i, sdict[str(newstate)]] += newstates[0][j]

            np.savetxt(dense_M + 'readable', M, fmt='%.3f')
            np.savetxt(dense_M, M)
            
            #\checkdiag(M, para)
            checksparce()

    #if os.path.exists(sparse_M) and os.path.exists(dense_M):
    #    check_same(sparse_M, dense_M)
    
    return M
   



