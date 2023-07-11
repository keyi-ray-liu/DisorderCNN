from collections import defaultdict
from hamiltonian import *
from scipy.sparse import csr_matrix, load_npz, save_npz
from scipy.special import comb
import os

def setMatrix(S, N, dis, sdict, para, tag=''):

    sparse = para['sparse'] 

    def checksparce():
        
        cnt = defaultdict(int)
        for row in M:
            cnt[len(np.argwhere(row))] += 1

        #print(cnt)

    frac = N // 100
    cnt = 0

    if sparse:

        matrix_file = 'M{}.npz'.format(tag)
        if os.path.exists(matrix_file):
            M = load_npz(matrix_file)

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
            save_npz(matrix_file, M)

    else:

        matrix_file = 'M'
        if os.path.exists(matrix_file):
            M = np.loadtxt(matrix_file)

        else:
            M = np.zeros((N, N))

            for i, state in enumerate(S):

                if i % frac == 0:
                    print('setup complete {}%'.format(cnt))
                    cnt += 1
            
                newstates = hamiltonian(state, dis, para)
                for j, newstate  in enumerate(newstates[1]):
                    M[i, sdict[str(newstate)]] += newstates[0][j]

            np.savetxt('M', M, fmt='%.3f')
            
            #checkdiag(M, para)
            checksparce()
    
    return M
   
def checkdiag(M, para):

    L = para['L']
    num_e = para['num_e']

    num = int(comb(L, num_e))
    rep = int(M.shape[0] // num)
    

    for r in range(rep):
        for c in range(rep):

            block = M[ r * num : (r + 1) * num, c * num: (c + 1)*num]
            zeros = np.count_nonzero( block - np.diag(np.diagonal(block)))

            print(r, c, zeros)


