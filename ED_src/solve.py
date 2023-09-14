from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
import time

def solve(M, para):
    ranges = para['Nth eig']
    sparse = para['sparse']
    # for use in sparse solver
    

    #print( check_symmetric(M))
    #print( M )

    start = time.time()

    if sparse:
        k = ranges[-1] - ranges[0] + 1

        #print('num of eigenvalues: {}'.format(k))
        w, v = eigsh(M, k, which='SA')

    else:

        #print('range of eigenvalues', ranges)
        w, v = eigh(M, subset_by_index=ranges)
    
    print(w)
    print('diag time {}'.format(time.time()- start))

    return w, v
    #return eigh(M)


