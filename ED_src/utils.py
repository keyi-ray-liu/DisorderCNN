import numpy as np
from scipy.sparse import csr_matrix, load_npz
from scipy.special import comb


def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

def calIPR(eigv):
    return np.sum(eigv**4, axis=0)

def saveresults(eigvs, allres, allmany_res, site_info, para):

    
    L = para['L']
    lo, hi = para['Nth eig']
    fullrange = hi - lo + 1
    strs = ['energy', 'ipr', 'disx', 'disy']
    num_e = para['num_e']


    allres = np.array(allres)

    disx = allres[ :, : L]
    disy = allres[ :, L : L * 2]
    energy = allres[ :, 2 * L : 2 * L + fullrange]
    ipr = allres[:, 2 * L + fullrange : 2 * L + fullrange * 2]

    data = [energy, ipr, disx, disy]
    
    np.savetxt('eigvs', eigvs, fmt='%.8e')

    for i in range(len(strs)):
        np.savetxt(strs[i], data[i], fmt='%.8f')

    if sum(num_e) > 1:

        #June 13 update, eigv only 

        # allmany_res = np.array(allmany_res)
        # tcd = allmany_res[:, : L * fullrange]
        # gpi = allmany_res[ :, L * fullrange : L * fullrange + fullrange]
        # balance = allmany_res [ :, L * fullrange + fullrange : L * fullrange + fullrange * 2]

        # manydata = [ tcd, gpi, balance]
        # tag = ['tcd', 'gpi', 'balance']

        # for i in range(len(tag)):
        #     np.savetxt( tag[i], manydata[i])
        np.savetxt( 'balance', allmany_res)


    np.savetxt('sites', site_info, fmt='%i')


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

def check_same(sparse, dense):

    sparse :csr_matrix= load_npz(sparse)
    dense = np.loadtxt(dense)

    print(np.allclose(sparse.toarray(), dense))

