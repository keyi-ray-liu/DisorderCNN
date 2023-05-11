import numpy as np
from utils import *
from solve import *
import matplotlib.pyplot as plt
from set import *

def single_iteration(case, disx, disy, sites, S,  sdict, occdict, balancestate, para):


    def cal_TCD():

        # assume full spectrum
        

        tcd = np.zeros(( k,  L))

        for n in range(k):
            for i in range(L):
                tcd[n, i] = sum(eigv[:, 0] * eigv[:, n] * occdict[:, i])

        return tcd

    def distance(i, j):

        return np.sqrt((j - i + dis[0][j] - dis[0][i]) ** 2 + ( dis[1][j] - dis[1][i]) ** 2) 

    def ee(i, j):
        
        #ex = 0
        #zeta = 0
        
        if abs(j - i) == 1:
            factor = 1 - ex

        else:
            factor = 1

        return z * int_ee * factor / ( distance(i, j) + zeta)
        

    def cal_GPI():
        gpi = np.zeros(k)

        for n in range(k):
            for i in range(L):
                for j in range(L):
                    
                    
                    gpi[n] += tcd[ n, i] * tcd[n, j] * ee(i, j) 

        #print(gpi)
        return gpi

    def cal_balance():
        return np.array([sum( (balancestate * eigv[:, n]) ** 2) for n in range(k) ])


    
    def allplot():

        fig, ax = plt.subplots(1,4, figsize = (20, 5))
        title = 'Plots for $\lambda$ = {}t, L = {}, N = {}'.format(int_ee, L, num_e)
        #internal_plot(fig, ax, tcd, gpi, balance, energy, disx[case], title)

    num_e = para['num_e']
    L = para['L']
    mode = para['mode']

    z, int_ee, zeta, ex = para['z'], para['int_ee'], para['zeta'], para['ex']

    print('case: {}'.format(case))

    dis = [disx[case], disy[case]]
    #print('x Disorder: {}\n y Disorder: {}'.format(dis[0], dis[1]))
    # total number of states
    N = len(S)

    # in full spectrum k == N
    k = para['Nth eig'][-1] - para['Nth eig'][0] + 1

    M = setMatrix(S, N, dis, sdict,  para)

    #print(M)
    energy, eigv = solve(M, para)

    #plotprob(eigv, para)
    #print('Eigenvectors (by column): \n {}'.format(eigv))
    ipr = calIPR(eigv)

    # June 13 update: remove explicit TCD, GPI calculate, leave everything to eigv
    if num_e > 1:

        if mode >= 3:
            tcd = cal_TCD()

            gpi = cal_GPI()

        balance = cal_balance()

        
    #print(energy)
    #print(ipr)

    if mode < 3 :
        #allplot()

        res = np.concatenate( (dis[0], dis[1], energy, ipr))
        site_res = sites[case]

        if num_e > 1:
            #many_res = np.concatenate( (tcd.reshape( k * L), gpi, balance))
            many_res = balance
            #print(len(many_res))

        else:
            many_res = []

        #print(len(res))
        return [eigv, res, many_res, site_res]
        #collect_result(energy, ipr)

        #print('Energy is {}'.format(energy))
        #print('Inverse participation ratio: {}'.format(ipr))

    else:
        
        return eigv, tcd, gpi, balance, energy

