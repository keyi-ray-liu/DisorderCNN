import numpy as np
import copy
from itertools import combinations


def hamiltonian(s, dis, para):
    L, t, int_ee, int_ne, z, zeta, ex, selfnuc = para['L'],  para['t'], para['int_ee'],para['int_ne'], para['z'], para['zeta'], para['ex'],  para['selfnuc']
    Lx, Ly = para['Lx'], para['Ly']

    num_e = para['num_e']
    # int_range only applies to ee and ne int
    int_range = para['int_range']
    tun, cou, decay = para['tun'], para['cou'], para['decay']
    allnewstates = [[], []]
    allee, allne = 0, 0
    hopmode = para['hopmode']
    alltoall = para['alltoall']
    qe_energy = para['qe_energy']
    qe = para['qe']
    qe_dis = para['qe_dis']
    dp = para['dp']

    def checkHopping(loc):
        # set up the NN matrix
        ts = []
        res = []
    

        #only hop to NN, candidates include right and up
        if not alltoall:
            cand = set()

            # if site not on the edge
            if loc < L - 1 and loc % Lx != Lx - 1:
                cand.add( loc + 1)

            if loc < L - Lx:
                cand.add( loc + Lx)

        #hop to all sites after
        else:
            cand = range(loc + 1, L)

        for site in cand:

            if s[loc] != s[site]:

                xloc = loc % Lx
                yloc = loc // Lx

                xs = site % Lx
                ys = site // Lx

                snew = copy.copy(s)
                snew[loc], snew[ site ] = snew[ site], snew[loc]
                res.append(snew)

                # jordan wigner string for cc operator
                jw = (-1) ** s[ loc + 1: site].count(1)
                if tun:
                    dx = - dis[0][loc]  -xloc + dis[0][site] + xs
                    dy = - dis[1][loc] -yloc + dis[1][site] + ys
                    dr = np.sqrt(dx ** 2 + dy ** 2)

                    # exponential decay 
                    if hopmode == 0:
                        factor = np.exp( - ( dr - 1) /decay )

                    else:
                        factor = np.sin( dr )
                    #print(factor)
                    ts.append( -t * factor * jw)
                else:
                    ts.append(- t * jw)
                

        # 2D lattice, hop down
            
            # #row number, determines which 'direction' the chain wraps around. If even, to the right, otherwise to the left
            # row = loc // Lx

            # #this section details the 'snake' geometry, removed
            # #odd
            # if row % 2:
            #     x = loc % Lx
            #     skip = 2 * x + 1

            # else:
            #     x = Lx - loc % Lx
            #     skip = 2 * x - 1

            # #if skip == 1, then it's only hop to right
            # if loc < L - skip and s[loc] != s[loc + skip] and skip != 1:



            # sum the hopping terms
            #print(ts)


        return ts, res

    def ee(loc):  
        total_ee = 0
        
        if Ly > 1:
            xloc = loc % Lx
            yloc = loc // Lx

        for site in range(loc + 1, L):
            # no same-site interaction bc of Pauli exclusion

            contn = 0

            # check distance and int_range
            if Ly == 1:
                r = site - loc

                if r <= int_range:
                    contn = 1

            else:
                xs = site % Lx
                ys = site // Lx

                xd = xs - xloc
                yd = ys - yloc

                r = np.sqrt(xd **2 + yd **2)
                if r <= int_range:
                    contn = 1
            
        
            # check if < int_range
            if contn:

                # exchange interaction
                if Ly == 1:
                    if abs(r) == 1:
                        factor = 1 - ex
                    
                    else:
                        factor = 1

                else:
                    if abs(xd) + abs(yd) == 1:
                        factor = 1 - ex
                    
                    else:
                        factor = 1


                # Disorder
                if cou:

                    if Ly == 1:
                        r = np.sqrt( ( - dis[0][loc] + r + dis[0][site]) ** 2 + ( - dis[1][loc] + dis[1][site] ) ** 2)

                    else:

                        r = np.sqrt( ( - dis[0][loc] + xd + dis[0][site]) ** 2 + ( - dis[1][loc] + yd + dis[1][site] ) ** 2)
                
                # adding contribution to the total contribution
                total_ee +=  int_ee * z * factor / ( abs(r) + zeta ) * s[loc] * s[site]

        return total_ee 


    def ne(loc):
        total_ne = 0
        # sum the contribution from all sites

        if Ly > 1:
            xloc = loc % Lx
            yloc = loc // Lx

        for site in range(L):
            
            contn = 0
            # distance between sites
            # check if int_range
            if Ly == 1:
                r = site - loc

                if np.abs(r) <= int_range:
                    contn = 1

            else:
                xs = site % Lx
                ys = site // Lx

                xd = xs - xloc
                yd = ys - yloc

                r = np.sqrt(xd **2 + yd **2)
                if r <= int_range:
                    contn = 1
            
            # if within range
            if contn:
                
                # if disorder flag
                if cou:
                    if Ly == 1:
                        r = np.sqrt( ( - dis[0][loc] + r + dis[0][site]) ** 2 + ( - dis[1][loc] + dis[1][site] ) ** 2)

                    else:
                        r = np.sqrt( ( - dis[0][loc] + xd + dis[0][site]) ** 2 + ( - dis[1][loc] + yd + dis[1][site] ) ** 2)

                total_ne +=  int_ne * z / ( abs(r) + zeta ) * s[site]

        # self nuclear interaction condition
        return total_ne if selfnuc else total_ne - int_ne * z / zeta * s[loc]


    def add_qe():
        
        qe_pot = []
        states = []

        qe_state = s[L:]
        chain_state = s[:L]
        # add diagonal energies

        qe_pot.append( qe_state.count(1) * qe_energy)
        states.append( s)

        # add off-diagonal energies
        
        # flip 1, 2, ..., N qe's 
        for activate_qe in range(1, qe ):
            
            # flip all possile x num
            for comb in combinations( range(qe), activate_qe):

                new_qe_state = copy.copy(qe_state)
                new_qe_pot = 0

                #res = 0
                for ind in comb:
                    new_qe_state[ind] = int(not qe_state[ind])

                    #calculate energy
                    np_s = np.array(chain_state)
                    r_vec = np.abs( np.arange(L) - qe_dis[ind])
                    new_qe_pot += np.sum(dp[ind] * (np_s - num_e/L) * ( r_vec / ( r_vec **3 + zeta)))


                qe_pot.append( new_qe_pot)
                states.append( chain_state + new_qe_state)
        
        return qe_pot, states



    for loc in range(L):

        # the hopping part. Set up the changed basis states
        ts, newstate =  checkHopping(loc)
        for i in range(len(ts)):
            allnewstates[0].append(ts[i])
            allnewstates[1].append(newstate[i])


        # the ee interaction part, the 0.5 is for the double counting of sites. 
        allee += ee(loc) 
        # the ne interaction part
        allne += ne(loc)
        #print(ne(row, col))
        

    #print(allee, allne)

    # onsite interactions
    allnewstates[0].append(allee + allne)
    allnewstates[1].append(s)

    if qe:

        qe_pot, qe_state = add_qe()
        for j in range(len(qe_pot)):
            allnewstates[0].append(qe_pot[j])
            allnewstates[1].append(qe_state[j])

    return allnewstates


