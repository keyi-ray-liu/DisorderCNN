import numpy as np
from collections import defaultdict

def generateState(L, num_e, qe):

    def recur(L, num):

        if L == num:
            return [[1] * num]
        if L == 0:
            return [[]]
        if num == 0:
            return [[0] * L]
        
        cur =  [ [0]  + state for state in recur(L - 1, num)] + [ [ 1] + state for state in recur(L - 1, num -1)]
        return cur

    if len(num_e) == 1:

        num_e = num_e[0]
        full = recur(L, num_e)

    elif len(num_e) == 2:

        up, dn = num_e
        full = [ ups + 2 * dns for ups in np.array(recur(L, up)) for dns in np.array(recur(L, dn))]

    else:
        raise(ValueError("invalid num_e"))

    for _ in range(qe):
        full = [state + [0] for state in full] + [state + [1] for state in full]
    return full
        

    

def generateDisorder(para):

    #L, batch, readdisorder, distype, seed , a, b = para['L'],   para['batch'], para['readdisorder'], para['distype'], para['seed'], para['a'], para['b']
    L, maxcase, readdisorder, distype, a, b = para['L'],   para['maxcase'], para['readdisorder'], para['distype'], para['a'], para['b']
    mode = para['mode']
    num_site = para['num_site']
    maxlen = para['maxlen']
    batch = para['batch']

    if readdisorder:
        disx = np.loadtxt('disx')
        disy = np.loadtxt('disy')
        sites = np.loadtxt('sites', dtype=int)

    else:
        rng = np.random.default_rng()
        
        if mode == 0:
            if distype == 'uniform':
                disx = a * rng.uniform(-1, 1, (batch, L))
                disy = b * rng.uniform(-1, 1, (batch, L))

            elif distype =='gaussian':
                disx = rng.normal(0, a, (batch, L))
                disy = rng.normal(0, b, (batch, L))

            sites = sites = np.array( [ list(range(L)) for _ in range(batch)], dtype=int)

        else:
            
            sites = []

            if mode == 1:
                # this mode generate target maxlen
                for begin in range(0, L - maxlen + 1):

                    # we first choose a random starting position, after which the disorder will be generated in that region
                    
                    if num_site > 2:

                        for _ in range(maxcase):
                            newstate = np.concatenate( ([begin], sorted(rng.choice(np.arange(begin + 1,  begin + maxlen - 1), num_site - 2, replace=False)), [begin + maxlen -1] ) )
                            sites += [newstate]

                    elif num_site == 2:

                        for _ in range(maxcase):
                            newstate = np.array( [begin, begin + maxlen -1])
                            sites += [newstate]

                    else:

                        for _ in range(maxcase):
                            newstate = np.array ( [begin])
                            sites += [newstate]

                    
            
                print(len(sites))
                #print(sites)

            elif mode == 2:
                # this is the # of cases for each maxlen 
                #maxcase = batch // (L - num_site + 1)
                # case counter for each maxlen case
                casecnt = defaultdict(int)
                caselist = list(range(num_site, L + 1))

                # when there are still maxlen not examined, generate cases of that maxlen
                while caselist:
                    
                    cur_max = caselist[-1]

                    start = rng.choice(L - cur_max + 1, 1)
                    newstate = sorted(rng.choice(np.arange(start, start+cur_max), num_site, replace=False))
                    begin, end = min(newstate), max(newstate)

                    if casecnt[ end - begin + 1] < maxcase:
                        sites += [newstate]
                        casecnt[ end - begin + 1] += 1

                        if casecnt[ end - begin + 1] == maxcase:
                            print('maxlen {} OK'.format(end - begin + 1 ))
                            caselist.remove(end - begin + 1 )

            
            sites = np.array(sites)
            #generate the site positions
            #sites = np.array([ sorted(rng.choice(np.arange(start, start + maxlen), num_site, replace=False)) for _ in range(batch)])

            disx = np.zeros((batch, L))
            disy = np.zeros((batch, L))

            sign = np.array([[ -1 if np.random.random() < 0.5 else 1 for _ in range(num_site)] for _ in range(batch) ])

            newdisx =  rng.uniform(a, b, (batch, num_site))  * sign

            for b in range(batch):
                # make sure that the sites do not cross each other
                site = 0
                while site < num_site - 1:
                    
                    # if nearest neighbor and N - 1 extends past N
                    if sites[b][site] == sites[b][site + 1] - 1 and newdisx [b][ site]  > 1 + newdisx [b][ site + 1] :
                        disx[b][ sites[b][site] ] = 1 + newdisx [b][ site + 1] 
                        disx[b][ sites[b][site + 1] ] = newdisx [b][ site] - 1
                        site += 2

                    else:
                        disx[b][ sites[b][site] ] = newdisx [b][ site]
                        site += 1

                if site < num_site:
                    disx[b][ sites[b][-1] ] = newdisx [b][ -1]


    if len(disx.shape) == 1:
        disx = np.reshape( disx, (1, disx.shape[0]))

    if len(disy.shape) == 1:
        disy = np.reshape( disy, (1, disy.shape[0]))

    return disx, disy, sites


