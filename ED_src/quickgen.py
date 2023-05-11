import numpy as np
import os
import sys

def gen():
    batch = 5000
    Lx = 10
    Ly = 10
    L = Lx * Ly
    flag = 'uniform'
    cwd = os.getcwd() + '/'
    target = sys.argv[1]

    dir = cwd + target + '/'
    seed = 1

    rng = np.random.default_rng(seed)

    a = 0.45
    b = 0.45
    if flag == 'uniform':
        disx = a * rng.uniform(-1, 1, (batch, L))
        disy = b * rng.uniform(-1, 1, (batch, L))

    elif flag =='gaussian':
        disx = rng.normal(0, a, (batch, L))
        disy = rng.normal(0, b, (batch, L))

    sites = sites = np.array( [ list(range(L)) for _ in range(batch)], dtype=int)

    np.savetxt( dir + 'disx', disx)
    np.savetxt( dir + 'disy', disy)
    np.savetxt( dir + 'sites', sites, fmt= '%i')
    
    with open( dir + 'genpara', 'w') as f:
        f.write( 'xmax' + str(a) + 'ymax' + str(b) + flag + str(seed) )

if __name__ == '__main__':
    gen()