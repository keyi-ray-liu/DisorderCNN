import numpy as np
import os
import sys

def gen():
    batch = 500
    Lx = 12
    Ly = 1
    L = Lx * Ly
    flag = 'uniform'
    cwd = os.getcwd() + '/'
    target = sys.argv[1]

    seeds = (10 ** 9 * np.random.random(4)).astype(int)
    x_seed, y_seed, x_sign_seed, y_sign_seed = seeds

    print( seeds)
    x_rng = np.random.default_rng(x_seed)
    y_rng = np.random.default_rng(y_seed)
    x_sign_rng = np.random.default_rng(x_sign_seed)
    y_sign_rng = np.random.default_rng(y_sign_seed)

    dir = cwd + target + '/'

    a = 0.45
    b = 0.45

    if flag == 'uniform':
        disx = a * x_rng.uniform(-1, 1, (batch, L))
        disy = b * y_rng.uniform(-1, 1, (batch, L))

    elif flag =='gaussian':
        disx = x_rng.normal(0, a, (batch, L))
        disy = y_rng.normal(0, b, (batch, L))

    elif flag == 'limited':
        sign_x = np.sign(x_sign_rng.uniform(-1, 1, (batch, L)))
        sign_y = np.sign(y_sign_rng.uniform(-1, 1, (batch, L)))

        raw_x = x_rng.uniform(0.4, 0.45, (batch, L))
        raw_y = y_rng.uniform(0.4, 0.45, (batch, L))
        
        disx = raw_x * sign_x
        disy = raw_y * sign_y


    sites = sites = np.array( [ list(range(L)) for _ in range(batch)], dtype=int)

    np.savetxt(dir + "seeds", seeds)

    np.savetxt( dir + 'disx', disx)
    np.savetxt( dir + 'disy', disy)
    np.savetxt( dir + 'sites', sites, fmt= '%i')


if __name__ == '__main__':
    gen()