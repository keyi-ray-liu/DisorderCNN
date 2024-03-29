import json
import collections

def gen_hopping(Lx, Ly, t):

    L = Lx * Ly
    d = collections.defaultdict(list)

    for s in range(L):

        # if site not on the edge
        if s < L - 1 and s % Lx != Lx - 1:
            d[s].append( [s + 1, t])

        if s < L - Lx:
            d[s].append( [s + Lx, t])

    
    with open('nn.json', 'w') as f:
        json.dump(d, f)

    with open('nn.json', 'r') as f:
        nn_raw = json.load(f)

        nn = collections.defaultdict(list)
        for key in nn_raw:
            nn[ int(key) ] = nn_raw[key]

    #print(nn)

if __name__ == '__main__':

    gen_hopping(4, 4, -0.001)