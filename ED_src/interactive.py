import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
import matplotlib as mpl
from matplotlib.widgets import Slider
from matplotlib.widgets import TextBox
import os.path
from plots import *
from single_run import *


def interactive(S, sdict, occdict, balancestate, para):


    case = 0
    L = para['L']
    int_ee = para['int_ee']
    N = len(S)
    slide = 0
    mode = para['mode']

    total = 4
    fig, ax = plt.subplots(1,total, figsize = (total * 5, 5))

    if mode == 3:

        if slide == 1:
            loc = Slider(plt.axes([0.25, 0.05, 0.5, 0.02]), 'location', 0, L, valstep=1)
            dis = Slider(plt.axes([0.25, 0.0, 0.5, 0.02]), 'disorder', -1, 1 , valstep=0.01)

        else:
            loc = TextBox(plt.axes([0.25, 0.05, 0.5, 0.02]), 'location')
            dis = TextBox(plt.axes([0.25, 0.0, 0.5, 0.02]), 'disorder')

        
        disy = np.zeros((1, L))

        def update(val):

            disx = np.zeros((1, L))

            if slide == 1:
                site = int(loc.val)
                mag = dis.val

            else:
                site = int(loc.text)
                mag = float(dis.text)

            disx[0][site] = mag

            eigv, tcd, gpi, balance, energy = single_iteration(case, disx, disy, site, S,  sdict, occdict, balancestate, para)
            
            title = 'Plots for $\lambda$ = {}t, L = {}, N = {}'.format(int_ee, L, N)
            internal_plot(fig, ax, tcd, gpi, balance, energy, disx[0], title)



        if slide == 1:
            loc.on_changed(update)
            dis.on_changed(update)

        else:
            loc.on_submit(update)
            dis.on_submit(update)

        plt.show()

    else:
        if os.path.exists('tcd') and os.path.exists('disx') and os.path.exists('sites'):
            
            def ee(i, j):
                
                ex, z, zeta = para['ex'], para['z'], zeta['zeta']
                #ex = 0
                #zeta = 0
                
                if j - i == 1:
                    factor = 1 - ex

                else:
                    factor = 1

                return z * int_ee * factor / ( abs(j - i) + zeta)

            sites = np.loadtxt('sites', dtype=int)
            disx = np.loadtxt('disx')

            
            #print(idx)
            

            discard = []
            for i, dis in enumerate(disx):

                if np.count_nonzero(dis) == 2:
                    one, two = np.nonzero(dis)[0]
                    if one == two - 1 and dis[one] >= dis[two] + 1:
                        discard += [i]

            print(len(discard))
            sites = np.delete(sites, discard, axis=0)

            idx = [ comb[0] for comb in sorted( [ (i, val) for i, val in enumerate(sites)], key = lambda x: [ x[1][0], disx[ x[0] ][ x[1][0] ], x[1][1],  disx[ x[0] ][ x[1][1]]   ])]

            sites = np.delete(sites, discard, axis =0 )[idx]
            disx = np.delete(disx, discard, axis=0)[idx]

            
            #idx = np.array([y[1] for y in sorted( [(val, i) for i, val in enumerate(disx)], key=lambda x: [x[0][i] for i in range(L)])])

                
            tcd = np.delete(np.loadtxt('tcd'), discard, axis= 0)[idx]

            if not os.path.exists('gpi'):
                cases = len(tcd)
                tcd = tcd.reshape((cases, 924, 12))
                gpi = np.zeros((cases, 924))
                for case in range(cases):
                    for n in range(924):
                        for i in range(L):
                            for j in range(L):
                                gpi[case] += tcd[case, n, i] * tcd[case, n, j] *  ee(i, j) 

                np.savetxt('gpi', gpi)

            gpi = np.delete(np.loadtxt('gpi'), discard, axis= 0)[idx]
            energy = np.delete(np.loadtxt('energy'), discard, axis = 0)[idx]

            balance = np.delete(np.loadtxt('balance'), discard, axis =0)[idx]
            case = len(tcd)

            tcd = tcd.reshape((case, N, L))
            

            for row in range(len(tcd)):
                for n in range(N):
                    
                    '''
                    idx = np.argmax( np.abs(tcd[row, n, :]))
                    if abs(tcd[row - 1, n, idx] - tcd[row, n, idx]) > np.abs(tcd[row, n, idx]):

                        tcd[row, n, :] = -tcd[row, n, :]
                    ''' 
                    if tcd[row, n, 0] > 0:
                        tcd[row, n, :] = - tcd[row, n, :]

            

            mpl.rcParams['animation.ffmpeg_path'] = r'C:\\Users\\Ray\\Downloads\\ffmpeg-master-latest-win64-gpl\\ffmpeg-master-latest-win64-gpl\\bin\\ffmpeg.exe'

            lo = np.amin( tcd[:, 1, :])
            hi = np.amax( tcd[:, 1, :])

            def animate(i):
                
                curtcd = tcd[i, :, :]
                curgpi = gpi[i, :]
                curbalance = balance[i, :]
                curenergy = energy[i, :]

                dis = disx[i]

                #pos1, pos2 = ref[ i // (19 * 19)]

                #val1 = dis[pos1]
                #val2 = dis[pos2]
                
                print('frame {}'.format(i))
                title = '2 site disorder scan, site 1 at position {} with disorder {}, site 2 at position {} with disorder {}'.format(sites[i][0] + 1, disx[i][sites[i][0] ], sites[i][1] + 1, disx[i][ sites[i][1]])
                internal_plot(fig, ax, curtcd, curgpi, curbalance, curenergy, dis, title, lo, hi)

            cnt = 0
            ref = {}

            for i in range(L):
                for j in range(i + 1, L):
                    ref[ cnt ] = (i, j)
                    cnt += 1


            anim = FuncAnimation(fig, animate, frames= case, interval=200,  repeat=False)

            #plt.show()
            writervideo = animation.FFMpegWriter(fps=5)
            anim.save( 'timelapse.mp4', writer=writervideo)

        else:

            print('Data missing')


