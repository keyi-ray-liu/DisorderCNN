from locale import ABDAY_1
import numpy as np
import os
import torch.nn as nn
import torch
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
from os.path import exists
from class_cnn import *
import sys
from metric_cnn import *



def batchify_data(x_data, y_data, batch_size):
    """Takes a set of data points and labels and groups them into batches."""
    # Only take batch_size chunks (i.e. drop the remainder)
    N = int(len(x_data) / batch_size) * batch_size
    batches = []
    for i in range(0, N, batch_size):
        batches.append({
            'x': torch.tensor(x_data[i:i + batch_size],
                              dtype=torch.float32),
            'y': torch.tensor(y_data[i:i + batch_size],
                               dtype=torch.float32)
        })
    return batches

def select_label():

    num = int(sys.argv[1])
    labels = {
        0: EnergyGSGap(),
        1: EnergyAllGap(),
        2: EnergyNearestNGSGap()
    }

    print(f"Model selected: {labels[num]}")
    return labels[num]
    

def get_metric(pred, y):
    
    avg = np.mean(pred, axis=0)

    
    erravg = (pred - y) / y
    abserravg = np.abs(erravg)

    return avg, np.mean(abserravg, axis=0), np.mean(erravg, axis=0),  np.std(abserravg, axis=0), np.std(erravg, axis=0)

def ploterr(workdirs):

    num = len(workdirs)

    if num > 1:
        fig, axes = plt.subplots(1, num, figsize = (15, 10))

    else:
        fig, ax = plt.subplots(figsize = (8, 10))
        axes = [ax]
    
    title = ['$\pm 0.45$', '$\pm 0.3$', '$\pm 0.15$']
    for i, workdir in enumerate(workdirs):
        trainerrs = np.loadtxt(workdir + 'trainerrs')
        validerrs = np.loadtxt(workdir + 'validerrs')
        # ref = np.arange(0, trainerrs.shape[1])
        # for i, trainerr in enumerate(trainerrs):
        #     ax.plot(ref, trainerr)
        #     ax.scatter( ref, trainerr, label='IPR {}'.format(i))

        cs = axes[i].imshow(trainerrs, origin='lower')
        
        axes[i].set_title('Train Error: Disorder at max ' + title[i])
        axes[i].set_xlabel('Training Epochs')
        axes[i].set_ylabel('Eigenstate')

        cbar = fig.colorbar(cs, ax=axes[i])
        cbar.set_label('Average Relative Error in IPR prediction')

    
    fig.suptitle(r'Training Error: Eigenstate vs. Training Epoch, $10\times10$, Single Particle State')
    fig.tight_layout()
    if len(workdirs) == 1:
        fig.savefig(workdirs[0] + 'TrainERR.pdf')

    else:
        fig.savefig('TraincompERR.pdf')

def plotanalysis(workdirs):

    num = len(workdirs)

    fig, ax = plt.subplots(figsize = (8, 10))

    label = ['$\pm 0.45$', '$\pm 0.3$', '$\pm 0.15$']
    c = ['r', 'g', 'b']
    for i, workdir in enumerate(workdirs):
        avg = np.loadtxt(workdir + 'testavg')
        std = np.loadtxt(workdir + 'teststd')

        ref = np.arange(0, avg.shape[0])
        # for i, trainerr in enumerate(trainerrs):
        #     ax.plot(ref, trainerr)
        #     ax.scatter( ref, trainerr, label='IPR {}'.format(i))

        ax.scatter(ref, avg, label = 'avg err' + label[i], c=c[i])
        ax.scatter(ref, std, marker = 'x', label = 'std' + label[i], c= c[i])
    
    ax.legend()
    ax.set_xlabel('eigenstate')
    ax.set_ylabel('Avg train error')
    ax.set_title(r'Prediction Error vs eigenstate, $10\times10$, Single Particle State')


    fig.savefig('FinalERR.pdf')

def plotcomp(workdir, ifsort, both):

    fig, ax = plt.subplots(2, 1, figsize = (8, 10))

    ed = np.loadtxt(workdir + 'sort{}testipr'.format(ifsort))

    files = []

    if both:
        models = ['MAPE', 'MSEdouble']

    else:
        models = ['MAPE']

    metrics = ['pred', 'absavg', 'avg']
    for metric in metrics:
        for model in models:

            file = np.loadtxt(workdir + '{}CNNsort{}test{}'.format(model, ifsort, metric))
            files.append(file)

    # clean

    def select(k):

        return 0 if k < len(models) else 1

    cl = ['r','g']
    l = ['-', '-', '--']

    for k, arr in enumerate(files):

        for i, val in enumerate(arr):
            if abs(val) <= 0.001 or abs(val) >= 0.999:
                
                if i > 0 and i < len(arr):

                    mid = (arr[i - 1] + arr[i + 1])/2
                    arr[i] = mid
                    ax[select(k)].scatter( i, mid, marker='x', c='black', s= 25  )
        

    ref = np.arange(1, 101)

    cnt = 0
    for metric in metrics:
        for model in models:
            ax[select(cnt)].plot(ref, files[cnt], label= model + metric, c=cl[cnt%2], ls=l[cnt//2])
            cnt += 1

    ax[0].plot( ref, ed, label='Actual') 

    ax[0].set_xlabel('Eigenstate (Sort = {}'.format(ifsort))
    ax[0].set_ylabel('Avg IPR ')
    ax[0].legend()
    ax[0].set_title('Average IPR (across 3000 test cases) Sort = {}'.format(ifsort))

    ax[1].legend()
    ax[1].set_xlabel('Eigenstate')
    ax[1].set_title('Error metrics')

    fig.tight_layout()
    plt.show()

def hist(workdir, ifsort, both):



    rawed = np.loadtxt(workdir + 'sort{}rawtestipr'.format(ifsort))
    rawMAPE = np.loadtxt(workdir + 'MAPECNNsort{}testrawpredall'.format(ifsort))
    
    skip = set()
    ipr = 100

    if both:

        startstack = 0
        for i in range(ipr):

            curraw = np.loadtxt( workdir + 'MSEdoubleCNNsort{}testrawpred{}'.format(ifsort, i))
            
            if curraw[0] < 0.001:
                skip.add(i)
            else:

                curraw = curraw.reshape((curraw.shape[0], 1))
                if startstack:
                    rawMSE = np.hstack( (rawMSE, curraw) )

                else:
                    rawMSE = curraw
                    startstack = 1

        rawMSEdiff = (rawMSE - rawed)
        rawMSEper = rawMSEdiff / rawed

    rawed = np.delete(rawed, list(skip), 1)
    rawMAPE = np.delete(rawMAPE, list(skip), 1)

    #print(rawed.shape, rawMAPE.shape, rawMSE.shape)

    rawMAPEdiff = (rawMAPE - rawed)
    rawMAPEper = rawMAPEdiff / rawed


    pdf = matplotlib.backends.backend_pdf.PdfPages( workdir + 'iprssort{}.pdf'.format(ifsort) )
    cnt = 0

    for i in range(ipr):

        if i not in skip:

            fig, ax = plt.subplots(3, 1, figsize=(8, 12))

            MAPEdiff = rawMAPEdiff[:, cnt]
            MAPEper = rawMAPEper[:, cnt]

            if both:
                MSEdiff = rawMSEdiff[:, cnt]
                MSEper = rawMSEper[:, cnt]
                ax[0].hist(MSEdiff.flatten(),200, (-0.8, 0.8), alpha=0.5, label='MSE')
                ax[1].hist(MSEper.flatten(),200, (-0.8, 0.8), alpha=0.5, label='MSE')
            

            ax[0].hist(MAPEdiff.flatten(), 200, (-0.8, 0.8), alpha=0.5, label='MAPE')
            ax[0].set_xlabel('IPR: (Prediction - Actual)')
            ax[0].legend()
            ax[0].set_ylabel('Counts')

            ax[1].hist(MAPEper.flatten(), 200, (-0.8, 0.8), alpha=0.5, label='MAPE')
            
            ax[1].set_xlabel('IPR: (Prediction - Actual) / Actual')
            ax[0].set_ylabel('Counts')

            ax[1].legend()

            ax[2].hist(rawed[:,cnt], 200, (0, 1), label='Actual')
            ax[2].set_xlabel('IPR: Actual')
            ax[2].legend()
            fig.suptitle('IPR {}'.format(i))
            fig.tight_layout()

            cnt += 1
            pdf.savefig(fig)

    pdf.close()
