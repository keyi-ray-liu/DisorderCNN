from locale import ABDAY_1
import numpy as np
import os
import torch.nn as nn
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
from os.path import exists
from class_cnn import *

class MAPELoss(nn.Module):
    def __init__(self):
        super(MAPELoss, self).__init__()
 
    def forward(self, inputs, targets):        
        
        MAPE = torch.mean( torch.abs( inputs - targets)/targets)
        
        return MAPE


def setworkdir(which):

    ret = []

    for num in which:
        cwd = os.getcwd() + '/batch{}cnn/'.format(str(num))

        if not exists(cwd):
            os.mkdir(cwd)

        ret += [cwd]

    return ret


def checkerrfile(path):
    return exists( path  + 'trainerrs')

def checktestfile(path):
    return exists( path + 'testavg')

def select_model(ipr):

    if ipr != 'all':
        #model = MSEdoubleCNN()
        model = MSELarge()

    else:
        model = MAPECNN()

    return model

def get_path(ipr, workdir, ifsort):

    model_temp = select_model(ipr)
    name = name = model_temp.__class__.__name__
    path = workdir + 'sort{}{}DisReg{}.pt'.format(ifsort, name, ipr)

    return model_temp, path

def load_model(path):
    model = torch.load(path)
    return model

def loaddata(key, idx, ifsort):

    cwd = os.getcwd()
    key = str(key)
    dir = cwd + '/' +  'batch' + key + '/'

    disx = np.loadtxt(dir + 'disx')
    disy = np.loadtxt(dir + 'disy')

    if not ifsort:
        ipr = np.loadtxt(dir + 'ipr')

    else:

        if os.path.exists(dir + 'iprsort'):
            ipr = np.loadtxt( dir + 'iprsort')

        else:
            ipr = np.loadtxt(dir + 'ipr')
            ipr = np.sort(ipr)
            np.savetxt('iprsort', ipr)
            
    #for the moment we assume a square lattice
    # get dim
    dim = int(np.rint(np.sqrt( disx.shape[-1])))

    disx = disx.reshape( ( disx.shape[0], dim, dim))
    disy = disy.reshape ( (disy.shape[0], dim, dim))

    dis = np.zeros( (disx.shape[0], 2, dim, dim))
    dis[:, 0, :, :] = disx
    dis[:, 1, :, :] = disy

    # currently only train for ipr of GS 
    if idx != 'all':
        ipr = ipr[:, idx].reshape((ipr.shape[0], 1))

    X_train, X_test, y_train, y_test = train_test_split(dis, ipr, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.4, random_state=55)


    return X_train, y_train, X_val, y_val, X_test, y_test

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

def train_model(train_data, dev_data, model, ipr, path, lr=0.03, momentum=0.9, weight_decay = 0.02, n_epochs=30):
    """Train a model for N epochs given data and hyper-params."""
   
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay = weight_decay)

    trainerr = np.zeros(n_epochs)
    validerr = np.zeros(n_epochs)

    print("-------------- Training: IPR = {} \n".format(ipr))
    
    for i, epoch in enumerate(range(1, n_epochs + 1)):
        print("-------------\nEpoch {}:\n".format(epoch))

        # Run **training***
        err = run_epoch(train_data, model.train(), optimizer, ipr)
        trainerr[i] = err
        print('Train | avg percent error : {:.6f} '.format(err))

        # Run **validation**
        err = run_epoch(dev_data, model.eval(), optimizer, ipr)
        validerr[i] = err
        print('Valid | avg percent error : {:.6f} '.format(err))

        # Save model
        torch.save(model, path)

    return trainerr, validerr
    
def cal_error(out, y):
    out = out.detach().numpy()
    y = y.detach().numpy()
    return np.mean( np.abs(out - y)/ y)

def run_epoch(data, model, optimizer, ipr):
    """Train model for one pass of train data, and return loss, acccuracy"""


    # training label
    is_training = model.training
    percenterror = []

    if ipr == 'all':
        loss_function = MAPELoss()

    else:
        loss_function = nn.MSELoss()
    # Iterate through batches

    for batch in tqdm(data):
        # Grab x and y
        x, y = batch['x'], batch['y']

        # get prediction
        out = model(x)

        #print('out:', out, 'y:', y)

        percenterror.append( cal_error(out, y))
        # If training, do an update.
        if is_training:
            optimizer.zero_grad()
            joint_loss = loss_function(out, y)
            joint_loss.backward()
            optimizer.step()

    # Calculate epoch level scores
    avg_percenterror = np.mean(percenterror)
    return avg_percenterror


def get_pred(X, model):

    model.eval()

    X = torch.tensor(X, dtype=torch.float32)
    pred = model(X).detach().numpy()

    return pred

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

def plotcomp(workdir):

    fig, ax = plt.subplots(2, 1, figsize = (8, 10))

    ed = np.loadtxt(workdir + 'testipr')

    files = []

    models = ('MAPE', 'MSEdouble')
    metrics = ('pred', 'absavg', 'avg')
    for metric in metrics:
        for model in models:

            file = np.loadtxt(workdir + '{}CNNtest{}'.format(model, metric))
            files.append(file)

    # clean

    def select(k):

        return 0 if k < 2 else 1

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

    ax[0].set_xlabel('Eigenstate')
    ax[0].set_ylabel('Avg IPR')
    ax[0].legend()
    ax[0].set_title('Average IPR (across 3000 test cases)')

    ax[1].legend()
    ax[1].set_xlabel('Eigenstate')
    ax[1].set_title('Error metrics')

    fig.tight_layout()
    plt.show()

def hist(workdir):

    

    rawed = np.loadtxt(workdir + 'rawtestipr')
    rawMAPE = np.loadtxt(workdir + 'MAPECNNtestrawpredall')
    
    ipr = 100 
    startstack = 0
    skip = []
    for i in range(ipr):

        curraw = np.loadtxt( workdir + 'MSEdoubleCNNtestrawpred{}'.format(i))
        
        if curraw[0] < 0.01:
            skip += [i]
        else:

            curraw = curraw.reshape((curraw.shape[0], 1))
            if startstack:
                rawMSE = np.hstack( (rawMSE, curraw) )

            else:
                rawMSE = curraw
                startstack = 1

    rawed = np.delete(rawed, skip, 1)
    rawMAPE = np.delete(rawMAPE, skip, 1)

    #print(rawed.shape, rawMAPE.shape, rawMSE.shape)

    rawMAPEdiff = (rawMAPE - rawed)
    rawMSEdiff = (rawMSE - rawed)

    rawMAPEper = rawMAPEdiff / rawed
    rawMSEper = rawMSEdiff / rawed

    pdf = matplotlib.backends.backend_pdf.PdfPages( workdir + 'iprs.pdf' )
    cnt = 0

    for i in range(ipr):

        if i not in skip:

            fig, ax = plt.subplots(3, 1, figsize=(8, 12))

            MAPEdiff = rawMAPEdiff[:, cnt]
            MSEdiff = rawMSEdiff[:, cnt]
            MAPEper = rawMAPEper[:, cnt]
            MSEper = rawMSEper[:, cnt]

            ax[0].hist(MAPEdiff.flatten(), 200, (-0.8, 0.8), alpha=0.5, label='MAPE')
            ax[0].hist(MSEdiff.flatten(),200, (-0.8, 0.8), alpha=0.5, label='MSE')

            ax[0].set_xlabel('IPR: (Prediction - Actual)')
            ax[0].legend()
            ax[0].set_ylabel('Counts')

            ax[1].hist(MAPEper.flatten(), 200, (-0.8, 0.8), alpha=0.5, label='MAPE')
            ax[1].hist(MSEper.flatten(),200, (-0.8, 0.8), alpha=0.5, label='MSE')
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
