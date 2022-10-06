import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from cnn_utils import *



if __name__ == '__main__':

    all = 1

    if not all:

        batch_size = 100
        n_epochs = 20
        num_ipr = 100
        iprs = range(num_ipr)
        trainerrs = np.zeros((num_ipr, n_epochs))
        validerrs = np.zeros((num_ipr, n_epochs))

    else:
        batch_size = 20
        n_epochs = 50
        iprs = ['all']

    lr = 0.03
    decay = 0.02
    momentum = 0.9

    # select which set of disorder parameters we want to look at. 
    # For cross comparison down in plot, enter multiple 
    # For function to work properly which_cases always need to be a list
    which_cases = [1]
    ifsort = 1
    workdir = setworkdir(which_cases)

    #print(workdir)

    for i, case in enumerate(which_cases):
        read = checkerrfile(workdir[i])

        if not read:
            for ipr in iprs:
                # for the training script we do not need test data
                X_train, y_train, X_val, y_val, _, _ = loaddata(case, ipr, ifsort)

                Train = batchify_data(X_train, y_train, batch_size)
                Val = batchify_data(X_val, y_val, batch_size)
                #Test = batchify_data(X_test, y_test, batch_size)

                model_temp, path = get_path(ipr, workdir[i], ifsort)


                trainerr, validerr = train_model(Train, Val, model_temp, ipr, path, lr=lr, \
                    weight_decay = decay, momentum = momentum, n_epochs=n_epochs)

                if not all:
                    trainerrs[ipr] = trainerr
                    validerrs[ipr] = validerr

                else:
                    trainerrs = trainerr
                    validerrs = validerr

            np.savetxt(workdir[i] + 'trainerrs', trainerrs)
            np.savetxt(workdir[i] + 'validerrs', validerrs)
