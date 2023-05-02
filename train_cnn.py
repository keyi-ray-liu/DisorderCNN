import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from cnn_utils import *



if __name__ == '__main__':


    batch_size = 100
    n_epochs = 30
    trainerrs = np.zeros(n_epochs)
    validerrs = np.zeros(n_epochs)

    lr = 0.03
    decay = 0.02
    momentum = 0.9

    # select which set of disorder parameters we want to look at. 
    # For cross comparison down in plot, enter multiple 
    # For function to work properly which_cases always need to be a list
    which_cases = [1]
    ifsort = 1
    workdir = setworkdir(which_cases)
    key = 'gpi'

    if key == 'tcd' and ifsort:
        print('Incompatible options, exiting')
        exit()

    #print(workdir)

    for i, case in enumerate(which_cases):
        read = checkerrfile(workdir[i])

        if not read:

            # for the training script we do not need test data
            X_train, y_train, X_val, y_val, _, _ = loaddata(case, key, ifsort)

            Train = batchify_data(X_train, y_train, batch_size)
            Val = batchify_data(X_val, y_val, batch_size)
            #Test = batchify_data(X_test, y_test, batch_size)

            model_temp, path = get_path(key, workdir[i], ifsort)


            trainerr, validerr = train_model(Train, Val, model_temp, key, path, lr=lr, \
                weight_decay = decay, momentum = momentum, n_epochs=n_epochs)



            trainerrs = trainerr
            validerrs = validerr

            np.savetxt(workdir[i] + 'trainerrs', trainerrs)
            np.savetxt(workdir[i] + 'validerrs', validerrs)
