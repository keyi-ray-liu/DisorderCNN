import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from cnn_utils import *
from metric_cnn import *



if __name__ == '__main__':


    batch_size = 100
    n_epochs = 30

    lr = 0.03
    decay = 0.02
    momentum = 0.9

    # select which set of disorder parameters we want to look at. 
    # For cross comparison down in plot, enter multiple 
    # For function to work properly which_cases always need to be a list
    which_cases = [0.05]
    label = Gpi(if_sort = 1)


    for i, case in enumerate(which_cases):


        label.set_data_dir(case)
        # for the training script we do not need test data
        X_train, y_train, X_val, y_val, _, _ = label.load_data()
        label.init_model()

        Train = batchify_data(X_train, y_train, batch_size)
        Val = batchify_data(X_val, y_val, batch_size)
        #Test = batchify_data(X_test, y_test, batch_size)

        
        trainerr, validerr, model = train_model(Train, Val, label, lr=lr, \
            weight_decay = decay, momentum = momentum, n_epochs=n_epochs)

        label.set_model_dir()
        label.save_model(model)
        label.export_model()

        np.savetxt(label.workdirs + 'trainerrs', trainerr)
        np.savetxt(label.workdirs + 'validerrs', validerr)
