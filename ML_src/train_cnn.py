import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from cnn_utils import *


if __name__ == '__main__':


    batch_size = 10
    n_epochs = 10

    lr = 0.07
    decay = 0.02
    momentum = 0.8

    weights = [4.0]
    # select which set of disorder parameters we want to look at. 
    # For cross comparison down in plot, enter multiple 
    # For function to work properly which_cases always need to be a list
    data_descriptions = ['4x4-0.3min']
    
    #label = Gpi(if_sort = 1)
    internal_label = int(sys.argv[1])

    

    for weight in weights:
        for i, data_description in enumerate(data_descriptions):
            
            ID = str(int(time.time()/60))
            label : Label = select_label(internal_label, ID)
            label.set_data_dir(data_description=data_description)
            label.set_model_dir()

            if isinstance(label, GSGap_GSW_MSE):
                label.set_weights(weight)
                
            # for the training script we do not need test data
            X_train, y_train, X_val, y_val, _, _ = label.load_data()

            Train = batchify_data(X_train, y_train, batch_size)
            Val = batchify_data(X_val, y_val, batch_size)
            #Test = batchify_data(X_test, y_test, batch_size)

            trainerr, validerr = label.train_model(Train, Val, lr=lr, weight_decay = decay, momentum = momentum, n_epochs=n_epochs)

            
        
            parameter = {
                'batch_size' : batch_size,
                'n_epochs' : n_epochs,
                'lr' : lr,
                'decay' : decay,
                'momentum' : momentum,
                'model' : label.get_model_name(),
                'label' : label.get_label_name(),
                'internal_label' : internal_label,
                'id' : label.get_id(),
                'trained_on': data_description,
                'activation' : label.get_activation()
            }

            if isinstance(label, GSGap_GSW_MSE):
                parameter['weight'] = weight

            label.export_model(parameter)

