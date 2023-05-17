import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from cnn_utils import *
from class_cnn import *
from metric_cnn import *


if __name__ == '__main__':



    # select which set of disorder parameters we want to look at. 
    # For cross comparison down in plot, enter multiple 
    # For function to work properly which_cases always need to be a list
    which_cases = [0.05]
    label = EnergyAllGap()

    for i, case in enumerate(which_cases):

        
        label.set_data_dir(case)
        label.init_model()
        label.set_model_dir(case)
        _, _, _, _, X_test, y_test = label.load_data()
        
        label.import_model()

        model = label.model

        y_pred = get_pred(X_test, model)
        
        label.visualize_pred(y_pred, y_test)
        avg, errabsavg, erravg, errabsavgstd, erravgstd = get_metric(y_pred, y_test)

