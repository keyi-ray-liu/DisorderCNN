import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from cnn_utils import *

if __name__ == '__main__':



    # select which set of disorder parameters we want to look at. 
    # For cross comparison down in plot, enter multiple 
    # For function to work properly which_cases always need to be a list
    which_cases = [0.05]
    label = select_label()

    for i, case in enumerate(which_cases):

        
        label.set_data_dir(case)
        label.init_model()
        label.set_model_dir(case)
        label.set_plot_dir(case)
        _, _, _, _, X_test, y_test = label.load_data()
        
        label.import_model()
        y_pred = label.get_pred(X_test)
        
        label.visualize_pred(y_pred, y_test)
        avg, errabsavg, erravg, errabsavgstd, erravgstd = get_metric(y_pred, y_test)

