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
    control = int(sys.argv[1])
    label = select_label( control)

    for i, case in enumerate(which_cases):
        
        if control >= 0:
            single_label(case, label)

        else:
            multi_label(case, label)
