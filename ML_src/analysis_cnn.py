import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from cnn_utils import *

if __name__ == '__main__':


    
    # select which set of disorder parameters we want to look at. 
    # For cross comparison down in plot, enter multiple 
    # For function to work properly which_cases always need to be a list
    data_descriptions = ['4x4-0.3min']

    # inputs will be all the file names for the paras for the labels waiting to be analyzed
    input = sys.argv[1:]

    for i, data_description in enumerate(data_descriptions):
        
        multi_label(input, data_description=data_description)
