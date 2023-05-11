import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from cnn_utils import *
from class_cnn import *


if __name__ == '__main__':



    # select which set of disorder parameters we want to look at. 
    # For cross comparison down in plot, enter multiple 
    # For function to work properly which_cases always need to be a list
    which_cases = [1]
    workdir = setworkdir(which_cases)
    ifsort = 1

    # since MSE trainings are a lot slower than all-to-all, this flag determines if we plot both
    both = 0

    if len(workdir) > 1:
        print('Error: can only plot one at a time')

    plotcomp(workdir[0], ifsort, both)
    hist(workdir[0], ifsort, both)
        
        