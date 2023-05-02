import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from cnn_utils import *
from class_cnn import *


if __name__ == '__main__':


    all = 1

    if not all:
        num_ipr = 100
        iprs = range(num_ipr)
    else:
        iprs = ['all']

    # select which set of disorder parameters we want to look at. 
    # For cross comparison down in plot, enter multiple 
    # For function to work properly which_cases always need to be a list
    which_cases = [1]
    ifsort = 1
    workdir = setworkdir(which_cases)

    for i, case in enumerate(which_cases):
        
        read = checktestfile(workdir[i])
        if not read:
            erravgs = []
            errabsavgs = []
            erravgstds = []
            errabsavgstds = []
            preds = []
            
            prefix = workdir[i] + model.__class__.__name__ + 'sort{}'.format(ifsort)

            for ipr in iprs:
                _, _, _, _, X_test, y_test = loaddata(case, ipr, ifsort)
                
                # path of the model
                _, path = get_path(ipr, workdir[i], ifsort)
                model = load_model(path)

                y_pred = get_pred(X_test, model)

                np.savetxt( prefix + 'testrawpred{}'.format(ipr), y_pred)
                avg, errabsavg, erravg, errabsavgstd, erravgstd = get_metric(y_pred, y_test)

                #print(avg, std)
                preds.append(avg)
                errabsavgs.append(errabsavg)
                erravgs.append(erravg)
                errabsavgstds.append(errabsavgstd)
                erravgstds.append(erravgstd)

        
            np.savetxt( prefix + 'testpred', np.array(preds))
            np.savetxt( prefix + 'testabsavg', np.array(errabsavgs))
            np.savetxt( prefix + 'testavg', np.array(erravgs))
            np.savetxt( prefix + 'testabsavgstd', np.array(errabsavgstds))
            np.savetxt( prefix + 'testavgstd', np.array(erravgstds))
            
            if all:
                np.savetxt( workdir[i] + 'sort{}rawtestipr'.format(ifsort), y_test)
                np.savetxt( workdir[i] + 'sort{}testipr'.format(ifsort), np.mean(y_test, axis=0))