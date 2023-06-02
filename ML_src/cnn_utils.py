from locale import ABDAY_1
import numpy as np
import os
import torch.nn as nn
import torch
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
from os.path import exists
from class_cnn import *
import sys
from metric_cnn import *
import time
from collections import Counter


class Energy_multi_processor():
    """processor class for multiple labels"""
    def __init__(self, inputs) -> None:
        
        model_dir = 'models/'
        parameters = []

        for input in inputs:
            with open( model_dir + input, 'r') as f:
                new = json.load(f)
            parameters.append(new)

        self.parameters = parameters

    def get_parameters(self):
        print(self.parameters)

    def visualize_pred(self, case):
        
        row, col = 2, 5
        s = 15
        inds = []
        sample = row * col
        plot_dir = os.getcwd() + '/plots/' + '_'.join([ parameter['label'] for parameter in self.parameters ]) + '.png'
        fig, ax = plt.subplots( row,  col, figsize=( 4 * col, 5 * row))

        for i, parameter in enumerate(self.parameters):
            
            label = select_label(parameter['internal_label'], parameter['id'])
            description = "case{}".format(case)

            if isinstance(label,GSGap_GSW_MSE ):
                label.set_weights(parameter['weight'])

            label.set_data_dir(case_id=case)
            label.set_model_dir(description=description)
            _, _, _, _, X_test, y_test = label.load_data()
            
            label.import_model()
            y_pred = label.get_pred(X_test)

            if inds == []:
                inds = np.random.choice( np.arange(y_pred.shape[0]), sample )

            pred = y_pred[inds]
            ref = y_test[inds]

            x = np.arange(1, y_pred.shape[-1] + 1)
            cnt = 0

            for i in range(row):
                for j in range(col):

                    ax[i][j].scatter(x, pred[cnt], label=parameter['weight'], s=s)

                    cnt += 1

        cnt = 0
        for i in range(row):
            for j in range(col):

                ax[i][j].scatter(x, ref[cnt], label='true', marker='x', s=s)
                
                cnt += 1

        ax[-1][-1].legend()
        fig.savefig(plot_dir)


   

def batchify_data(x_data, y_data, batch_size):
    """Takes a set of data points and labels and groups them into batches."""
    # Only take batch_size chunks (i.e. drop the remainder)
    N = int(len(x_data) / batch_size) * batch_size
    batches = []
    for i in range(0, N, batch_size):
        batches.append({
            'x': torch.tensor(x_data[i:i + batch_size],
                              dtype=torch.float32),
            'y': torch.tensor(y_data[i:i + batch_size],
                               dtype=torch.float32)
        })
    return batches

def select_label( arg, ID):

    num = int(arg)
    labels = {
        0: GSGap_MAPE(ID),
        1: AllGap_MAPE(ID),
        2: NearestN_GSGap_MAPE(ID),
        3: GSGap_MSE(ID),
        4: GSGap_GSW_MSE(ID),
        5: AllGap_MSE(ID)
    }
    
    try:
        print(f"Model selected: {labels[num]}")
        return labels[num]
    except KeyError:
        raise KeyError("not a valid label index")
    

def hist(res):

    fig, ax = plt.subplots()

    ax.bar(res.keys(), res.values(), width=0.1)
    plt.show()


def compare(y_pred, y_test):
    
    res = np.round(np.sum( ( y_pred - y_test) ** 2, axis = 1), decimals=1)

    res = Counter(res)
    hist(res)

def single_label( label :Label, **kwargs):

    case = kwargs.pop('case')
    
    if isinstance(label, GSGap_GSW_MSE):
        weight = kwargs.pop('weight')
        label.set_weights(weight)

    description = "case{}".format(case)
    label.set_data_dir(case_id =case)
    label.set_model_dir(description=description)
    label.set_plot_dir(description=description)

    _, _, _, _, X_test, y_test = label.load_data()
    
    label.import_model()
    y_pred = label.get_pred(X_test)
    
    label.visualize_pred(y_pred, y_test)
    compare(y_pred, y_test)


def multi_label( inputs, case):

    # get test data set. Has to be on the same data set, so
    processor = Energy_multi_processor(inputs)

    #processor.get_parameters()
    processor.visualize_pred(case)





