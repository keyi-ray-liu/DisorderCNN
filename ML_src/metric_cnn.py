import numpy as np
import os
from sklearn.model_selection import train_test_split
from class_cnn import *
from cnn_utils import *
import matplotlib.pyplot as plt

class Label():

    def __init__(self, name, description=''):
        self.name = name
        self.model_dir = ''
        self.data_dir = ''
        self.model = None

    def set_data_dir(self): 
        return NotImplementedError("label must define their data dir setting method")
    
    def set_model_dir(self, description=''):

        cwd = os.getcwd() + '/models/'
        if not os.path.exists(cwd):
            os.mkdir(cwd)
        
        label_name = self.__class__.__name__
        model_name = self.model.__class__.__name__
        self.model_dir = cwd + model_name + label_name + str(description)
        
    def load_data(self):
        return NotImplementedError("label must define their data loading method")
    
    def process_x(self):
        return NotImplementedError("label must define their x process method")
    
    def process_y(self):
        return NotImplementedError("label must define their y process method")
    
    def train_model(self):
        return NotImplementedError("label must define their training method")
    
    def init_model(self):
        return NotImplementedError("label must define their model init method")

    def import_model(self):

        model_dir  = self.model_dir

        try:
            model = torch.load(model_dir)
        except:
            raise ValueError("THe model defined in model_dir does not exist")
        
        self.model = model

    def visualize_pred(self):
        return NotImplementedError("label must define their visualization method")
    
    def export_model(self):
        model_dir = self.model_dir
        torch.save(self.model, model_dir)


class Ipr(Label):

    def train(self,data):
        return None

class Gpi(Label):

    def __init__(self, name='gpi', if_sort=0):
        self.name = name
        self.if_sort = if_sort

    def set_data_dir(self, case_id): 

        cwd = os.getcwd() + '/batch{}/'.format(str(case_id))
        if not os.path.exists(cwd):
            os.mkdir(cwd)
        
        self.data_dir = cwd

    def load_data(self):
        
        dir = self.data_dir
        ifsort = self.if_sort
        key = self.name

        disx = np.loadtxt(dir + 'disx')
        disy = np.loadtxt(dir + 'disy')
        
        if not ifsort:
            arr = np.loadtxt(dir + key)

        else:

            if os.path.exists(dir + key + 'sort'):
                arr = np.loadtxt( dir + key + 'sort')

            else:
                arr = np.loadtxt(dir + key)
                arr = np.sort(arr)
                np.savetxt(dir + key + 'sort', arr)
                

        arr = np.log(arr)

        #for the moment we assume a square lattice
        # get dim

        dis = np.zeros( (disx.shape[0], 2, disx.shape[1]))
        dis[:, 0, :] = disx
        dis[:, 1, :] = disy

        # currently only train for ipr of gs_subtracted 

        X_train, X_test, y_train, y_test = train_test_split(dis, arr, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.4, random_state=55)

        return X_train, y_train, X_val, y_val, X_test, y_test

        
    def init_model(self):
        self.model = GPICNN()



class Energy(Label):

    def __init__(self, name='energy', cutoff=30):
        self.name = name
        self.cutoff = cutoff


    def process_x(self):

        dir = self.data_dir

        disx = np.loadtxt(dir + 'disx')
        disy = np.loadtxt(dir + 'disy')
        
        dis = np.zeros( (disx.shape[0], 2, disx.shape[1]))
        dis[:, 0, :] = disx
        dis[:, 1, :] = disy

        return dis
    
    def set_data_dir(self, case_id): 

        cwd = os.getcwd() + '/batch{}/'.format(str(case_id))
        if not os.path.exists(cwd):
            os.mkdir(cwd)
        
        self.data_dir = cwd

    def load_data(self):
        
        
        arr = self.process_y()
        dis = self.process_x()
        
        # currently only train for ipr of gs_subtracted 

        X_train, X_test, y_train, y_test = train_test_split(dis, arr, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.4, random_state=55)

        return X_train, y_train, X_val, y_val, X_test, y_test


    def train_model(self, train_data, dev_data, lr=0.03, momentum=0.9, weight_decay = 0.02, n_epochs=30):
        """Train a model for N epochs given data and hyper-params."""
        
        model = self.model

        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay = weight_decay)

        trainerr = np.zeros(n_epochs)
        validerr = np.zeros(n_epochs)

        print("-------------- Training: METRIC = Energy \n")
        
        for i, epoch in enumerate(range(1, n_epochs + 1)):
            print("-------------\nEpoch {}:\n".format(epoch))

            # Run **training***
            err = run_epoch(train_data, model.train(), optimizer)
            trainerr[i] = err
            print('Train | avg percent error : {:.6f} '.format(err))

            # Run **validation**
            err = run_epoch(dev_data, model.eval(), optimizer)
            validerr[i] = err
            print('Valid | avg percent error : {:.6f} '.format(err))

            # Save model
            
        self.model = model
        return trainerr, validerr
    

    def visualize_pred(self, y_pred, y_test):
        
        sample = 5
        fig, ax = plt.subplots( 1, sample, figsize=(10, 10 * sample))
        inds = np.random.choice( np.arange(y_pred.shape[0]), sample )

        pred = y_pred[inds]
        ref = y_test[inds]

        x = np.arange(1, y_pred.shape[-1] + 1)

        for i in range(sample):

            ax[i].scatter(x, pred[i], label='pred' )
            ax[i].scatter(x, ref[i], label='true')
            ax[i].legend()

        plt.show()

class EnergyGSGap(Energy):

    def process_y(self):

        dir = self.data_dir
        key = self.name
        cutoff = self.cutoff

        out = dir + key + 'cutoff{}gsgap'.format(cutoff)

        if os.path.exists(out):
            arr = np.loadtxt( out)

        else:
            arr = np.loadtxt(dir + key)
            arr = arr[:, :cutoff]
            arr = arr - np.reshape( np.repeat(arr[:, 0], arr.shape[-1]), arr.shape)
            arr = arr[:, 1:]

            np.savetxt(out, arr)

        return arr

    def init_model(self):
        self.model = Energy1DCNN(self.cutoff - 1)

class EnergyAllGap(Energy):

    def process_y(self):

        dir = self.data_dir
        key = self.name
        cutoff = self.cutoff

        out = dir + key + 'cutoff{}allgap'.format(cutoff)

        if os.path.exists(out):
            arr = np.loadtxt( out)

        else:
            arr = np.loadtxt(dir + key)
            arr = arr[:, :cutoff]
            arr = arr[:, 1:] - arr[:, :-1]

            np.savetxt(out, arr)

        return arr

    def init_model(self):
        self.model = Energy1DCNN(self.cutoff - 1)
