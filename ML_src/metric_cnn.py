import numpy as np
import os
from sklearn.model_selection import train_test_split
from class_cnn import *

class Label():

    def __init__(self, name):
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
        
        class_name = self.model.__class__.__name__
        self.model_dir = cwd + class_name + description
        
    def load_data(self):
        return NotImplementedError("label must define their data loading method")
    
    def init_model(self):
        return NotImplementedError("label must define their model init method")

    def import_model(self, path):
        model = torch.load(path)
        self.model = model

    def save_model(self, model):
        self.model = model

    def export_model(self):
        path = self.model_dir
        torch.save(self.model, path)


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

    def __init__(self, name='energy', cutoff=30, gs_subtracted=1):
        self.name = name
        self.cutoff = cutoff
        self.gs_subtracted = gs_subtracted

    def set_data_dir(self, case_id): 

        cwd = os.getcwd() + '/batch{}/'.format(str(case_id))
        if not os.path.exists(cwd):
            os.mkdir(cwd)
        
        self.data_dir = cwd

    def load_data(self):
        
        dir = self.data_dir
        key = self.name
        cutoff = self.cutoff
        gs_subtracted = self.gs_subtracted

        disx = np.loadtxt(dir + 'disx')
        disy = np.loadtxt(dir + 'disy')
        
        out = dir + key + 'cutoff{}gssubtracted{}'.format(cutoff, gs_subtracted)

        if os.path.exists(out):
            arr = np.loadtxt( out)

        else:
            arr = np.loadtxt(dir + key)
            arr = arr[:, :cutoff]

            if gs_subtracted:
                arr = arr - np.reshape( np.repeat(arr[:, 0], arr.shape[-1]), arr.shape)
                arr = arr[:, 1:]

            np.savetxt(out, arr)
        
        dis = np.zeros( (disx.shape[0], 2, disx.shape[1]))
        dis[:, 0, :] = disx
        dis[:, 1, :] = disy

        # currently only train for ipr of gs_subtracted 

        X_train, X_test, y_train, y_test = train_test_split(dis, arr, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.4, random_state=55)

        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def init_model(self):
        self.model = Energy1DCNN(self.cutoff - self.gs_subtracted)

