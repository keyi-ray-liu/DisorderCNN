import numpy as np
import os
from sklearn.model_selection import train_test_split
from class_cnn import *
import matplotlib.pyplot as plt
from tqdm import tqdm

class Label():

    def __init__(self, name, description=''):
        self.name = name
        self.model_dir = ''
        self.data_dir = ''
        self.plot_dir = ''
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

    def set_plot_dir(self, description=''):

        cwd = os.getcwd() + '/plots/'
        if not os.path.exists(cwd):
            os.mkdir(cwd)
        
        label_name = self.__class__.__name__
        model_name = self.model.__class__.__name__
        self.plot_dir = cwd + model_name + label_name + str(description) + ".png"
        
    def load_data(self):
        
        y = self.process_y()
        x = self.process_x()
        
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.4, random_state=55)

        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def process_x(self):
        return NotImplementedError("label must define their x process method")
    
    def process_y(self):
        return NotImplementedError("label must define their y process method")
    
    def train_model(self):
        return NotImplementedError("label must define their training method")
    
    def init_model(self):
        return NotImplementedError("label must define their model init method")

    def get_pred(self):
        return NotImplementedError("label must define their prediction method")

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


class CNNBase(Label):

    def process_x(self):

        dir = self.data_dir

        disx = np.loadtxt(dir + 'disx')
        disy = np.loadtxt(dir + 'disy')
        
        dis = np.zeros( (disx.shape[0], 2, disx.shape[1]))
        dis[:, 0, :] = disx
        dis[:, 1, :] = disy

        return dis
    
class NearestNeighborBase(Label):

    def process_x(self):

        dir = self.data_dir

        disx = np.loadtxt(dir + 'disx')
        disy = np.loadtxt(dir + 'disy')
        
        dis = np.exp( np.sqrt(( disx[:, 1:] + 1 - disx[:, :-1]) **2 + (disy[:, 1:] - disy[:, :-1]) ** 2 ) - 1)

        return dis

class GSGapBase(Label):

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
    
class AllGapBase(Label):

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

class TorchTrainerBase(Label):

    def cal_error(self, out, y):
        out = out.detach().numpy()
        y = y.detach().numpy()
        return np.mean( np.abs ((out - y)/ y))

    def set_loss_function(self):
        return NotImplementedError("label must define their data dir setting method")
    
    def run_epoch(self, data, model, optimizer):
        """Train model for one pass of train data, and return loss, acccuracy"""

        # training label
        is_training = model.training
        percenterror = []

        loss_function = self.set_loss_function()

        # Iterate through batches

        for batch in tqdm(data):

            # Grab x and y
            x, y = batch['x'], batch['y']

            print(x.shape)
            # get prediction
            out = model(x)

            #print('out:', out, 'y:', y)

            percenterror.append( self.cal_error(out, y))
            # If training, do an update.
            if is_training:
                optimizer.zero_grad()
                joint_loss = loss_function(out, y)
                joint_loss.backward()
                optimizer.step()

        # Calculate epoch level scores
        avg_percenterror = np.mean(percenterror)
        return avg_percenterror

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
            err = self.run_epoch(train_data, model.train(), optimizer)
            trainerr[i] = err
            print('Train | avg percent error : {:.6f} '.format(err))

            # Run **validation**
            err = self.run_epoch(dev_data, model.eval(), optimizer)
            validerr[i] = err
            print('Valid | avg percent error : {:.6f} '.format(err))

            # Save model
            
        self.model = model
        return trainerr, validerr
    
    def get_pred(self, X):
        
        model = self.model
        model.eval()

        X = torch.tensor(X, dtype=torch.float32)
        pred = model(X).detach().numpy()

        return pred

class MAPEtorch(TorchTrainerBase):

    def set_loss_function(self):
        return MAPELoss()
    
class Energy(Label):

    def __init__(self, name='energy', cutoff=30):
        self.name = name
        self.cutoff = cutoff

    
    def set_data_dir(self, case_id): 

        cwd = os.getcwd() + '/batch{}/'.format(str(case_id))
        if not os.path.exists(cwd):
            os.mkdir(cwd)
        
        self.data_dir = cwd


    def visualize_pred(self, y_pred, y_test):
        
        row, col = 2, 5
        sample = row * col
        plot_dir = self.plot_dir
        fig, ax = plt.subplots( row,  col, figsize=( 4 * col, 5 * row))
        inds = np.random.choice( np.arange(y_pred.shape[0]), sample )

        pred = y_pred[inds]
        ref = y_test[inds]

        x = np.arange(1, y_pred.shape[-1] + 1)
        cnt = 0 

        for i in range(row):
            for j in range(col):

                ax[i][j].scatter(x, pred[cnt], label='pred' )
                ax[i][j].scatter(x, ref[cnt], label='true')
                ax[i][j].legend()

                cnt += 1

        fig.savefig(plot_dir)

class EnergyGSGap(Energy, CNNBase, GSGapBase, MAPEtorch):

    def init_model(self):
        self.model = Energy1DCNN(self.cutoff - 1)

class EnergyAllGap(Energy, CNNBase, AllGapBase, MAPEtorch):

    def init_model(self):
        self.model = Energy1DCNN(self.cutoff - 1)

class EnergyNearestNGSGap(Energy, NearestNeighborBase, GSGapBase, MAPEtorch):

    def init_model(self):
        self.model = EnergyForward(self.cutoff - 1)