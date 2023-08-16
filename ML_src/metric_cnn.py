import numpy as np
import os
from sklearn.model_selection import train_test_split
from class_cnn import *
from tqdm import tqdm
import time
import json

class Label():

    def __init__(self, metric_name, ID):
        self.metric_name = metric_name
        self.label_name = self.__class__.__name__
        self.id = ID
        self.init_model()
        


    def set_data_dir(self): 
        return NotImplementedError("label must define their data dir setting method")
    
    def set_model_dir(self, description=''):

        cwd = os.getcwd() + '/models/'
        if not os.path.exists(cwd):
            os.mkdir(cwd)
        
        self.model_name  = self.model.__class__.__name__
        self.full_id = '-'.join([self.model_name, self.label_name, str(description), self.id ])
        self.model_dir = cwd + self.full_id + '.pt'
        self.para_dir = cwd + self.full_id + '.json'

    def get_label_name(self):
        return self.label_name
    
    def get_model_name(self):
        return self.model_name
    
    def get_id(self):
        return self.id
        
    def load_data(self):
        
        y = self.process_y()
        x = self.process_x()
        
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=55)

        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def get_model_weights(self):

        print(self.model)
        print(self.model.cnn1.weight)
        print(self.model.hidden.weight)
        print(self.model.hidden2.weight)

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
    
    def export_model(self, parameter):
        model_dir = self.model_dir
        torch.save(self.model, model_dir)

        para_dir = self.para_dir
        with open(para_dir, 'w') as f:
            json.dump(parameter, f, indent=4)


class Ipr(Label):

    def __init__(self, metric_name='gpi', if_sort=0):
        self.metric_name = metric_name
        self.if_sort = if_sort

class Gpi(Label):

    def __init__(self, metric_name='gpi', if_sort=0):
        self.metric_name = metric_name
        self.if_sort = if_sort

    def set_data_dir(self, data_description='0'): 

        cwd = os.getcwd() + '/batch-{}/'.format(str(data_description))
        if not os.path.exists(cwd):
            os.mkdir(cwd)
        
        self.data_dir = cwd

        
    def init_model(self):
        self.model = GPICNN()


class CNNDis(Label):

    def process_x(self):

        dir = self.data_dir

        disx = np.loadtxt(dir + 'disx')
        disy = np.loadtxt(dir + 'disy')
        
        dis = np.zeros( (disx.shape[0], 2, disx.shape[1]))
        dis[:, 0, :] = disx
        dis[:, 1, :] = disy

        return dis

class CNNxOnlyDis(Label):

    def process_x(self):

        dir = self.data_dir

        disx = np.loadtxt(dir + 'disx')
        
        dis = np.zeros( (disx.shape[0], 1, disx.shape[1]))
        dis[:, 0, :] = disx

        return dis
    
class NearestNeighborDis(Label):

    def process_x(self):

        dir = self.data_dir

        disx = np.loadtxt(dir + 'disx')
        disy = np.loadtxt(dir + 'disy')
        
        dis = np.exp( np.sqrt(( disx[:, 1:] + 1 - disx[:, :-1]) **2 + (disy[:, 1:] - disy[:, :-1]) ** 2 ) - 1)

        return dis

class GSGap(Label):

    def process_y(self):

        dir = self.data_dir
        key = self.metric_name
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

class GSAvgGap(Label):

    def process_y(self):

        dir = self.data_dir
        key = self.metric_name
        cutoff = self.cutoff

        out = dir + key + 'cutoff{}gsavggap'.format(cutoff)

        if os.path.exists(out):
            arr = np.loadtxt( out)

        else:
            arr = np.loadtxt(dir + key)
            arr = arr[:, :cutoff]
            arr = arr - np.reshape( np.repeat(arr[:, 0], arr.shape[-1]), arr.shape)

            raw_ref = np.loadtxt('ref_energy')[:cutoff]
            ref = np.reshape(np.tile( raw_ref - raw_ref[0] , (1, arr.shape[0])), arr.shape)

            arr = arr[:, 1:] - ref[:, 1:]

            np.savetxt(out, arr)

        return arr
    

    
class AllGap(Label):

    def process_y(self):

        dir = self.data_dir
        key = self.metric_name
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
    
class MSEtorch(TorchTrainerBase):

    def set_loss_function(self):
        return MSELoss()
    
class MSEWtorch(TorchTrainerBase):

    def mod_weights(self):
        raise NotImplementedError("specify weight modding method")
    
    def set_weights(self, weights):
        self.weights = weights
    
    def set_loss_function(self):
        return MSEWLoss(self.mod_weights())

class GSWtorch(MSEWtorch):
    """Ground state weighted MSE"""

    def mod_weights(self):

        if not isinstance( self.weights, float):
            raise ValueError("GSW only allows one float input")
        
        dim = self.output
        gs = 0
        
        weights = torch.ones(dim)
        weights[gs] = self.weights

        print(weights)
        return weights
    

class Energy(Label):

    def __init__(self, ID, cutoff=30, activation_func="ReLU"):
        self.cutoff = cutoff
        self.output = cutoff - 1
        self.activation_func = activation_func
        super(Energy, self).__init__('energy', ID)
    
    def get_activation(self):
        return self.activation_func
    
    def set_data_dir(self, data_description='0'): 

        cwd = os.getcwd() + '/batch-{}/'.format(str(data_description))
        if not os.path.exists(cwd):
            os.mkdir(cwd)
        
        self.data_dir = cwd


    # def visualize_pred(self, y_pred, y_test):
        
    #     row, col = 2, 5
    #     sample = row * col
    #     plot_dir = self.plot_dir
    #     fig, ax = plt.subplots( row,  col, figsize=( 4 * col, 5 * row))
    #     inds = np.random.choice( np.arange(y_pred.shape[0]), sample )

    #     pred = y_pred[inds]
    #     ref = y_test[inds]

    #     x = np.arange(1, y_pred.shape[-1] + 1)
    #     cnt = 0 

    #     diff = np.round(np.sum( ( pred - ref) ** 2, axis = 1), decimals=2)

    #     for i in range(row):
    #         for j in range(col):

    #             ax[i][j].scatter(x, pred[cnt], label='pred' )
    #             ax[i][j].scatter(x, ref[cnt], label='true')
    #             ax[i][j].set_title( 'diff = {}'.format(diff[cnt]))
    #             ax[i][j].legend()

    #             cnt += 1

    #     fig.suptitle('Model = {}'.format(self.full_model_name))
    #     fig.savefig(plot_dir)
        

class GSGap_MAPE(Energy, CNNDis, GSGap, MAPEtorch):

    def init_model(self):
        self.model = Energy1DCNN(outputdim = self.output)

class AllGap_MAPE(Energy, CNNDis, AllGap, MAPEtorch):

    def init_model(self):
        self.model = Energy1DCNN(outputdim = self.output)

class AllGap_MSE(Energy, CNNDis, AllGap, MSEtorch):

    def init_model(self):
        self.model = Energy1DCNNComplex(self.output)

class NearestN_GSGap_MAPE(Energy, NearestNeighborDis, GSGap, MAPEtorch):

    def init_model(self):
        self.model = EnergyForward(self.output)

class GSGap_MSE(Energy, CNNDis, GSGap, MSEtorch):

    def init_model(self):
        self.model = Energy1DCNN(outputdim = self.output, activation_func=self.activation_func)

class GSGap_GSW_MSE(Energy, CNNDis, GSGap, GSWtorch):
    
    def init_model(self):
        self.model = Energy1DCNN(outputdim = self.output)
    
    def set_model_dir(self, description=''):
        description = 'GSweight' + str(self.weights) + str(description)
        return super(GSGap_GSW_MSE, self).set_model_dir(description)

class GSGap_MSE_Simple(Energy, CNNDis, GSGap, MSEtorch):

    def init_model(self):
        self.model = Energy1DCNNSimple(outputdim = self.output, activation_func=self.activation_func)

class GSAvgGap_MSE(Energy, CNNDis, GSAvgGap, MSEtorch):

    def init_model(self):
        self.model = Energy1DCNN(outputdim = self.output, activation_func=self.activation_func)

class GSAvgGap_MSE_Simple(Energy, CNNDis, GSAvgGap, MSEtorch):

    def init_model(self):
        self.model = Energy1DCNNSimple(outputdim = self.output, activation_func=self.activation_func)

class GSGapXonly_MSE_Simple(Energy, CNNxOnlyDis, GSGap, MSEtorch):

    def init_model(self):
        self.model = Energy1DCNNSimple(inputdim = 1, outputdim = self.output, activation_func=self.activation_func)