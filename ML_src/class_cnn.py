import torch
import torch.nn as nn
from torch.nn import MSELoss



     

class Flatten(nn.Module):
    """A custom layer that views an input as 1D."""
    
    def forward(self, input):
        return input.view(input.size(0), -1)

class GPICNN(nn.Module):

    def __init__(self):
        super(GPICNN, self).__init__()
        # TODO initialize model layers here
        self.cnn1 = nn.Conv2d(2, 1024, (2, 2))
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d( (2, 2) )
        self.dropout = nn.Dropout(p=0.5)
        self.flatten = Flatten()
        self.hidden = nn.Linear(1024 , 1024)
        self.hidden2 = nn.Linear( 1024, 1024)
        self.out = nn.Linear(1024, 126)

    def forward(self, x):

        # TODO use model layers to predict the two digits
        x = self.cnn1(x) # 2x2
        x = self.relu(x)
        x = self.pool(x) # 1x1
        x = self.flatten(x)
        x = self.hidden(x)
        x = self.relu(x)
        x = self.hidden2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.out(x)
        x = self.relu(x)

        gpi = x
        # ipr = x[:, 10:]


        return gpi


class Energy2DCNN(nn.Module):
    def __init__(self, outputdim):
        super(Energy2DCNN, self).__init__()
        # TODO initialize model layers here
        self.cnn1 = nn.Conv2d(2, 1024, (2, 2))
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d( (2, 2) )
        self.dropout = nn.Dropout(p=0.5)
        self.flatten = Flatten()
        self.hidden = nn.Linear(1024 , 1024)
        self.hidden2 = nn.Linear( 1024, 1024)
        self.out = nn.Linear(1024, outputdim)

    def forward(self, x):

        # TODO use model layers to predict the two digits
        x = self.cnn1(x) # 2x2
        x = self.relu(x)
        x = self.pool(x) # 1x1
        x = self.flatten(x)
        x = self.hidden(x)
        x = self.relu(x)
        x = self.hidden2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.out(x)
        x = self.relu(x)

        energy = x
        # ipr = x[:, 10:]


        return energy
    

class Energy1DCNNComplex(nn.Module):
    def __init__(self, outputdim):
        super(Energy1DCNNComplex, self).__init__()
        # TODO initialize model layers here
        self.cnn1 = nn.Conv1d(2, 512, kernel_size = 3, padding=1)
        self.cnn2 = nn.Conv1d(512, 1024, kernel_size = 3, padding = 1)
        self.cnn3 = nn.Conv1d(1024, 2048, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.flatten = Flatten()
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(p=0.5)
        self.hidden = nn.Linear(2048 , 2048)
        self.hidden2 = nn.Linear( 2048, 2048)
        self.out = nn.Linear(2048, outputdim)

    def forward(self, x):

        x = self.cnn1(x) # size 12
        x = self.relu(x)
        x = self.pool(x) # size 6
        x = self.cnn2(x) # size 6
        x = self.pool(x) # size 3
        x = self.cnn3(x) # size 3
        x = self.pool(x) # size 1
        x = self.flatten(x)
        x = self.hidden(x)
        x = self.relu(x)
        x = self.hidden2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.out(x)
        x = self.relu(x)

        energy = x
        # ipr = x[:, 10:]


        return energy
    

class Energy1DCNN(nn.Module):

    def set_activation(self, activation_func):
        if activation_func == 'ReLU':
            return nn.ReLU()
        
        elif activation_func == 'tanh':
            return nn.Tanh()
        
        else:
            raise ValueError('Not Recognized act funct')
        
    def __init__(self, outputdim = 1, activation_func='ReLU'):
        super(Energy1DCNN, self).__init__()
        # TODO initialize model layers here
        self.cnn1 = nn.Conv1d(2, 512, kernel_size = 3, padding=1)
        self.cnn2 = nn.Conv1d(512, 512, kernel_size = 3, padding = 1)
        self.activation = self.set_activation(activation_func)
        self.flatten = Flatten()
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(p=0.5)
        self.hidden = nn.Linear(512 * 3 , 1024)
        self.hidden2 = nn.Linear( 1024, 1024)
        self.out = nn.Linear(1024, outputdim)

    def forward(self, x):

        x = self.cnn1(x) # size 12
        x = self.activation(x)
        x = self.pool(x) # size 6
        x = self.cnn2(x) # size 6
        x = self.pool(x) # size 3
        x = self.flatten(x)
        x = self.hidden(x)
        x = self.activation(x)
        x = self.hidden2(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.out(x)
        x = self.activation(x)

        energy = x
        # ipr = x[:, 10:]


        return energy


class EnergyForward(nn.Module):
    def __init__(self, outputdim):
        super(EnergyForward, self).__init__()
        # TODO initialize model layers here
        self.relu = nn.ReLU()
        self.flatten = Flatten()
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(p=0.5)
        self.first  = nn.Linear(11, 1024)
        self.hidden = nn.Linear(1024, 1024)
        self.hidden2 = nn.Linear( 1024, 1024)
        self.out = nn.Linear(1024, outputdim)

    def forward(self, x):

        x = self.first(x)
        x = self.relu(x)
        x = self.hidden(x)
        x = self.relu(x)
        x = self.hidden2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.out(x)
        x = self.relu(x)

        energy = x
        # ipr = x[:, 10:]


        return energy

class MSEdoubleCNN(nn.Module):

    def __init__(self):
        super(MSEdoubleCNN, self).__init__()
        # TODO initialize model layers here
        self.cnn1 = nn.Conv2d(2, 256, (3, 3))
        self.relu = nn.ReLU()
        self.cnn2 = nn.Conv2d(256, 512, (3, 3))
        self.pool = nn.MaxPool2d( (2, 2) )
        self.dropout = nn.Dropout(p=0.5)
        self.flatten = Flatten()
        self.hidden = nn.Linear(512 , 512)
        self.hidden2 = nn.Linear( 512, 512)
        self.out = nn.Linear(512, 1)

    def forward(self, x):

        # TODO use model layers to predict the two digits
        x = self.cnn1(x) # 8x8
        x = self.relu(x)
        x = self.pool(x) # 4x4
        x = self.cnn2(x) # 2x2
        x = self.relu(x)
        x = self.pool(x) # 1x1
        x = self.flatten(x)
        x = self.hidden(x)
        x = self.relu(x)
        x = self.hidden2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.out(x)
        x = self.relu(x)

        ipr = x
        # ipr = x[:, 10:]


        return ipr

class MSELarge(nn.Module):

    def __init__(self):
        super(MSELarge, self).__init__()
        # TODO initialize model layers here
        self.cnn1 = nn.Conv2d(2, 512, (5, 5))
        self.relu = nn.ReLU()
        self.cnn2 = nn.Conv2d(512, 1024, (3, 3))
        self.pool = nn.MaxPool2d( (2, 2) )
        self.dropout = nn.Dropout(p=0.5)
        self.flatten = Flatten()
        self.hidden = nn.Linear(1024 , 1024)
        self.hidden2 = nn.Linear( 1024, 1024)
        self.out = nn.Linear(1024, 1)

    def forward(self, x):

        # TODO use model layers to predict the two digits
        x = self.cnn1(x) # 6x6
        x = self.relu(x)
        x = self.pool(x) # 3x3
        x = self.cnn2(x) # 1x1
        x = self.relu(x)
        x = self.flatten(x)
        x = self.hidden(x)
        x = self.relu(x)
        x = self.hidden2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.out(x)
        x = self.relu(x)

        ipr = x
        # ipr = x[:, 10:]


        return ipr


class MAPECNN(nn.Module):

    def __init__(self):
        super(MAPECNN, self).__init__()
        # TODO initialize model layers here
        self.cnn1 = nn.Conv2d(2, 256, (3, 3))
        self.relu = nn.ReLU()
        self.cnn2 = nn.Conv2d(256, 512, (3, 3))
        self.pool = nn.MaxPool2d( (2, 2) )
        self.dropout = nn.Dropout(p=0.5)
        self.flatten = Flatten()
        self.hidden = nn.Linear( 512, 512)
        self.hidden2 = nn.Linear( 512, 512)
        self.out = nn.Linear(512, 100)

    def forward(self, x):

        # TODO use model layers to predict the two digits
        x = self.cnn1(x) # 8x8
        x = self.relu(x)
        x = self.pool(x) # 4x4
        x = self.cnn2(x) # 2x2
        x = self.relu(x)
        x = self.pool(x) # 1x1
        x = self.flatten(x)
        x = self.hidden(x)
        x = self.relu(x)
        x = self.hidden2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.out(x)
        x = self.relu(x)

        ipr = x


        return ipr

class MAPELoss(nn.Module):
    def __init__(self):
        super(MAPELoss, self).__init__()
 
    def forward(self, inputs, targets):        
        MAPE = torch.mean(torch.abs ( ( inputs - targets)/targets))
        return MAPE

class MSEWLoss(nn.Module):

    def __init__(self, weights):
        self.weights = weights
        super(MSEWLoss, self).__init__()
 
    def forward(self, inputs, targets):   
    
        #MSEW = torch.mean(self.weights.tile( inputs.shape[0], 1) * (inputs - targets) ** 2)
        MSEW = torch.mean(self.weights * (inputs - targets) ** 2)
        return MSEW
    


