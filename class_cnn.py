import torch
import torch.nn as nn


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

