import torch
import torch.nn as nn
import torch.nn.functional as func

class Net(nn.Module):

    # Architecture of network
    def __init__(self):

        super(Net, self).__init__()

        # Create the layers of our neural net
        self.lstm1 = nn.LSTM(10,20)
        self.fc1 = nn.Linear(20,20)
        self.fc2 = nn.Linear(20,10)
        self.fc3 = nn.Linear(10,2)

    # Feedforward function
    def forward(self,word):

        x = self.lstm1(word)
        x = func.relu(self.fc1(x))
        x = func.relu(self.fc2(x))
        x = func.sigmoid(self.fc3(x))

        return x

    # Reset all training weights
    def reset(self):

        self.lstm1.reset_parameters()
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.fc3.reset_parameters()




