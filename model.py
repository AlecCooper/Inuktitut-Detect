import torch
import torch.nn as nn
import torch.nn.functional as func

class Net(nn.Module):

    # Architecture of network
    def __init__(self):

        super(Net, self).__init__()

        # Create the layers of our neural net

        # temporary vals
        embedded_dim = 29
        hidden_dim = 29
        seq_length = 10            
        batch_size = 10
        num_layers = 1

        # The lstm layer of our net
        self.lstm1 = nn.LSTM(embedded_dim,hidden_dim,num_layers=num_layers)
        # The linear layer, mapping to the 2 classifications
        self.hidden1 = nn.Linear(hidden_dim,2)

    # Feedforward function
    def forward(self,word):

        x = self.lstm1(word)
        x = self.hidden1(x)

        return x

    # Reset all training weights
    def reset(self):

        self.lstm1.reset_parameters()



