import torch
import torch.nn as nn
import torch.nn.functional as func

class Net(nn.Module):

    # Architecture of network
    def __init__(self,batch_size):

        super(Net, self).__init__()

        # Create the layers of our neural net

        # temporary vals
        embedded_dim = 29
        hidden_size = 29
        seq_length = 10            
        num_layers = 1

        # is seq_length right here?
        self.h0 = torch.randn((num_layers,batch_size,hidden_size),dtype=torch.float)
        self.c0 = torch.randn((num_layers,batch_size,hidden_size),dtype=torch.float)

        # The lstm layer of our net
        self.lstm1 = nn.LSTM(embedded_dim,hidden_size,num_layers=num_layers,batch_first=True)

        # Convolutional layer
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=6,kernel_size=3)
        self.pool1 = nn.MaxPool2d((2,2))
        self.conv2 = nn.Conv2d(in_channels=6,out_channels=18,kernel_size=3)
        self.pool2 = nn.MaxPool2d((2,2))

        # The linear layer, mapping to the 2 classifications
        self.hidden1 = nn.Linear(90,1000)
        self.hidden2 = nn.Linear(1000,100)
        self.hidden3 = nn.Linear(100,1)

    # Feedforward function
    def forward(self,word):

        # LSTM layers
        #x = self.lstm1(word, (self.h0,self.c0))
        x = self.lstm1(word)

        # Convolution layers
        x = torch.unsqueeze(x[0],1) #add singleton dimension for channel dim in convs
        x = func.relu(self.conv1(x))
        x = self.pool1(x)
        x = func.relu(self.conv2(x))
        x = self.pool2(x)

        # Linear layers
        # Convert to flat vector
        x = torch.reshape(x,(x.size()[0],18*5))
        x = func.relu(self.hidden1(x))
        x = func.relu(self.hidden2(x))
        x = torch.sigmoid(self.hidden3(x))

        return x

    # Test function. Avoids calculation of gradients.
    def test(self, data, loss_func, epoch):
        self.eval()
        with torch.no_grad():
            inputs= data.x_test
            targets= data.y_test
            output = self.forward(inputs).reshape(-1)
            cross_val= loss_func(output, targets)
        return cross_val.item(),output

    # Reset all training weights
    def reset(self):

        self.lstm1.reset_parameters()
        self.hidden1.reset_parameters()
        self.hidden2.reset_parameters()
        self.hidden3.reset_parameters()
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()



