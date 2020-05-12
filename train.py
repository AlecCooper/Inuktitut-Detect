#------------------------------------------------
#
#  Alexander Cooper
#  alec.n.cooper@gmail.com
#
#------------------------------------------------
import torch.optim as optim
import torch as torch
import torch.nn.functional as func
import numpy as np
import model as nn
import data_parse as parser
import json
import os

# Main training loop
def train(hyper):

    # Create our model
    net = nn.Net()

    # Load in our dataset
    data = parser.Data(hyper["file location"],10,False,5)

    # Extract learning rate from hyper parameters
    lr=hyper["learning rate"]

    # Create our optimizer
    optimizer = optim.Adam(net.parameters(), lr)

    # Extract num epochs from hyper parameters
    num_epochs = hyper["num_epochs"]

    # list of our training loss values
    loss_vals = []

    # Training loop
    for epoch in range(1,num_epochs+1):

        # Clear our gradient buffer
        optimizer.zero_grad()

        # Clear our gradients
        net.zero_grad

        



if  __name__ == "__main__":

    # TEMP: location of paramater file
    p_file = "/Users/aleccooper/Documents/Translate/Detection/hyper.json"

    # import hyper parameters
    with open(p_file) as paramfile:
        hyper = json.load(paramfile)

    # We need to check if the data parser has run on the corpus yet
    dir_path = os.path.dirname(os.path.realpath(__file__))
    if not os.path.isfile(dir_path + "/test_data.npy"):
        data = parser.Data(hyper["file location"],10,False,5)

    train(hyper)


    
    