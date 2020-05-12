import torch.optim as optim
import torch as torch
import torch.nn.functional as func
import numpy as np
import model as nn
import json

# Main training loop
def train(hyper):

    # Create our model
    net = nn.Net()

    # Extract learning rate from hyper parameters
    lr=hyper["learning rate"]

    # Create our optimizer
    optimizer = optim.Adam(net.parameters(), lr)

    # Extract num epochs from hyper parameters
    num_epochs = hyper["num_epochs"]

    # Training loop
    for epoch in range(1,num_epochs+1):

        # Clear our gradient buffer
        optimizer.zero_grad()

        # Clear our gradients


if  __name__ == "__main__":

    # TEMP: location of paramater file
    p_file = "/Users/aleccooper/Documents/Translate/Detection/hyper.json"

    # import hyper parameters
    with open(p_file) as paramfile:
        hyper = json.load(paramfile)

    train(hyper)


    
    