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
import lang_model as model
import data_parse as data_parser
import json
import os
import argparse
import matplotlib.pyplot as plt

# Main training loop
def train(hyper,args):

    # Load in our dataset
    data = data_parser.Data(args.datafile,10,args.tokienize,hyper["num test"])

    # Create our model
    net = model.Net(data.x_train.size()[0])

    # Extract learning rate from hyper parameters
    lr=hyper["learning rate"]

    # Create our optimizer
    optimizer = optim.Adam(net.parameters(), lr)

    # Create our loss function
    loss_func = torch.nn.BCELoss(reduction="mean")

    # Extract num epochs from hyper parameters
    num_epochs = hyper["num_epochs"]

    # list of our training loss values
    loss_vals = []
    
    # lists to track preformance of network
    obj_vals= []
    cross_vals= []
    correct_vals = []

    # Training loop
    for epoch in range(1,num_epochs+1):

        # Clear our gradient buffer
        optimizer.zero_grad()

        # Clear our gradients
        net.zero_grad

        # Feed the output throught the net
        output = net.forward(data.x_train)

        # calculate loss function
        loss = loss_func(output[:,0],data.y_train)
        loss_vals.append(loss)

        # Backpropagate the loss
        loss.backward()

        # Calculate test data
        test_data = net.test(data, loss_func, epoch)
        test_val = test_data[0]
        test_output = test_data[1]

        # Calc percent correct
        i = 0
        num_correct = 0
        for choice in test_output:
            if choice < 0.5 and data.y_test[i] < 0.5:
                num_correct += 1
            elif choice > 0.5 and data.y_test[i] > 0.5:
                num_correct += 1
            i+=1
        pecent_correct = num_correct/len(data.y_test) * 100.0

        # Graph our progress
        obj_vals.append(loss)
        cross_vals.append(test_val)
        correct_vals.append(pecent_correct)

        optimizer.step()

        # High verbosity report in output stream
        if args.v >=2:
            print('Epoch [{}/{}]'.format(epoch, num_epochs) +\
                '\tTraining Loss: {:.4f}'.format(loss) +\
                '\tTest Loss: {:.4f}'.format(test_val) +\
                "\tPercent Correct: {:.2f}".format(pecent_correct) +\
                    "%")
    
    # Low verbosity final report
    if args.v:
        print('Final training loss: {:.4f}'.format(obj_vals[-1]))
        print('Final test loss: {:.4f}'.format(cross_vals[-1]))

    # Plot Results
    plt.plot(range(num_epochs), obj_vals, label= "Training loss", color="blue")
    plt.plot(range(num_epochs), cross_vals, label= "Test loss", color= "green")
    plt.ylabel("Binary Cross Entropy Loss")
    plt.xlabel("Training Epoch")
    plt.legend()
    plt.savefig("Training-Test-Loss")
    plt.show()

if  __name__ == "__main__":

    # Get Command Line Arguments
    parser = argparse.ArgumentParser(description="English-Inuktitut Text Classifier in PyTorch")
    parser.add_argument("datafile",metavar="data_file_name.txt",type=str)
    parser.add_argument("params",metavar="param_file_name.json",type=str)
    parser.add_argument('-v', type=int, default=1, metavar='N',
                        help='Verbosity (Default: 1)')
    parser.add_argument("-t", action="store_true", dest="tokienize",help="Retokienizes data")
    args = parser.parse_args()

    # import hyper parameters
    with open(args.params) as paramfile:
        hyper = json.load(paramfile)

    # We need to check if the data parser has run on the corpus yet
    dir_path = os.path.dirname(os.path.realpath(__file__))
    if not os.path.isfile(dir_path + "/Data/x_test.npy"):
        data = data_parser.Data(args.datafile,10,False,hyper["num test"])

    train(hyper,args)


    
    