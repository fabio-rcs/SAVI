#!/usr/bin/env python3
# --------------------------------------------------
# Fábio Sousa.
# SAVI, December 2022.
# --------------------------------------------------

import pickle
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
import torch


#Definition of the model. 
#For now a 1 neuron network
class Model(torch.nn.Module):

    def __init__(self):
        super().__init__() #Call superclass constructor        

        #Define the structure of the neural network
        self.layer1 = torch.nn.Linear(1,1) #In 1 and out 1

    def forward(self, xs):
        
        ys = self.layer1(xs) #Here we apply the network. We give the x and get the y in return
        
        return ys


def main():

    # -----------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------

    #Load and read file with points
    file = open('pts.pkl', 'rb')
    pts = pickle.load(file)
    file.close()

    #Convert the pts into np arrays
    xs_np = np.array(pts['xs'], dtype=np.float32).reshape(-1,1) 
    #Pytorch only works with data in float32
    #Reshape -1 keeps that dimension
    #Shape was (a, ) and we transformed to (a,1) because Pytorch likes that

    ys_np_labels = np.array(pts['ys'], dtype=np.float32).reshape(-1,1)

    # #Draw training data
    # plt.plot(xs_np, ys_np_labels,'g.', label = 'Points')
    # plt.legend(loc='best')
    # plt.grid('on')
    # plt.title('Points to fit')
    # plt.show()

    #Define hyper parameters
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu' # cuda: 0 index of gpu


    model = Model() #Instantiate model
    model.to(device) #Move the model variable to the gpu if one exists

    learning_rate = 0.01
    maximum_num_epochs = 50 
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
  
    # -----------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------
    
    #An epoch is an iteration of what comes next (of each prediction)
    idx_epoch = 0

    while True:

        #Convert fom NumPy to Tensor, so that Pytorch can use the values
        #Keeps all variables in the same GPU or CPU
        xs_ten = torch.from_numpy(xs_np).to(device)
        ys_ten_labels = torch.from_numpy(ys_np_labels).to(device)

        #Apply the network to get the predicted ys
        ys_ten_predicted = model.forward(xs_ten)

        #Compute the error based on the predictions
        loss = criterion(ys_ten_predicted, ys_ten_labels)

        #Update the model, i.e. the neural network's weights 
        optimizer.zero_grad() #Resets the weights to make sure we are not accumulating
        loss.backward() #Propagates the loss error into each neuron
        optimizer.step() #Update the weights

        #Report
        print('Epoch ' + str(idx_epoch) + ', Loss ' + str(loss.item()))

        idx_epoch += 1 #Go to next epoch
        
        #Termination criteria
        if idx_epoch > maximum_num_epochs:
            print('Finished training. Reached maximum number of epochs.')
            break

    # -----------------------------------------------------------------
    # Finalization
    # -----------------------------------------------------------------

    #Run the model once to get ys_predicted
    ys_ten_predicted = model.forward(xs_ten)
    ys_np_predicted = ys_ten_predicted.cpu().detach().numpy()

    #Draw data in a plot
    plt.plot(xs_np, ys_np_labels,'g.', label = 'Original')
    plt.plot(xs_np, ys_np_predicted,'rx', label = 'Predicted')
    plt.legend(loc='best')
    plt.grid('on')
    plt.title('Deep Learning for line fitting')
    plt.show()


    

if __name__ == "__main__":
    main()

