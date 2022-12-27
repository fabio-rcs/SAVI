#!/usr/bin/env python3
# --------------------------------------------------
# Fábio Sousa.
# SAVI, December 2022.
# --------------------------------------------------

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from model import Model
from dataset import Dataset
from statistics import mean
from colorama import Fore, Style

def main():

    # -----------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------

    #Create dataset
    dataset = Dataset(3000, 0.9, 14, sigma=3)
    
    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=256, shuffle=True)
    #It's good practice to guarantee that the batches have an appropriate size
 
    # #Draw training data
    # plt.plot(dataset.xs_np, dataset.ys_np_labels,'g.', label = 'labels')
    # plt.legend(loc='best')
    # plt.show()

    #Define hyper parameters
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu' # cuda: 0 index of gpu

    model = Model() #Instantiate model
    model.to(device) #Move the model variable to the gpu if one exists

    learning_rate = 0.001 #It's the step for each iteration. Big rates is faster, but has convergence limitations. Low rates are slower but allow a better convergence
    maximum_num_epochs = 500
    termination_loss_threshold =  9.25

    criterion = torch.nn.MSELoss()
    
    #Here we used a different optimizer because it gives better results
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
  
    # -----------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------
    
    #An epoch is an iteration of what comes next (of each prediction)
    idx_epoch = 0

    while True:

        losses = []

        #Train batch by batch
        #TQDM shows progress bar
        for batch_idx, (xs_ten, ys_ten_labels) in tqdm(enumerate(loader), total=len(loader), desc=Fore.GREEN + 'Training batches for Epoch ' + str(idx_epoch) +  Style.RESET_ALL):
            #.to(device) keeps all variables in the same GPU or CPU
            xs_ten = xs_ten.to(device)
            ys_ten_labels = ys_ten_labels.to(device)

            #Apply the network to get the predicted ys
            ys_ten_predicted = model.forward(xs_ten)

            #Compute the error based on the predictions
            loss = criterion(ys_ten_predicted, ys_ten_labels)

            #Update the model, i.e. the neural network's weights 
            optimizer.zero_grad() #Resets the weights to make sure we are not accumulating
            loss.backward() #Propagates the loss error into each neuron
            optimizer.step() #Update the weights
         
            losses.append(loss.data.item())

            # #Report
            # print('Epoch ' + str(idx_epoch) + ', Loss ' + str(loss.item()))

        #Compute the loss for the epoch
        epoch_loss = mean(losses)

        print(Fore.BLUE + 'Epoch ' + str(idx_epoch) + ' Loss ' + str(epoch_loss) + Style.RESET_ALL)
        
        idx_epoch += 1 #Go to next epoch
        
        #Termination criteria
        if idx_epoch > maximum_num_epochs:
            print('Finished training. Reached maximum number of epochs.')
            break
        elif epoch_loss < termination_loss_threshold:
            print('Finished training. Reached target loss.')
            break

    # -----------------------------------------------------------------
    # Finalization
    # -----------------------------------------------------------------

    #Run the model once to get ys_predicted
    ys_ten_predicted = model.forward(dataset.xs_ten.to(device))
    ys_np_predicted = ys_ten_predicted.cpu().detach().numpy()

    #Draw data in a plot
    plt.plot(dataset.xs_np, dataset.ys_np_labels,'g.', label = 'Original')
    plt.plot(dataset.xs_np, ys_np_predicted,'rx', label = 'Predicted')
    plt.legend(loc='best')
    plt.grid('on')
    plt.title('Deep Learning for line fitting')
    plt.show()


    

if __name__ == "__main__":
    main()

