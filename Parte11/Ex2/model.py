#!/usr/bin/env python3
# --------------------------------------------------
# Fábio Sousa.
# SAVI, December 2022.
# --------------------------------------------------

import torch

#Definition of the model. 
class Model(torch.nn.Module):

    def __init__(self):
        super().__init__() #Call superclass constructor        

        #Define the structure of the neural network
        self.layer1 = torch.nn.Linear(1,1) #In 1 and out 1

    def forward(self, xs):
        
        ys = self.layer1(xs) #Here we apply the network. We give the x and get the y in return
        
        return ys