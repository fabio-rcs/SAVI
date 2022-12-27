#!/usr/bin/env python3
# --------------------------------------------------
# Fábio Sousa.
# SAVI, December 2022.
# --------------------------------------------------

import torch
import numpy as np

class Dataset(torch.utils.data.Dataset):
    
    def __init__(self, num_points, f=1 ,a=2, sigma=1.4):

        self.num_points = num_points
        
        #Generate data
        self.xs_np = np.random.rand(num_points,1) * 20 - 10 #To get values between -10 and 10
        self.xs_np = self.xs_np.astype(np.float32)
        self.ys_np_labels = np.sin(f * self.xs_np) * a #Compute ys 
        self.ys_np_labels += np.random.normal(loc=0.0, scale=sigma, size=(num_points,1)) # add noise

        self.xs_ten = torch.from_numpy(self.xs_np)
        self.ys_ten = torch.from_numpy(self.ys_np_labels)

    def __getitem__(self, index): #Return a specific element x,y given the index, of the dataset
        return self.xs_ten[index], self.ys_ten[index]
    
    def __len__(self): #Return the length of the dataset
        return self.num_points