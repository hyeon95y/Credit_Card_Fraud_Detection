import numpy as np
import pandas as pd
import os
import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class CCFDataset(Dataset) : 
    def __init__(self, df, transform=transforms.ToTensor()) :
        self.df = df.reset_index(drop=True)
        
    def __len__(self) :
        return self.df.shape[0]
    
    def __getitem__(self, idx) :
        # x, y
        x = torch.from_numpy(self.df.drop(['Class'], axis=1).iloc[idx].values).type('torch.FloatTensor')
        #y = torch.from_numpy(self.df['Class'].iloc[idx]).type('torch.FloatTensor')
        y = torch.tensor([self.df['Class'].iloc[idx]], dtype=torch.float32)
        return x, y 
    
