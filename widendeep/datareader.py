# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

from torch.utils.data import Dataset

class WideDeepLoader(Dataset):
    def __init__(self, data):
        """
        data : dict
        """
        self.input_types = list(data.keys())
#         self.input_types = input_types
        self.X_wide = np.array(data['wide'], dtype = np.float)
        if 'deep_dense' in self.input_types: 
            self.X_deep_dense = np.array(data['deep_dense'], dtype = np.float)
            
        self.Y = data['target'] 
#         if self.mode is 'train':
#             self.Y = data['target']
#         elif self.mode is 'test':
#             self.Y = None

    def __getitem__(self, idx):

        xw = self.X_wide[idx]
        X = (xw, )
        
        if 'deep_dense' in self.input_types:
            xdd = self.X_deep_dense[idx]
            X += (xdd,)
        
#         import pdb; pdb.set_trace()
        if type(self.Y) == np.ndarray:
            y = np.array([self.Y[idx]], dtype = np.float)
            return X + tuple(y)
        else:
            return X

#         if self.mode is 'train':
#             y  = self.Y[idx]
#             return X, y
#         elif self.mode is 'test':
#             return X

    def __len__(self):
        return len(self.X_wide)