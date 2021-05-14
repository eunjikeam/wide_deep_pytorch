# coding: utf-8

import math
import json
from typing import NamedTuple

import numpy as np
import torch
import torch.nn as nn

def dense_layer(inp, out, dropout):
    return nn.Sequential(
        nn.Linear(inp, out),
        nn.LeakyReLU(inplace=True),
        nn.Dropout(dropout)
        )

class Wide(nn.Module):
    def __init__(self, wide_dim, output_dim):
        r"""wide (linear) component
        Linear model implemented via an Embedding layer connected to the output
        neuron(s).
        Parameters
        -----------
        wide_dim: int
            size of the Embedding layer. `wide_dim` is the summation of all the
            individual values for all the features that go through the wide
            component. For example, if the wide component receives 2 features with
            5 individual values each, `wide_dim = 10`
        output_dim: int, default = 1
            size of the ouput tensor containing the predictions
        Attributes
        -----------
        wide_linear: :obj:`nn.Module`
            the linear layer that comprises the wide branch of the model
        Examples
        --------
        >>> import torch
        >>> from pytorch_widedeep.models import Wide
        >>> X = torch.empty(4, 4).random_(6)
        >>> wide = Wide(wide_dim=X.unique().size(0), output_dim=1)
        >>> out = wide(X)
        """
        super(Wide, self).__init__()
        # Embeddings: val + 1 because 0 is reserved for padding/unseen cateogories.
        self.wide_linear = nn.Embedding(wide_dim + 1, output_dim, padding_idx=0)
        # (Sum(Embedding) + bias) is equivalent to (OneHotVector + Linear)
        self.bias = nn.Parameter(torch.zeros(output_dim))
        self._reset_parameters()

    def _reset_parameters(self):
        r"""initialize Embedding and bias like nn.Linear. See `original
        implementation
        <https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html#Linear>`_.
        """
        nn.init.kaiming_uniform_(self.wide_linear.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.wide_linear.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, X):  # type: ignore
        r"""Forward pass. Simply connecting the Embedding layer with the ouput
        neuron(s)"""
        out = self.wide_linear(X.long()).sum(dim=1) + self.bias
        return out
    
class DeepDense(nn.Module):
    def __init__(self, embeddings_input, embeddings_encoding_dict, continuous_cols, deep_column_idx, hidden_layers, dropout, output_dim):
        """
        Model consisting in a series of Dense Layers that receive continous
        features concatenated with categorical features represented with
        embeddings
        Parameters:
        embeddings_input: List
            List of Tuples with the column name, number of unique values and
            embedding dimension. e.g. [(education, 11, 32), ...]
        embeddings_encoding_dict: Dict
            Dict containing the encoding mappings
        continuous_cols: List
            List with the name of the so called continuous cols
        deep_column_idx: Dict
            Dict containing the index of the embedding columns. Required to
            slice the tensors.
        hidden_layers: List
            List with the number of neurons per dense layer. e.g: [64,32]
        dropout: List
            List with the dropout between the dense layers. We do not apply dropout
            between Embeddings and first dense or last dense and output. Therefore
            this list must contain len(hidden_layers)-1 elements. e.g: [0.5]
        output_dim: int
            1 for logistic regression or regression, N-classes for multiclass
        """
        super().__init__()

        self.embeddings_input = embeddings_input
        self.embeddings_encoding_dict = embeddings_encoding_dict
        self.continuous_cols = continuous_cols
        self.deep_column_idx = deep_column_idx

        for col,val,dim in embeddings_input:
            setattr(self, 'emb_layer_'+col, nn.Embedding(val+1, dim))
        input_emb_dim = np.sum([emb[2] for emb in embeddings_input])+len(continuous_cols)
        hidden_layers = [input_emb_dim] + hidden_layers
        dropout = [0.0] + dropout
        self.dense = nn.Sequential()
        for i in range(1, len(hidden_layers)):
            self.dense.add_module(
                'dense_layer_{}'.format(i-1),
                dense_layer( hidden_layers[i-1], hidden_layers[i], dropout[i-1])
                )
        self.dense.add_module('last_linear', nn.Linear(hidden_layers[-1], output_dim))

    def forward(self, X):
        emb = [getattr(self, 'emb_layer_'+col)(X[:,self.deep_column_idx[col]].long())
               for col,_,_ in self.embeddings_input]
        if self.continuous_cols:
            cont_idx = [self.deep_column_idx[col] for col in self.continuous_cols]
            cont = [X[:, cont_idx].float()]
            inp = torch.cat(emb+cont, 1)
        else:
            inp = torch.cat(emb, 1)
        out = self.dense(inp)
        return out
    
class WideDeep(nn.Module):
    """ 
    Wide and Deep model (Heng-Tze Cheng et al., 2016)
    """
    def __init__(self, 
                 wide_dim,
                 embeddings_input,
                 embeddings_encoding_dict, 
                 continuous_cols, 
                 deep_column_idx, 
                 hidden_layers, 
                 dropout, 
                 output_dim):
        super().__init__()
        
        self.wide = Wide(wide_dim, output_dim)
        self.deep_dense = DeepDense(
            embeddings_input,
            embeddings_encoding_dict,
            continuous_cols,
            deep_column_idx,
            hidden_layers, 
            dropout, 
            output_dim
        )
    
    def forward(self, x_wide, x_deep_dense):
        wide = self.wide(x_wide)
        deep = self.deep_dense(x_deep_dense)
        
        wide_deep = wide + deep
        
        return wide_deep