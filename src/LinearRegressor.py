import torch
import torch.nn as nn


class LinearRegressor(nn.Module):

    '''Linear regressor module'''

    def __init__(self, input_dim, output_dim=1, interaction=1, bias=True):
        super(self, LinearRegressor).__init__()
        self._effective_dim = input_dim + 1 if bias else input_dim
        if interaction > 1:
            self._reformat_req = True
            self.interaction = interaction
            for i in range(2, interaction+1):
                self._effective_dim += self._binomial(self.input_dim, i)
        self.linear = nn.Linear(self._effective_dim, self.output_dim, bias)

    def _reformat(self, x):
        # TODO: finish this.
        return x
    
    def __call__(self, x):
        if self._reformat_req:
            return self.linear(self._reformat(x))
        else:
            return self.linear(x)



    
        
