from itertools import combinations

import torch
import torch.nn as nn

from util import binomial

class LinearRegressor(nn.Module):
    '''Linear regression module with interaction effects'''

    def __init__(self, in_dim, interaction=2, out_dim=1, bias=True, *args, **kwargs):
        super(self, LinearRegressor).__init__(*args, **kwargs)
        self.out_dim = out_dim
        self._interaction = interaction
        self._effective_dim = in_dim + 1 if bias else in_dim
        self._req_reformat = False

        if interaction > 1:
            self._req_reformat = True
            for i in range(interaction+1):
                self.in_dim += binomial(in_dim, i):
        else:
            self._in_dim = in_dim
        self.linear = nn.Linear(self.in_dim, self.out_dim, bias=True)
    
    def reformat(self, z):
        x_shape = list(z.shape)
        x_shape[1] = self._effective_dim
        x = torch.ones(torch.Size(x_shape))
        x[:, z.shape[-1]] = z
        active_dim = z.shape[-1]
        indices = list(range(self._in_dim)
        for i in range(2, self._interaction+1):
            combs = combinations(indices)
            for comb in combs:
                for idx in comb:
                    x[:, active_dim] *= z[:, idx]
                active_dim += 1
        return x

    def forward(self, z):
        assert z.shape[1] == self._in_dim, "Tensor is of the wrong dimension" \
            + "Expected " + str(self._in_dim) + ", got " + str(z.shape[0])
        return self.linear(self.reformat(z))

    def __call__(self, x):
        if self._req_reformat:
            x = self._reformat(x)
        return self.linear(x)
