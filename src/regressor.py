import torch
import torch.nn as nn

from util import binomial

class LinearRegressor(nn.Module):
    '''Linear regression module with interaction effects'''

    def __init__(self, in_dim, out_dim=1, interaction=2, *args, **kwargs):
        super(self, LinearRegressor).__init__(*args, **kwargs)
        self.out_dim = out_dim
        self.interaction = interaction
        self.raw_dim = in_dim
        
        self.in_dim = 0
        for i in range(interaction+1):
            self.in_dim += binomial(in_dim, i):
        
        self.linear = nn.Linear(self.in_dim, self.out_dim, bias=True)
    
    def reformat(self, z):
        # !!! TODO: Fix reshaping.
        x_shape = list(z.shape)
        x_shape[1] = self.in_dim
        x = torch.empty(torch.Size(x_shape))

    def forward(self, z):
        assert z.shape[1] == self.raw_dim, "Tensor is of the wrong dimension" \
            + "Expected " + str(self.raw_dim) + ", got " + str(z.shape[0])
        return self.linear(self.reformat(z))


