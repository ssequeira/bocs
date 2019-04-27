import torch
import torch.nn as nn


import pyro
import pyro.distributions as dist
import pyro.poutine as poutine


class Prior(object):
    '''Abstract class for Bayesian priors on linear regression.
    Implement further priors by overriding _sample'''

    def __init__(self, *args, **kwargs):
        super(self, Prior).__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self._sample(*args, **kwargs)

    def _sample(self, *args, **kwargs):
        pass

class DiagonalGaussian(Prior):
    '''Independent centered normal with constant scale.'''

    def __init__(self, scale, *args, **kwargs):
        super(self, DiagonalGaussian).__init__(*args, **kwargs)
        self.scale = scale

    def _sample(self, *args, **kwargs):
        return dist.Normal(0., self.scale).sample()
