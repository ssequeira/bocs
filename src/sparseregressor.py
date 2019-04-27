import torch
from torch.distributions.constraints import positive

import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

import itertools

import tqdm


class SparseRegressor(object):

    def __init__(self, in_dim, interaction=2, name=None):
        self.in_dim = in_dim
        self.effective_dim = 0
        for i in range(0, interaction+1):
            self.effective_dim += binomial(self.in_dim, i)
        if name is not None:
            self.name = name + '_'
        else:
            self.name = ''
        
    def _model(self, x_data, y_data):
        model_var = pyro.sample(self.name + 'sigma', dist.Gamma(self.prior_conc, 1))
        assert (x_data.shape[-1] == self.in_dim), 'Wrong input dimension: ' \
            'Expected ' + str(self.in_dim) + ', obtained ' + \
            str(x_data.shape[-1]) + '.'
        tau = pyro.sample(self.name + 'hs_tau', dist.HalfCauchy(self.scale))
        with pyro.plate('hs_regressors', self.effective_dim):
            lam = pyro.sample(self.name + 'hs_lambda', dist.HalfCauchy(1.))
            beta = pyro.sample(self.name + 'hs_beta', dist.Normal(0., tau * lam))
        y_pred = beta @ x_data
        with pyro.plate('data_loop', len(y_data)):
            pyro.sample(self.name + 'obs', dist.Normal(y_pred, model_var), obs=y_data)

    def _guide(self, x_data, y_data):
        q_prior_conc1 = pyro.param(self.name +'q_prior_conc1', torch.tensor(1.),
                                   constraint=positive)
        q_prior_conc2 = pyro.param(self.name +'q_prior_conc2', torch.tensor(1.),
                                   constraint=positive)
        model_var = pyro.sample(self.name + 'sigma', dist.Gamma(q_prior_conc1,
                                                    q_prior_conc2))
        q_tau = pyro.param(self.name +'q_tau', torch.tensor(self.scale),
                           constraint=positive)
        tau = pyro.sample(self.name + 'hs_tau', dist.HalfCauchy(q_tau))
        with pyro.plate('hs_regressors', self.effective_dim):
            q_lam_conc = pyro.param(self.name +'q_lam_conc', torch.tensor(1.),
                                    constraint=positive)
            lam = pyro.sample(self.name + 'hs_lambda', dist.HalfCauchy(q_lam_conc))
            q_beta_loc = pyro.sample(self.name + 'q_beta_loc', torch.tensor(0.))
            q_beta_var = pyro.sample(self.name + 'q_beta_var', torch.tensor(1.),
                                     constraint=positive)
            beta = pyro.sample(self.name + 'hs_beta', dist.Normal(q_beta_loc, q_beta_var))

    def fit(self, x_data, y_data, prior_conc, scale, num_iter=4000):
        self.prior_conc = prior_conc
        self.scale = scale
        pyro.clear_param_store()
        svi = SVI(self._model, self._guide, Adam(), Trace_ELBO())
        iters = tqdm(range(num_iter))
        for iter in iters:
            elbo = svi.step(x_data, y_data)
            iters.set_description('ELBO: ' + str(elbo.item()))
        
        

        
