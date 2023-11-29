import torch
from torch.distributions import Normal,Uniform,MultivariateNormal


class AppendedMVN:
    """
    Custom prior usable with the sbi package.

    Sample from a MVN with mean and covariance matrix determined by a GP.
    The mean of the GP is shifted by a sample from a normal distribution.
    The variance of the GP is scaled by a sample from a uniform distribution.
    """

    def __init__(self, GP_mean_mu,GP_mean_sd,GP_var_min,GP_var_max,mvn_mean,mvn_cov):
        self.GP_mean_mu = GP_mean_mu
        self.GP_mean_sd = GP_mean_sd
        self.GP_var_min = GP_var_min
        self.GP_var_max = GP_var_max
        self.mvn_mean = mvn_mean
        self.mvn_cov = mvn_cov
        self.gauss_mean = Normal(loc = self.GP_mean_mu,scale = self.GP_mean_sd)
        self.unif_var = Uniform(self.GP_var_min,self.GP_var_max)
        self.mvn = MultivariateNormal(loc = self.mvn_mean,covariance_matrix = self.mvn_cov)

    def sample(self, sample_shape=torch.Size([])):
        mean = self.gauss_mean.sample(sample_shape)
        var = self.unif_var.sample(sample_shape)
        mvns = self.mvn.sample(sample_shape)
        theta = torch.cat((mean,var,mvns),dim = -1)
        return theta

    def log_prob(self, values):
        means = values[...,0]
        vars = values[...,1]
        rels = values[...,2:]
        log_probs = self.gauss_mean.log_prob(means) + self.unif_var.log_prob(vars) + self.mvn.log_prob(rels)
        return log_probs
    

class AppendedMVN2:
    """
    Custom prior usable with the sbi package.

    Sample from a MVN with mean and covariance matrix determined by a GP.
    The mean of the GP is shifted by a sample from a uniform distribution.
    The variance of the GP is scaled by a sample from a uniform distribution.
    """

    def __init__(self, GP_mean_min,GP_mean_max,GP_var_min,GP_var_max,mvn_mean,mvn_cov):
        self.GP_mean_min = GP_mean_min
        self.GP_mean_max = GP_mean_max
        self.GP_var_min = GP_var_min
        self.GP_var_max = GP_var_max
        self.mvn_mean = mvn_mean
        self.mvn_cov = mvn_cov
        self.unif_mean = Uniform(self.GP_mean_min,self.GP_mean_max)
        self.unif_var = Uniform(self.GP_var_min,self.GP_var_max)
        self.mvn = MultivariateNormal(loc = self.mvn_mean,covariance_matrix = self.mvn_cov)

    def sample(self, sample_shape=torch.Size([])):
        mean = self.unif_mean.sample(sample_shape)
        var = self.unif_var.sample(sample_shape)
        mvns = self.mvn.sample(sample_shape)
        theta = torch.cat((mean,var,mvns),dim = -1)
        return theta

    def log_prob(self, values):
        means = values[...,0]
        vars = values[...,1]
        rels = values[...,2:]
        log_probs = self.unif_mean.log_prob(means) + self.unif_var.log_prob(vars) + self.mvn.log_prob(rels)
        return log_probs