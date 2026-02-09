import pymc as pm
import numpy as np
import arviz as az
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Optional
import warnings
import pandas as pd 
warnings.filterwarnings('ignore')

class BayesianChangePointModel:
    """Bayesian Change Point Detection for Oil Prices"""
    
    def __init__(self, data: np.ndarray, model_type: str = 'mean_shift'):
        """
        Initialize the change point model
        
        Args:
            data: Time series data (preferably stationary)
            model_type: Type of model ('mean_shift' or 'variance_shift')
        """
        self.data = data
        self.model_type = model_type
        self.n_obs = len(data)
        self.model = None
        self.trace = None
        self.summary = None
        
    def build_mean_shift_model(self) -> pm.Model:
        """
        Build a Bayesian change point model with mean shift
        
        Returns:
            PyMC model
        """
        with pm.Model() as model:
            # Discrete uniform prior for change point
            tau = pm.DiscreteUniform(
                "tau", 
                lower=0, 
                upper=self.n_obs - 1,
                initval=self.n_obs // 2
            )
            
            # Priors for means before and after change point
            mu1 = pm.Normal("mu1", mu=np.mean(self.data), sigma=np.std(self.data))
            mu2 = pm.Normal("mu2", mu=np.mean(self.data), sigma=np.std(self.data))
            
            # Priors for standard deviations (optional: could also change)
            sigma1 = pm.HalfNormal("sigma1", sigma=np.std(self.data))
            sigma2 = pm.HalfNormal("sigma2", sigma=np.std(self.data))
            
            # Switch between parameters based on tau
            mean = pm.math.switch(tau > np.arange(self.n_obs), mu1, mu2)
            sigma = pm.math.switch(tau > np.arange(self.n_obs), sigma1, sigma2)
            
            # Likelihood
            likelihood = pm.Normal(
                "likelihood", 
                mu=mean, 
                sigma=sigma, 
                observed=self.data
            )
            
        return model
    
    def build_variance_shift_model(self) -> pm.Model:
        """
        Build a model where only variance changes
        
        Returns:
            PyMC model
        """
        with pm.Model() as model:
            # Change point
            tau = pm.DiscreteUniform(
                "tau", 
                lower=0, 
                upper=self.n_obs - 1,
                initval=self.n_obs // 2
            )
            
            # Common mean
            mu = pm.Normal("mu", mu=np.mean(self.data), sigma=np.std(self.data))
            
            # Different variances before/after
            sigma1 = pm.HalfNormal("sigma1", sigma=np.std(self.data))
            sigma2 = pm.HalfNormal("sigma2", sigma=np.std(self.data))
            
            # Switch variance
            sigma = pm.math.switch(tau > np.arange(self.n_obs), sigma1, sigma2)
            
            # Likelihood
            likelihood = pm.Normal(
                "likelihood", 
                mu=mu, 
                sigma=sigma, 
                observed=self.data
            )
            
        return model
    
    def fit(self, 
            draws: int = 2000,
            tune: int = 1000,
            chains: int = 2,
            target_accept: float = 0.8) -> None:
        """
        Fit the Bayesian model
        
        Args:
            draws: Number of posterior samples
            tune: Number of tuning samples
            chains: Number of MCMC chains
            target_accept: Target acceptance rate
        """
        # Build model based on type
        if self.model_type == 'mean_shift':
            self.model = self.build_mean_shift_model()
        elif self.model_type == 'variance_shift':
            self.model = self.build_variance_shift_model()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Sample from posterior
        with self.model:
            self.trace = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                target_accept=target_accept,
                return_inferencedata=True,
                random_seed=42
            )
        
        # Get summary statistics
        self.summary = az.summary(self.trace, round_to=4)
        
    def diagnose_convergence(self) -> Dict:
        """
        Check MCMC convergence
        
        Returns:
            Dictionary with convergence diagnostics
        """
        if self.trace is None:
            raise ValueError("Model must be fitted first")
        
        diagnostics = {}
        
        # Check R-hat statistics
        rhats = az.rhat(self.trace)
        diagnostics['rhat_convergence'] = all(rhats < 1.1)
        
        # Check effective sample size
        ess = az.ess(self.trace)
        diagnostics['ess_adequate'] = all(ess > 400)
        
        # Trace plots
        az.plot_trace(self.trace)
        plt.tight_layout()
        
        return diagnostics
    
    def get_change_point_posterior(self) -> np.ndarray:
        """
        Extract posterior distribution of change point
        
        Returns:
            Array of tau posterior samples
        """
        if self.trace is None:
            raise ValueError("Model must be fitted first")
        
        return self.trace.posterior.tau.values.flatten()
    
    def plot_posterior_distributions(self, dates: Optional[pd.DatetimeIndex] = None) -> None:
        """
        Plot posterior distributions of parameters
        
        Args:
            dates: Datetime index for converting tau to dates
        """
        if self.trace is None:
            raise ValueError("Model must be fitted first")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot tau posterior
        tau_samples = self.get_change_point_posterior()
        axes[0, 0].hist(tau_samples, bins=50, edgecolor='black', alpha=0.7)
        axes[0, 0].set_title('Posterior Distribution of Change Point (τ)')
        axes[0, 0].set_xlabel('Time Index')
        axes[0, 0].set_ylabel('Frequency')
        
        # Convert to dates if provided
        if dates is not None:
            tau_dates = dates[np.array(tau_samples, dtype=int)]
            axes[0, 1].hist(tau_dates, bins=50, edgecolor='black', alpha=0.7)
            axes[0, 1].set_title('Posterior Distribution of Change Date')
            axes[0, 1].set_xlabel('Date')
            axes[0, 1].set_ylabel('Frequency')
            plt.setp(axes[0, 1].xaxis.get_majorticklabels(), rotation=45)
        
        # Plot means
        if 'mu1' in self.trace.posterior:
            mu1_samples = self.trace.posterior.mu1.values.flatten()
            mu2_samples = self.trace.posterior.mu2.values.flatten()
            
            axes[1, 0].hist(mu1_samples, bins=50, alpha=0.7, label='Before', edgecolor='black')
            axes[1, 0].hist(mu2_samples, bins=50, alpha=0.7, label='After', edgecolor='black')
            axes[1, 0].set_title('Posterior Distributions of Means')
            axes[1, 0].set_xlabel('Mean Value')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].legend()
        
        # Plot standard deviations
        if 'sigma1' in self.trace.posterior:
            sigma1_samples = self.trace.posterior.sigma1.values.flatten()
            sigma2_samples = self.trace.posterior.sigma2.values.flatten()
            
            axes[1, 1].hist(sigma1_samples, bins=50, alpha=0.7, label='Before', edgecolor='black')
            axes[1, 1].hist(sigma2_samples, bins=50, alpha=0.7, label='After', edgecolor='black')
            axes[1, 1].set_title('Posterior Distributions of Standard Deviations')
            axes[1, 1].set_xlabel('Standard Deviation')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].legend()
        
        plt.tight_layout()
        plt.show()
    
    def quantify_impact(self) -> Dict:
        """
        Quantify the impact of the change point
        
        Returns:
            Dictionary with impact metrics
        """
        if self.trace is None:
            raise ValueError("Model must be fitted first")
        
        impact = {}
        
        if self.model_type == 'mean_shift':
            # Calculate mean difference
            mu1_mean = self.trace.posterior.mu1.mean().item()
            mu2_mean = self.trace.posterior.mu2.mean().item()
            mean_diff = mu2_mean - mu1_mean
            mean_pct_change = (mean_diff / abs(mu1_mean)) * 100
            
            impact['mean_before'] = mu1_mean
            impact['mean_after'] = mu2_mean
            impact['mean_difference'] = mean_diff
            impact['percent_change'] = mean_pct_change
        
        return impact