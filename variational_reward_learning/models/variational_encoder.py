import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from variational_reward_learning.utils.math import kl_divergence_normal

class VariationalEncoder(torch.nn.Module):
    """
    VAE-inspired reward encoder that outputs mean and log variance of the reward distribution.
    The reward itself is the latent variable.
    """

    def __init__(self, n_dct_fns: int, n_actions: int, hidden_dims: list[int], dropout: float = 0.1):
        super().__init__()
        input_dim = n_dct_fns**2 + n_actions
        
        # Encoder layers - output 2 dimensions: [reward_mean, reward_log_var]
        encoder_dims = [input_dim] + hidden_dims + [2]  # 2 for mean and log_var of reward
        encoder_layers = []
        for i in range(len(encoder_dims) - 1):
            encoder_layers.append(torch.nn.Linear(encoder_dims[i], encoder_dims[i+1]))
            if i < len(encoder_dims) - 2:  # Don't add activation after the last layer
                encoder_layers.append(torch.nn.LeakyReLU())
                encoder_layers.append(torch.nn.Dropout(dropout))
        
        self.encoder = torch.nn.Sequential(*encoder_layers)

    def encode(self, state_action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode state-action features to reward mean and log variance.
        
        Args:
            state_action: State-action features of shape (batch_size, features)
            
        Returns:
            Tuple of (reward_mean, reward_log_var) each of shape (batch_size, 1)
        """
        encoded = self.encoder(state_action)
        reward_mean = encoded[:, 0:1]  # Shape: (batch_size, 1)
        reward_log_var = encoded[:, 1:2]  # Shape: (batch_size, 1)
        return reward_mean, reward_log_var

    def reparameterize(self, mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick to sample from the reward distribution.
        
        Args:
            mean: Reward mean of shape (batch_size, 1)
            log_var: Reward log variance of shape (batch_size, 1)
            
        Returns:
            Sampled reward values of shape (batch_size, 1)
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, state_action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the VAE reward encoder.
        
        Args:
            state_action: State-action features of shape (batch_size, features)
            
        Returns:
            Tuple of (reward, mean, log_var) where reward is sampled from the reward distribution
        """
        mean, log_var = self.encode(state_action)
        reward = self.reparameterize(mean, log_var)
        return reward, mean, log_var