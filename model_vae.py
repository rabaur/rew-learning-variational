import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class VAERewardEncoder(torch.nn.Module):
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


class VAEPreferenceModel(torch.nn.Module):
    """
    VAE-inspired preference model that uses a probabilistic reward encoder.
    The reward itself is the latent variable, and the model optimizes the ELBO objective.
    """

    def __init__(self, n_dct_fns: int, n_actions: int, hidden_dims: list[int], 
                 dropout: float = 0.1, beta: float = 1.0):
        super().__init__()
        self.reward_encoder = VAERewardEncoder(n_dct_fns, n_actions, hidden_dims, dropout)
        self.beta = beta  # Weight for KL divergence term in ELBO

    def forward(self, t1: torch.Tensor, t2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for preference prediction with VAE.
        
        Args:
            t1: Trajectory 1 tensor of shape (batch_size, trajectory_length, features)
            t2: Trajectory 2 tensor of shape (batch_size, trajectory_length, features)
            
        Returns:
            Tuple of (preferences, kl_loss, reward_samples) where:
            - preferences: Preference predictions of shape (batch_size, 1)
            - kl_loss: KL divergence loss
            - reward_samples: Dictionary containing reward samples and parameters
        """
        batch_size, traj_length, _ = t1.shape
        
        # Reshape to process all state-action pairs at once
        t1_flat = t1.view(-1, t1.shape[-1])  # (batch_size * traj_length, features)
        t2_flat = t2.view(-1, t2.shape[-1])  # (batch_size * traj_length, features)
        
        # Get VAE outputs for each state-action pair
        r1_flat, mean1, log_var1 = self.reward_encoder(t1_flat)  # (batch_size * traj_length, 1)
        r2_flat, mean2, log_var2 = self.reward_encoder(t2_flat)  # (batch_size * traj_length, 1)
        
        # Reshape back to batch format
        r1 = r1_flat.view(batch_size, traj_length)  # (batch_size, traj_length)
        r2 = r2_flat.view(batch_size, traj_length)  # (batch_size, traj_length)
        
        # Sum rewards along trajectory dimension
        r1_sum = r1.sum(dim=1, keepdim=True)  # (batch_size, 1)
        r2_sum = r2.sum(dim=1, keepdim=True)  # (batch_size, 1)
        
        # Predict preference using sigmoid of reward difference
        preferences = torch.sigmoid(r2_sum - r1_sum)
        
        # Compute KL divergence loss
        # Prior is standard normal N(0, 1) for the reward distribution
        kl_loss1 = self._kl_divergence(mean1, log_var1)
        kl_loss2 = self._kl_divergence(mean2, log_var2)
        kl_loss = kl_loss1 + kl_loss2
        
        # Store reward samples and parameters for analysis
        reward_samples = {
            'r1': r1,
            'r2': r2,
            'r1_sum': r1_sum,
            'r2_sum': r2_sum,
            'mean1': mean1,
            'mean2': mean2,
            'log_var1': log_var1,
            'log_var2': log_var2
        }
        
        return preferences, kl_loss, reward_samples

    def _kl_divergence(self, mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Compute KL divergence between q(reward|x) and p(reward) = N(0, 1).
        
        Args:
            mean: Mean of the approximate posterior reward distribution
            log_var: Log variance of the approximate posterior reward distribution
            
        Returns:
            KL divergence loss
        """
        kl_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        return kl_loss

    def sample_rewards(self, state_action: torch.Tensor, num_samples: int = 10) -> torch.Tensor:
        """
        Sample multiple reward values for given state-action pairs.
        
        Args:
            state_action: State-action features of shape (batch_size, features)
            num_samples: Number of samples to generate
            
        Returns:
            Sampled rewards of shape (num_samples, batch_size, 1)
        """
        mean, log_var = self.reward_encoder.encode(state_action)
        std = torch.exp(0.5 * log_var)
        
        # Expand for multiple samples
        mean_expanded = mean.unsqueeze(0).expand(num_samples, -1, -1)
        std_expanded = std.unsqueeze(0).expand(num_samples, -1, -1)
        
        # Sample from the reward distribution
        eps = torch.randn_like(mean_expanded)
        reward_samples = mean_expanded + eps * std_expanded
        
        return reward_samples 