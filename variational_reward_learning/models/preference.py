import torch
from variational_reward_learning.models.variational_encoder import VariationalEncoder
from variational_reward_learning.utils.math import kl_divergence_normal

class VariationalPreferenceModel(torch.nn.Module):
    """
    VAE-inspired preference model that uses a probabilistic reward encoder.
    The reward itself is the latent variable, and the model optimizes the ELBO objective.
    """

    def __init__(self, n_dct_fns: int, n_actions: int, hidden_dims: list[int], 
                 dropout: float = 0.1, beta: float = 1.0):
        super().__init__()
        self.reward_encoder = VariationalEncoder(
            n_dct_fns, n_actions, hidden_dims, dropout)
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
        kl_loss1 = kl_divergence_normal(mean1, log_var1)
        kl_loss2 = kl_divergence_normal(mean2, log_var2)
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