import torch
from variational_reward_learning.models.variational_encoder import VariationalEncoder
from variational_reward_learning.utils.math import kl_divergence_normal

class VariationalDemonstrationModel(torch.nn.Module):
    """
    VAE-inspired demonstration model that uses a probabilistic reward encoder.
    This model is based on AVRIL (https://arxiv.org/abs/2102.06483).
    The reward itself is the latent variable, and the model optimizes the ELBO objective.
    """

    def __init__(
        self,
        n_dct_fns: int,
        n_actions: int,
        hidden_dims: list[int],
        dropout: float = 0.1,
        gamma: float = 0.99,
        beta: float = 1.0,
        lambd: float = 0.01,
        q_model_hidden_dims: list[int] = [128, 128],
        state_features: torch.Tensor = None  # Precomputed state features
    ):
        super().__init__()
        self.reward_encoder = VariationalEncoder(
            n_dct_fns, n_actions, hidden_dims, dropout)
        self.beta = beta  # Weight for KL divergence term in ELBO
        self.lambd = lambd  # Lagrange multiplier for the Q-value constraint loss
        self.gamma = gamma  # Discount factor for the Q-value constraint loss
        self.n_actions = n_actions
        self.n_dct_fns = n_dct_fns

        # Q-value model
        input_dim = n_dct_fns**2 + n_actions
        q_model_dims = [input_dim] + q_model_hidden_dims + [1]
        q_model_layers = []
        for i in range(len(q_model_dims) - 1):
            q_model_layers.append(torch.nn.Linear(q_model_dims[i], q_model_dims[i+1]))
            if i < len(q_model_dims) - 2:
                q_model_layers.append(torch.nn.LeakyReLU())
                q_model_layers.append(torch.nn.Dropout(dropout))
        self.q_model = torch.nn.Sequential(*q_model_layers)
        
        # Precompute all state-action features
        if state_features is not None:
            self.all_state_action_features = self._create_all_state_action_features(state_features)
        else:
            self.register_buffer('all_state_action_features', None)

    def _create_all_state_action_features(self, state_features: torch.Tensor):
        """
        Create all possible state-action features.
        
        Args:
            state_features: State features tensor of shape (n_states, n_dct_fns**2)
        """
        n_states = state_features.shape[0]
        n_actions = self.n_actions
        n_dct_features = self.n_dct_fns**2
        
        # Create all state-action combinations
        all_features = torch.zeros(n_states * n_actions, n_dct_features + n_actions)
        
        for s in range(n_states):
            for a in range(n_actions):
                # Get state features
                state_feat = state_features[s]
                
                # Create one-hot encoded action
                action_one_hot = torch.zeros(n_actions)
                action_one_hot[a] = 1
                
                # Concatenate state features and action
                all_features[s * n_actions + a] = torch.cat([state_feat, action_one_hot])
        
        # Register as buffer so it moves with the model to the correct device
        self.register_buffer('all_state_action_features', all_features)
        return all_features

    def forward(self, traj: torch.Tensor, traj_indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for demonstration learning with VAE.
        
        Args:
            traj: Expert trajectory tensor of shape (batch_size, trajectory_length, features)
            traj_indices: Trajectory indices tensor of shape (batch_size, trajectory_length, 2) with (state_idx, action_idx) pairs
            
        Returns:
            Tuple of (reward_samples, kl_loss, q_constraint_loss, traj_ll)
        """
        batch_size, traj_length, _ = traj.shape
        
        # Reshape to process all state-action pairs at once
        traj_flat = traj.view(-1, traj.shape[-1])  # (batch_size * traj_length, features)
        
        # Get VAE outputs for each state-action pair in the trajectory
        r_flat, mean, log_var = self.reward_encoder(traj_flat)  # (batch_size * traj_length, 1)
        
        # Reshape back to batch format
        r = r_flat.view(batch_size, traj_length)  # (batch_size, traj_length)
        mean = mean.view(batch_size, traj_length)  # (batch_size, traj_length)
        log_var = log_var.view(batch_size, traj_length)  # (batch_size, traj_length)
        
        # Compute KL divergence loss
        # Prior is standard normal N(0, 1) for the reward distribution
        kl_loss = kl_divergence_normal(mean, log_var)

        n_states = self.all_state_action_features.shape[0] // self.n_actions
        all_q_values = self.q_model(self.all_state_action_features)  # (n_states * n_actions, 1)
        all_q_values = all_q_values.view(n_states, self.n_actions)  # (n_states, n_actions)
        
        # Compute expert policy using softmax: π(a|s) = exp(βQ(s,a)) / sum_b exp(βQ(s,b))
        expert_policy = torch.softmax(self.beta * all_q_values, dim=1)  # (n_states, n_actions)
        
        # Extract state and action indices from trajectory
        state_indices = traj_indices[:, :, 0]  # (batch_size, trajectory_length)
        action_indices = traj_indices[:, :, 1]  # (batch_size, trajectory_length)
        
        # Get the expert policy probabilities for the actions taken in the trajectory
        expert_probs = expert_policy[state_indices, action_indices]  # (batch_size, trajectory_length)
        
        # Compute log-likelihood of the trajectory under the expert policy
        traj_ll = torch.sum(torch.log(expert_probs + 1e-8))  # Add small epsilon for numerical stability


        # Compute the Q-value constraint loss: log(N(Q(s, a) - γQ(s', a'); µ(s, a), σ(s, a)))
        # where µ(s, a) and σ(s, a) are the mean and variance of the reward distribution

        # These should be ≈ R(s, a)
        q = self.q_model(traj_flat)  # (batch_size * traj_length, 1)
        q = q.view(batch_size, traj_length)  # (batch_size, traj_length)
        q_diffs = q[:, :-1] - self.gamma * q[:, 1:]  # (batch_size, traj_length - 1)

        # Compute the log-likelihood of the reward estimates given the reward model
        # Use the mean and log_var from the first (traj_length - 1) timesteps
        # Transform log_var to std for the Normal distribution
        std = torch.exp(0.5 * log_var[:, :-1])
        q_diffs_ll = torch.distributions.Normal(mean[:, :-1], std).log_prob(q_diffs)
        q_constraint_loss = q_diffs_ll.sum() # sum over the batch and trajectory length
        
        # Store reward samples and parameters for analysis
        reward_samples = {
            'r': r,
            'q': q,
            'mean': mean,
            'log_var': log_var,
        }
        
        return reward_samples, kl_loss, q_constraint_loss, traj_ll

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