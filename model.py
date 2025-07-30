import torch

class RewardEncoder(torch.nn.Module):
    """
    Reward model that takes state-action features and predicts the reward.
    """

    def __init__(self, n_dct_fns: int, n_actions: int, hidden_dims: list[int], dropout: float = 0.1):
        super().__init__()
        input_dim = n_dct_fns**2 + n_actions
        dims = [input_dim] + hidden_dims + [1]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(torch.nn.Linear(dims[i], dims[i+1]))
            layers.append(torch.nn.LeakyReLU())
            layers.append(torch.nn.Dropout(dropout))
        layers.append(torch.nn.Linear(dims[-1], 1))
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.layers(state)

class PreferenceModel(torch.nn.Module):
    """
    Preference model that takes trajectory pairs and predicts preferences.
    It uses a reward encoder to predict the rewards of the individual state-action pairs in the trajectories,
    sums them, and then uses a sigmoid to predict the preference.
    """

    def __init__(self, n_dct_fns: int, n_actions: int, hidden_dims: list[int], dropout: float = 0.1):
        super().__init__()
        self.reward_encoder = RewardEncoder(n_dct_fns, n_actions, hidden_dims, dropout)

    def forward(self, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for preference prediction.
        
        Args:
            t1: Trajectory 1 tensor of shape (batch_size, trajectory_length, features)
            t2: Trajectory 2 tensor of shape (batch_size, trajectory_length, features)
            
        Returns:
            Preference predictions of shape (batch_size, 1)
        """
        # Get rewards for each state-action pair in trajectories
        batch_size, traj_length, _ = t1.shape
        
        # Reshape to process all state-action pairs at once
        t1_flat = t1.view(-1, t1.shape[-1])  # (batch_size * traj_length, features)
        t2_flat = t2.view(-1, t2.shape[-1])  # (batch_size * traj_length, features)
        
        # Get rewards for each state-action pair
        r1_flat = self.reward_encoder(t1_flat)  # (batch_size * traj_length, 1)
        r2_flat = self.reward_encoder(t2_flat)  # (batch_size * traj_length, 1)
        
        # Reshape back to batch format
        r1 = r1_flat.view(batch_size, traj_length)  # (batch_size, traj_length)
        r2 = r2_flat.view(batch_size, traj_length)  # (batch_size, traj_length)
        
        # Sum rewards along trajectory dimension
        r1_sum = r1.sum(dim=1, keepdim=True)  # (batch_size, 1)
        r2_sum = r2.sum(dim=1, keepdim=True)  # (batch_size, 1)
        
        # Predict preference using sigmoid of reward difference
        return torch.sigmoid(r2_sum - r1_sum)