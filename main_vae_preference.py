import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import CenteredNorm
from grid_env import dct_grid_env
from model_vae import VAEPreferenceModel
from data import PreferenceDataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os

def main():
    # Create figures directory if it doesn't exist
    os.makedirs("figures", exist_ok=True)
    
    grid_size = 32
    n_dct_fns = 8
    n_actions = 5
    n_states = grid_size ** 2
    n_dct_features = n_dct_fns ** 2
    
    # Create environment
    P, R, S = dct_grid_env(grid_size=grid_size, n_dct_basis_fns=n_dct_fns, reward_type="sparse", p_rand=0.0)
    
    # Create a simple policy (uniform random)
    policy = np.ones((n_states, n_actions)) / n_actions
    
    # Create initial state distribution (uniform)
    init_state_dist = np.ones(n_states) / n_states
    
    # Create preference dataset
    num_samples = 3000
    trajectory_length = 50
    beta_true = 2.0
    
    dataset = PreferenceDataset(
        num_samples=num_samples,
        P=P,
        R_true=R,
        policy=policy,
        init_state_dist=init_state_dist,
        S=S,
        n_actions=n_actions,
        beta_true=beta_true,
        trajectory_length=trajectory_length,
        seed=42
    )
    
    # Create data loader
    batch_size = 128
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize VAE model
    hidden_dims = [64, 32]
    beta_vae = 0.1 # Weight for KL divergence term
    model = VAEPreferenceModel(
        n_dct_fns=n_dct_fns, 
        n_actions=n_actions, 
        hidden_dims=hidden_dims,
        beta=beta_vae
    )
    
    # Loss function and optimizer
    criterion = nn.BCELoss()  # Binary Cross Entropy for preference prediction
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    num_epochs = 300
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    print(f"Training on device: {device}")
    print(f"Dataset size: {len(dataset)}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Reward encoder outputs: 2 dimensions (mean, variance)")
    print(f"Beta VAE weight: {beta_vae}")
    
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        total_kl_loss = 0.0
        total_reconstruction_loss = 0.0
        num_batches = 0
        
        for batch_idx, (traj1_batch, traj2_batch, preferences) in enumerate(dataloader):
            # Move data to device
            traj1_batch = traj1_batch.to(device)
            traj2_batch = traj2_batch.to(device)
            preferences = preferences.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            pred_preferences, kl_loss, reward_samples = model(traj1_batch, traj2_batch)
            
            # Reconstruction loss (data likelihood)
            reconstruction_loss = criterion(pred_preferences, preferences)
            
            # Total ELBO loss
            elbo_loss = reconstruction_loss + beta_vae * kl_loss
            
            # Backward pass
            elbo_loss.backward()
            optimizer.step()
            
            total_loss += elbo_loss.item()
            total_kl_loss += kl_loss.item()
            total_reconstruction_loss += reconstruction_loss.item()
            num_batches += 1
        
        # Print progress
        avg_loss = total_loss / num_batches
        avg_kl_loss = total_kl_loss / num_batches
        avg_reconstruction_loss = total_reconstruction_loss / num_batches
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}]")
            print(f"  ELBO Loss: {avg_loss:.6f}")
            print(f"  Reconstruction Loss: {avg_reconstruction_loss:.6f}")
            print(f"  KL Loss: {avg_kl_loss:.6f}")
            print(f"  Beta * KL: {beta_vae * avg_kl_loss:.6f}")
    
    print("Training completed!")
    
    # Evaluate the model
    model.eval()
    with torch.no_grad():
        total_eval_loss = 0.0
        total_eval_kl_loss = 0.0
        total_eval_reconstruction_loss = 0.0
        num_eval_batches = 0
        correct_predictions = 0
        total_predictions = 0
        
        for traj1_batch, traj2_batch, preferences in dataloader:
            # Move data to device
            traj1_batch = traj1_batch.to(device)
            traj2_batch = traj2_batch.to(device)
            preferences = preferences.to(device)
            
            pred_preferences, kl_loss, reward_samples = model(traj1_batch, traj2_batch)
            reconstruction_loss = criterion(pred_preferences, preferences)
            elbo_loss = reconstruction_loss + beta_vae * kl_loss
            
            total_eval_loss += elbo_loss.item()
            total_eval_kl_loss += kl_loss.item()
            total_eval_reconstruction_loss += reconstruction_loss.item()
            num_eval_batches += 1
            
            # Calculate accuracy
            pred_classes = (pred_preferences > 0.5).float()
            correct_predictions += (pred_classes == preferences).sum().item()
            total_predictions += preferences.size(0)
        
        final_loss = total_eval_loss / num_eval_batches
        final_kl_loss = total_eval_kl_loss / num_eval_batches
        final_reconstruction_loss = total_eval_reconstruction_loss / num_eval_batches
        accuracy = correct_predictions / total_predictions
        
        print(f"Final evaluation:")
        print(f"  ELBO Loss: {final_loss:.6f}")
        print(f"  Reconstruction Loss: {final_reconstruction_loss:.6f}")
        print(f"  KL Loss: {final_kl_loss:.6f}")
        print(f"  Accuracy: {accuracy:.4f}")
    
    # Visualize learned rewards
    print("Visualizing learned rewards...")
    
    # Create state-action features for all states and actions
    state_features = S.reshape(n_states, -1)
    state_action_feats = np.zeros((n_actions * state_features.shape[0], state_features.shape[1] + n_actions))
    
    for i in range(state_features.shape[0]):
        for a in range(n_actions):
            action_one_hot = np.zeros(n_actions)
            action_one_hot[a] = 1
            state_action_feats[i * n_actions + a] = np.concatenate([state_features[i], action_one_hot])
    
    # Get learned rewards using the VAE reward encoder
    state_action_tensor = torch.tensor(state_action_feats, dtype=torch.float32).to(device)
    
    # Get mean rewards and uncertainties directly from encoder
    with torch.no_grad():
        mean, log_var = model.reward_encoder.encode(state_action_tensor)
        std = torch.exp(0.5 * log_var)
        
        R_learned_mean = mean.detach().cpu().numpy()
        R_learned_std = std.detach().cpu().numpy()
    
    R_learned_mean = R_learned_mean.reshape((grid_size, grid_size, n_actions))
    R_learned_std = R_learned_std.reshape((grid_size, grid_size, n_actions))
    
    # Plot original vs learned rewards
    fig, axs = plt.subplots(n_actions, 3, figsize=(15, 3*n_actions))
    fig.suptitle('VAE Preference-based Reward Learning: True vs Learned Rewards with Uncertainty', fontsize=16)
    
    for a in range(n_actions):
        # True rewards
        im1 = axs[a, 0].imshow(R.reshape((grid_size, grid_size, n_actions))[:, :, a], cmap='viridis')
        axs[a, 0].set_title(f'True Reward - Action {a}')
        axs[a, 0].axis('off')
        plt.colorbar(im1, ax=axs[a, 0])
        
        # Learned mean rewards
        im2 = axs[a, 1].imshow(R_learned_mean[:, :, a], cmap='viridis')
        axs[a, 1].set_title(f'Learned Mean Reward - Action {a}')
        axs[a, 1].axis('off')
        plt.colorbar(im2, ax=axs[a, 1])
        
        # Learned uncertainty
        im3 = axs[a, 2].imshow(R_learned_std[:, :, a], cmap='viridis')
        axs[a, 2].set_title(f'Learned Uncertainty - Action {a}')
        axs[a, 2].axis('off')
        plt.colorbar(im3, ax=axs[a, 2])
    
    plt.tight_layout()
    plt.savefig('figures/vae_reward_learning.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Additional analysis: Reward distribution visualization
    print("Analyzing reward distributions...")
    
    # Get reward distributions for a subset of state-action pairs
    subset_size = 1000
    subset_indices = np.random.choice(len(state_action_feats), subset_size, replace=False)
    subset_features = state_action_tensor[subset_indices]
    
    with torch.no_grad():
        mean, log_var = model.reward_encoder.encode(subset_features)
        reward_samples = model.reward_encoder.reparameterize(mean, log_var)
        
        # Plot reward distribution
        plt.figure(figsize=(10, 6))
        plt.hist(reward_samples.cpu().numpy().flatten(), bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('Reward Value')
        plt.ylabel('Frequency')
        plt.title('Distribution of Sampled Rewards')
        plt.grid(True, alpha=0.3)
        plt.savefig('figures/vae_reward_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Plot reward variance distribution
        plt.figure(figsize=(10, 6))
        plt.hist(log_var.cpu().numpy().flatten(), bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('Log Variance')
        plt.ylabel('Frequency')
        plt.title('Distribution of Reward Log Variances')
        plt.grid(True, alpha=0.3)
        plt.savefig('figures/vae_variance_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Plot mean vs variance scatter
        plt.figure(figsize=(10, 6))
        plt.scatter(mean.cpu().numpy().flatten(), 
                   log_var.cpu().numpy().flatten(), 
                   alpha=0.6, s=20)
        plt.xlabel('Reward Mean')
        plt.ylabel('Reward Log Variance')
        plt.title('Reward Mean vs Variance')
        plt.grid(True, alpha=0.3)
        plt.savefig('figures/vae_mean_variance_scatter.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    print("VAE preference-based reward learning completed and figures saved!")

if __name__ == "__main__":
    main() 