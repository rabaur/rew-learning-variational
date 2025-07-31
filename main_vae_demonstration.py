import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import CenteredNorm
from grid_env import dct_grid_env
from variational_reward_learning.models.demonstration import VariationalDemonstrationModel
from variational_reward_learning.data.datasets import DemonstrationDataset
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
    
    # Create initial state distribution (uniform)
    init_state_dist = np.zeros(n_states)
    init_state_dist[0] = 1.0
    
    # Create demonstration dataset
    num_samples = 128
    trajectory_length = 100
    rationality = 5.0
    gamma = 0.99
    
    dataset = DemonstrationDataset(
        T=P,
        R_true=R,
        init_state_dist=init_state_dist,
        S_features=S,
        n_actions=n_actions,
        rationality=rationality,
        gamma=gamma,
        num_steps=trajectory_length,
        num_samples=num_samples,
        seed=42
    )
    
    # Create data loader
    batch_size = 16
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize VAE model
    hidden_dims = [64, 32]
    beta_vae = 1.0  # Weight for KL divergence term
    lambd = 1.0    # Weight for Q-value constraint loss
    
    # Convert state features to tensor
    S_tensor = torch.tensor(S, dtype=torch.float32)
    
    model = VariationalDemonstrationModel(
        n_dct_fns=n_dct_fns, 
        n_actions=n_actions, 
        hidden_dims=hidden_dims,
        beta=rationality,
        lambd=lambd,
        gamma=gamma,
        state_features=S_tensor
    )
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    
    # Training loop
    num_epochs = 1000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    print(f"Training on device: {device}")
    print(f"Dataset size: {len(dataset)}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Reward encoder outputs: 2 dimensions (mean, variance)")
    print(f"Beta VAE weight: {beta_vae}")
    print(f"Lambda (Q-constraint weight): {lambd}")
    
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        total_kl_loss = 0.0
        total_q_constraint_loss = 0.0
        total_traj_ll = 0.0
        num_batches = 0
        
        for batch_idx, (trajectories, traj_indices) in enumerate(dataloader):
            # Move data to device
            trajectories = trajectories.to(device)
            traj_indices = traj_indices.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            reward_samples, kl_loss, q_constraint_loss, traj_ll = model(trajectories, traj_indices)
            
            # Total ELBO loss according to AVRIL paper
            # Objective: maximize trajectory likelihood while minimizing KL divergence
            # and ensuring Q-value consistency
            elbo_loss = -traj_ll + beta_vae * kl_loss - lambd * q_constraint_loss
            
            # Backward pass
            elbo_loss.backward()
            optimizer.step()
            
            total_loss += elbo_loss.item()
            total_kl_loss += kl_loss.item()
            total_q_constraint_loss += q_constraint_loss.item()
            total_traj_ll += traj_ll.item()
            num_batches += 1
        
        # Print progress
        avg_loss = total_loss / num_batches
        avg_kl_loss = total_kl_loss / num_batches
        avg_q_constraint_loss = total_q_constraint_loss / num_batches
        avg_traj_ll = total_traj_ll / num_batches
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}]")
            print(f"  ELBO Loss: {avg_loss:.6f}")
            print(f"  Trajectory LL: {avg_traj_ll:.6f}")
            print(f"  KL Loss: {avg_kl_loss:.6f}")
            print(f"  Q-Constraint Loss: {avg_q_constraint_loss:.6f}")
            print(f"  Beta * KL: {beta_vae * avg_kl_loss:.6f}")
            print(f"  Lambda * Q-Constraint: {lambd * avg_q_constraint_loss:.6f}")
    
    print("Training completed!")
    
    # Evaluate the model
    model.eval()
    with torch.no_grad():
        total_eval_loss = 0.0
        total_eval_kl_loss = 0.0
        total_eval_q_constraint_loss = 0.0
        total_eval_traj_ll = 0.0
        num_eval_batches = 0
        
        for trajectories, traj_indices in dataloader:
            # Move data to device
            trajectories = trajectories.to(device)
            traj_indices = traj_indices.to(device)
            
            reward_samples, kl_loss, q_constraint_loss, traj_ll = model(trajectories, traj_indices)
            elbo_loss = -traj_ll + beta_vae * kl_loss - lambd * q_constraint_loss
            
            total_eval_loss += elbo_loss.item()
            total_eval_kl_loss += kl_loss.item()
            total_eval_q_constraint_loss += q_constraint_loss.item()
            total_eval_traj_ll += traj_ll.item()
            num_eval_batches += 1
        
        final_loss = total_eval_loss / num_eval_batches
        final_kl_loss = total_eval_kl_loss / num_eval_batches
        final_q_constraint_loss = total_eval_q_constraint_loss / num_eval_batches
        final_traj_ll = total_eval_traj_ll / num_eval_batches
        
        print(f"Final evaluation:")
        print(f"  ELBO Loss: {final_loss:.6f}")
        print(f"  Trajectory LL: {final_traj_ll:.6f}")
        print(f"  KL Loss: {final_kl_loss:.6f}")
        print(f"  Q-Constraint Loss: {final_q_constraint_loss:.6f}")
    
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
    fig.suptitle('VAE Demonstration-based Reward Learning: True vs Learned Rewards with Uncertainty', fontsize=16)
    
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
    plt.savefig('figures/vae_demonstration_reward_learning.png', dpi=300, bbox_inches='tight')
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
        plt.title('Distribution of Sampled Rewards (Demonstration Learning)')
        plt.grid(True, alpha=0.3)
        plt.savefig('figures/vae_demonstration_reward_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Plot reward variance distribution
        plt.figure(figsize=(10, 6))
        plt.hist(log_var.cpu().numpy().flatten(), bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('Log Variance')
        plt.ylabel('Frequency')
        plt.title('Distribution of Reward Log Variances (Demonstration Learning)')
        plt.grid(True, alpha=0.3)
        plt.savefig('figures/vae_demonstration_variance_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Plot mean vs variance scatter
        plt.figure(figsize=(10, 6))
        plt.scatter(mean.cpu().numpy().flatten(), 
                   log_var.cpu().numpy().flatten(), 
                   alpha=0.6, s=20)
        plt.xlabel('Reward Mean')
        plt.ylabel('Reward Log Variance')
        plt.title('Reward Mean vs Variance (Demonstration Learning)')
        plt.grid(True, alpha=0.3)
        plt.savefig('figures/vae_demonstration_mean_variance_scatter.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    print("VAE demonstration-based reward learning completed and figures saved!")

if __name__ == "__main__":
    main() 