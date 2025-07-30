import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import CenteredNorm
from grid_env import dct_grid_env
from model import PreferenceModel
from model_vae import VAEPreferenceModel
from data import PreferenceDataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time

def train_mle_model(dataloader, n_dct_fns, n_actions, hidden_dims, num_epochs, device):
    """Train the original MLE preference model."""
    print("Training MLE model...")
    
    # Initialize model
    model = PreferenceModel(n_dct_fns=n_dct_fns, n_actions=n_actions, hidden_dims=hidden_dims)
    model.to(device)
    
    # Loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    model.train()
    start_time = time.time()
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (traj1_batch, traj2_batch, preferences) in enumerate(dataloader):
            # Move data to device
            traj1_batch = traj1_batch.to(device)
            traj2_batch = traj2_batch.to(device)
            preferences = preferences.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            pred_preferences = model(traj1_batch, traj2_batch)
            loss = criterion(pred_preferences, preferences)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        # Print progress
        avg_loss = total_loss / num_batches
        if (epoch + 1) % 50 == 0:
            print(f"  MLE Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.6f}")
    
    training_time = time.time() - start_time
    print(f"MLE training completed in {training_time:.2f} seconds")
    
    return model, training_time

def train_vae_model(dataloader, n_dct_fns, n_actions, hidden_dims, beta_vae, num_epochs, device):
    """Train the VAE preference model."""
    print("Training VAE model...")
    
    # Initialize model
    model = VAEPreferenceModel(
        n_dct_fns=n_dct_fns, 
        n_actions=n_actions, 
        hidden_dims=hidden_dims,
        beta=beta_vae
    )
    model.to(device)
    
    # Loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    model.train()
    start_time = time.time()
    
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
        
        if (epoch + 1) % 50 == 0:
            print(f"  VAE Epoch [{epoch+1}/{num_epochs}]")
            print(f"    ELBO Loss: {avg_loss:.6f}")
            print(f"    Reconstruction Loss: {avg_reconstruction_loss:.6f}")
            print(f"    KL Loss: {avg_kl_loss:.6f}")
    
    training_time = time.time() - start_time
    print(f"VAE training completed in {training_time:.2f} seconds")
    
    return model, training_time

def evaluate_model(model, dataloader, device, model_type="MLE"):
    """Evaluate a trained model."""
    model.eval()
    criterion = nn.BCELoss()
    
    with torch.no_grad():
        total_loss = 0.0
        num_batches = 0
        correct_predictions = 0
        total_predictions = 0
        
        for traj1_batch, traj2_batch, preferences in dataloader:
            # Move data to device
            traj1_batch = traj1_batch.to(device)
            traj2_batch = traj2_batch.to(device)
            preferences = preferences.to(device)
            
            if model_type == "VAE":
                pred_preferences, kl_loss, reward_samples = model(traj1_batch, traj2_batch)
                loss = criterion(pred_preferences, preferences)
            else:
                pred_preferences = model(traj1_batch, traj2_batch)
                loss = criterion(pred_preferences, preferences)
            
            total_loss += loss.item()
            num_batches += 1
            
            # Calculate accuracy
            pred_classes = (pred_preferences > 0.5).float()
            correct_predictions += (pred_classes == preferences).sum().item()
            total_predictions += preferences.size(0)
        
        final_loss = total_loss / num_batches
        accuracy = correct_predictions / total_predictions
        
        return final_loss, accuracy

def main():
    # Configuration
    grid_size = 32
    n_dct_fns = 8
    n_actions = 5
    n_states = grid_size ** 2
    
    # Create environment
    P, R, S = dct_grid_env(grid_size=grid_size, n_dct_basis_fns=n_dct_fns, reward_type="dense", p_rand=0.0)
    
    # Create a simple policy (uniform random)
    policy = np.ones((n_states, n_actions)) / n_actions
    
    # Create initial state distribution (uniform)
    init_state_dist = np.ones(n_states) / n_states
    
    # Create preference dataset
    num_samples = 2000
    trajectory_length = 50
    beta_true = 1.0
    
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
    batch_size = 16
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Model parameters
    hidden_dims = [64, 32]
    beta_vae = 1.0
    num_epochs = 200
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Dataset size: {len(dataset)}")
    
    # Train MLE model
    mle_model, mle_time = train_mle_model(dataloader, n_dct_fns, n_actions, hidden_dims, num_epochs, device)
    
    # Train VAE model
    vae_model, vae_time = train_vae_model(dataloader, n_dct_fns, n_actions, hidden_dims, beta_vae, num_epochs, device)
    
    # Evaluate both models
    print("\nEvaluating models...")
    mle_loss, mle_accuracy = evaluate_model(mle_model, dataloader, device, "MLE")
    vae_loss, vae_accuracy = evaluate_model(vae_model, dataloader, device, "VAE")
    
    # Print comparison results
    print("\n" + "="*50)
    print("COMPARISON RESULTS")
    print("="*50)
    print(f"MLE Model:")
    print(f"  Training Time: {mle_time:.2f} seconds")
    print(f"  Final Loss: {mle_loss:.6f}")
    print(f"  Accuracy: {mle_accuracy:.4f}")
    print(f"  Parameters: {sum(p.numel() for p in mle_model.parameters()):,}")
    print()
    print(f"VAE Model:")
    print(f"  Training Time: {vae_time:.2f} seconds")
    print(f"  Final Loss: {vae_loss:.6f}")
    print(f"  Accuracy: {vae_accuracy:.4f}")
    print(f"  Parameters: {sum(p.numel() for p in vae_model.parameters()):,}")
    print(f"  Reward encoder outputs: 2 dimensions (mean, variance)")
    print(f"  Beta VAE: {beta_vae}")
    
    # Visualize learned rewards comparison
    print("\nVisualizing learned rewards comparison...")
    
    # Create state-action features for all states and actions
    state_features = S.reshape(n_states, -1)
    state_action_feats = np.zeros((n_actions * state_features.shape[0], state_features.shape[1] + n_actions))
    
    for i in range(state_features.shape[0]):
        for a in range(n_actions):
            action_one_hot = np.zeros(n_actions)
            action_one_hot[a] = 1
            state_action_feats[i * n_actions + a] = np.concatenate([state_features[i], action_one_hot])
    
    state_action_tensor = torch.tensor(state_action_feats, dtype=torch.float32).to(device)
    
    # Get MLE rewards
    with torch.no_grad():
        R_mle = mle_model.reward_encoder(state_action_tensor).detach().cpu().numpy()
        R_mle = R_mle.reshape((grid_size, grid_size, n_actions))
    
    # Get VAE rewards (mean and uncertainty)
    with torch.no_grad():
        mean, log_var = vae_model.reward_encoder.encode(state_action_tensor)
        std = torch.exp(0.5 * log_var)
        
        R_vae_mean = mean.detach().cpu().numpy()
        R_vae_std = std.detach().cpu().numpy()
        
        R_vae_mean = R_vae_mean.reshape((grid_size, grid_size, n_actions))
        R_vae_std = R_vae_std.reshape((grid_size, grid_size, n_actions))
    
    # Plot comparison
    fig, axs = plt.subplots(n_actions, 4, figsize=(20, 3*n_actions))
    for a in range(n_actions):
        # True rewards
        axs[a, 0].imshow(R.reshape((grid_size, grid_size, n_actions))[:, :, a])
        axs[a, 0].set_title(f'True Reward - Action {a}')
        axs[a, 0].axis('off')
        
        # MLE learned rewards
        im = axs[a, 1].imshow(R_mle[:, :, a])
        axs[a, 1].set_title(f'MLE Learned Reward - Action {a}')
        axs[a, 1].axis('off')
        plt.colorbar(im, ax=axs[a, 1])
        
        # VAE learned mean rewards
        im = axs[a, 2].imshow(R_vae_mean[:, :, a])
        axs[a, 2].set_title(f'VAE Learned Mean - Action {a}')
        axs[a, 2].axis('off')
        plt.colorbar(im, ax=axs[a, 2])
        
        # VAE uncertainty
        im = axs[a, 3].imshow(R_vae_std[:, :, a])
        axs[a, 3].set_title(f'VAE Uncertainty - Action {a}')
        axs[a, 3].axis('off')
        plt.colorbar(im, ax=axs[a, 3])
    
    plt.tight_layout()
    plt.show()
    
    # Calculate reward reconstruction error
    print("\nReward Reconstruction Analysis:")
    mle_error = np.mean((R_mle - R.reshape((grid_size, grid_size, n_actions)))**2)
    vae_error = np.mean((R_vae_mean - R.reshape((grid_size, grid_size, n_actions)))**2)
    
    print(f"MLE Mean Squared Error: {mle_error:.6f}")
    print(f"VAE Mean Squared Error: {vae_error:.6f}")
    print(f"VAE provides uncertainty estimates with mean std: {np.mean(R_vae_std):.6f}")
    
    print("\nComparison completed!")

if __name__ == "__main__":
    main() 