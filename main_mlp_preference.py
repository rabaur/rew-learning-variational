import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import CenteredNorm
from grid_env import dct_grid_env
from model import PreferenceModel
from data import PreferenceDataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def main():
    grid_size = 16
    n_dct_fns = 8
    n_actions = 5
    n_states = grid_size ** 2
    n_dct_features = n_dct_fns ** 2
    
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
    
    # Initialize model
    hidden_dims = [64, 32]
    model = PreferenceModel(n_dct_fns=n_dct_fns, n_actions=n_actions, hidden_dims=hidden_dims)
    
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
    
    model.train()
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
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.6f}")
    
    print("Training completed!")
    
    # Evaluate the model
    model.eval()
    with torch.no_grad():
        total_eval_loss = 0.0
        num_eval_batches = 0
        correct_predictions = 0
        total_predictions = 0
        
        for traj1_batch, traj2_batch, preferences in dataloader:
            # Move data to device
            traj1_batch = traj1_batch.to(device)
            traj2_batch = traj2_batch.to(device)
            preferences = preferences.to(device)
            
            pred_preferences = model(traj1_batch, traj2_batch)
            loss = criterion(pred_preferences, preferences)
            
            total_eval_loss += loss.item()
            num_eval_batches += 1
            
            # Calculate accuracy
            pred_classes = (pred_preferences > 0.5).float()
            correct_predictions += (pred_classes == preferences).sum().item()
            total_predictions += preferences.size(0)
        
        final_loss = total_eval_loss / num_eval_batches
        accuracy = correct_predictions / total_predictions
        print(f"Final evaluation loss: {final_loss:.6f}")
        print(f"Accuracy: {accuracy:.4f}")
    
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
    
    # Get learned rewards using the reward encoder
    state_action_tensor = torch.tensor(state_action_feats, dtype=torch.float32).to(device)
    R_learned = model.reward_encoder(state_action_tensor).detach().cpu().numpy()
    R_learned = R_learned.reshape((grid_size, grid_size, n_actions))
    
    # Plot original vs learned rewards
    fig, axs = plt.subplots(n_actions, 2, figsize=(10, 3*n_actions))
    for a in range(n_actions):
        axs[a, 0].imshow(R.reshape((grid_size, grid_size, n_actions))[:, :, a])
        axs[a, 0].set_title(f'True Reward - Action {a}')
        axs[a, 0].axis('off')
        
        axs[a, 1].imshow(R_learned[:, :, a])
        axs[a, 1].set_title(f'Learned Reward - Action {a}')
        axs[a, 1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("Preference-based reward learning completed!") 

if __name__ == "__main__":
    main()