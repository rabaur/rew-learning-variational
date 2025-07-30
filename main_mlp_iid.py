import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import CenteredNorm
from grid_env import dct_grid_env
from model import RewardEncoder
from data import IIDDCTGridDataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

if __name__ == "__main__":
    grid_size = 16
    n_dct_fns = 8
    n_actions = 5
    n_states = grid_size ** 2
    n_dct_features = n_dct_fns ** 2
    P, R, S = dct_grid_env(grid_size=grid_size, n_dct_basis_fns=n_dct_fns, reward_type="sparse", p_rand=0.0)

    # Prepare data for training
    # Reshape S to have shape (n_states, n_dct_features)
    state_features = S.reshape(n_states, -1)  # Shape: (n_states, n_dct_fns^2)

    # Create dataset
    dataset = IIDDCTGridDataset(state_features, n_actions, R)
    
    # Create data loader
    batch_size = 16
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    hidden_dims = [64, 32]
    model = RewardEncoder(n_dct_fns=n_dct_fns, n_actions=n_actions, hidden_dims=hidden_dims)
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
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
        
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            # Move data to device
            inputs = inputs.float().to(device)
            targets = targets.float().to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
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
        
        for inputs, targets in dataloader:
            inputs = inputs.float().to(device)
            targets = targets.float().to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_eval_loss += loss.item()
            num_eval_batches += 1
        
        final_loss = total_eval_loss / num_eval_batches
        print(f"Final evaluation loss: {final_loss:.6f}")
    
    # draw the reward function as a heatmap
    # get the state features and one-hot encoded actions
    state_action_feats = dataset.state_action_feats
    R_reconstructed = model(torch.tensor(state_action_feats).float().to(device)).detach().cpu().numpy()
    R_reconstructed = R_reconstructed.reshape((grid_size, grid_size, -1))
    _, axs = plt.subplots(5, 2)
    for a in range(5):
        axs[a, 0].imshow(R.reshape((grid_size, grid_size, -1))[:, :, a])
        axs[a, 1].imshow(R_reconstructed.reshape((grid_size, grid_size, -1))[:, :, a])
    plt.show()