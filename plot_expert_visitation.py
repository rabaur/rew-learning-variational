import numpy as np
import matplotlib.pyplot as plt
from grid_env import dct_grid_env
from variational_reward_learning.data.datasets import DemonstrationDataset

def plot_expert_visitation():
    # Create environment
    grid_size = 64
    n_dct_fns = 8
    n_actions = 5
    n_states = grid_size ** 2
    
    # Create environment
    P, R, S = dct_grid_env(grid_size=grid_size, n_dct_basis_fns=n_dct_fns, reward_type="sparse", p_rand=0.0)
    
    # Create initial state distribution (uniform)
    init_state_dist = np.zeros(n_states)
    init_state_dist[0] = 1.0
    
    # Create demonstration dataset
    num_samples = 512
    trajectory_length = 100
    beta_true = 5.0
    gamma = 0.99
    
    dataset = DemonstrationDataset(
        T=P,
        R_true=R,
        init_state_dist=init_state_dist,
        S_features=S,
        n_actions=n_actions,
        rationality=beta_true,
        gamma=gamma,
        num_steps=trajectory_length,
        num_samples=num_samples,
        seed=42
    )
    
    # Get all demonstrations
    demonstrations = dataset.get_all_demonstrations()
    
    # Count state visits
    state_visits = np.zeros(n_states)
    
    for trajectory in demonstrations:
        for state, action, reward in trajectory:
            state_visits[state] += 1
    
    # Reshape to grid format
    state_visits_grid = state_visits.reshape((grid_size, grid_size))
    
    # Create the plot
    plt.figure(figsize=(12, 10))
    
    # Plot state visitation heatmap
    plt.subplot(2, 2, 1)
    im1 = plt.imshow(state_visits_grid, cmap='viridis')
    plt.title('Expert State Visitation Distribution')
    plt.colorbar(im1, label='Number of Visits')
    plt.xlabel('Grid X')
    plt.ylabel('Grid Y')
    
    # Plot log-scale version for better visualization
    plt.subplot(2, 2, 2)
    log_visits = np.log(state_visits_grid + 1)  # Add 1 to avoid log(0)
    im2 = plt.imshow(log_visits, cmap='viridis')
    plt.title('Expert State Visitation (Log Scale)')
    plt.colorbar(im2, label='Log(Number of Visits + 1)')
    plt.xlabel('Grid X')
    plt.ylabel('Grid Y')
    
    # Plot true rewards for comparison
    plt.subplot(2, 2, 3)
    true_rewards = R.reshape((grid_size, grid_size, n_actions))
    # Sum rewards across actions for visualization
    total_rewards = np.sum(true_rewards, axis=2)
    im3 = plt.imshow(total_rewards, cmap='viridis')
    plt.title('True Rewards (Sum across Actions)')
    plt.colorbar(im3, label='Total Reward')
    plt.xlabel('Grid X')
    plt.ylabel('Grid Y')
    
    # Plot action distribution (most common action per state)
    plt.subplot(2, 2, 4)
    action_counts = np.zeros((n_states, n_actions))
    
    for trajectory in demonstrations:
        for state, action, reward in trajectory:
            action_counts[state, action] += 1
    
    # Find most common action per state
    most_common_actions = np.argmax(action_counts, axis=1).reshape((grid_size, grid_size))
    im4 = plt.imshow(most_common_actions, cmap='tab10', vmin=0, vmax=n_actions-1)
    plt.title('Most Common Expert Action per State')
    plt.colorbar(im4, label='Action Index')
    plt.xlabel('Grid X')
    plt.ylabel('Grid Y')
    
    plt.tight_layout()
    plt.savefig('figures/expert_visitation_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print some statistics
    print(f"Total state visits: {np.sum(state_visits)}")
    print(f"Average visits per state: {np.mean(state_visits):.2f}")
    print(f"Max visits to any state: {np.max(state_visits)}")
    print(f"Min visits to any state: {np.min(state_visits)}")
    print(f"States never visited: {np.sum(state_visits == 0)}")
    print(f"States visited at least once: {np.sum(state_visits > 0)}")
    
    # Find most and least visited states
    most_visited_state = np.argmax(state_visits)
    least_visited_states = np.where(state_visits == 0)[0]
    
    print(f"\nMost visited state: {most_visited_state} (visited {state_visits[most_visited_state]} times)")
    if len(least_visited_states) > 0:
        print(f"Sample of never-visited states: {least_visited_states[:10]}")

if __name__ == "__main__":
    plot_expert_visitation() 