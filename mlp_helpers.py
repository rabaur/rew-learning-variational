from numpy.typing import NDArray
import numpy as np

def simulate_trajectory(
    P: NDArray,
    R: NDArray,
    policy: NDArray,
    init_state_dist: NDArray,
    n_steps: int
) -> list[tuple[int, int, float]]:
    """
    Simulate a trajectory of the MDP given by P and R according to the policy.

    Args:
        P: Transition matrix.
        R: Reward matrix.
        policy: Policy matrix.
        init_state_dist: Initial state distribution (vector of length P.shape[0]).
        n_steps: Number of steps to simulate.

    Returns:
        List of tuples of (state, action, reward).
    """
    trajectory = []
    n_states = P.shape[0]
    n_actions = P.shape[1]

    # Sample initial state from init_state_dist
    initial_state = np.random.choice(np.arange(n_states), p=init_state_dist)

    state = initial_state
    for _ in range(n_steps):
        action = np.random.choice(np.arange(n_actions), p=policy[state, :])
        next_state = np.random.choice(np.arange(n_states), p=P[state, action, :])
        reward = R[state, action]
        trajectory.append((state, action, reward))
        state = next_state
    return trajectory


def compute_trajectory_return(trajectory: list[tuple[int, int, float]], reward_matrix: NDArray) -> float:
    """Compute the total return of a trajectory given a reward matrix."""
    total_return = 0.0
    for state, action, _ in trajectory:
        total_return += reward_matrix[state, action]
    return total_return