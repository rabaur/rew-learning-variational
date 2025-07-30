import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import CenteredNorm
from grid_env import dct_grid_env
from model import RewardEncoder
from data import IIDDCTGridDataset

if __name__ == "__main__":
    grid_size = 32
    n_dct_fns = 16
    P, R, S = dct_grid_env(grid_size=grid_size, n_dct_basis_fns=n_dct_fns, reward_type="dense", p_rand=0.0)

    # plot dct basis functions
    S_square = S.reshape((grid_size, grid_size, -1))
    _, axs = plt.subplots(n_dct_fns, n_dct_fns)
    for i in range(n_dct_fns):
        for j in range(n_dct_fns):
            axs[i, j].imshow(S_square[:, :, i * n_dct_fns + j], cmap="bwr", norm=CenteredNorm(vcenter=0))
    plt.show()

    # draw the MDP
    # draw_mdp(P, R, pos=lambda G: custom_layout(G, grid_size))

    # draw the reward function as a heatmap
    # R_normalized = (R - np.min(R)) / (np.max(R) - np.min(R))
    # plt.imshow(R_normalized.reshape((grid_size, grid_size, -1))[:, :, 2:])
    # plt.show()

    # create design matrix for normal equations
    # we have (n_dct_fns**2 + 1) parameters per action, so we have 5 * (n_dct_fns**2 + 1) parameters in total
    n_params = n_dct_fns ** 2 + 1
    n_s = grid_size ** 2

    # append intercept to each states features
    S_with_intercept = np.hstack((S, np.ones((n_s, 1))))
    X = np.zeros((5 * n_s, 5 * n_params)) # plus 5 for intercept per action
    b = np.zeros((5 * n_s, 1))
    for s in range(n_s):
        for a in range(5):
            # dct features
            X[s * 5 + a, a*n_params:(a + 1)*n_params] = S_with_intercept[s, :]
            b[s * 5 + a] = R[s, a]
    plt.imshow(X, cmap="bwr", norm=CenteredNorm(vcenter=0))
    plt.show()
    
    # solve normal equations
    print(X)
    w = np.linalg.solve(X.T @ X, X.T @ b)
    print(X)
    print(f"number of coeffs: {w.shape[0]}")

    # reconstruct reward function
    R_reconstructed_flat = X @ w
    R_reconstructed = R_reconstructed_flat.reshape((grid_size, grid_size, -1))

    # take only every fifth row of the reconstructed reward function, since the other rows are just repeats

    # draw original and reconstructed reward functions as heatmaps
    _, axs = plt.subplots(5, 2)
    for a in range(5):
        axs[a, 0].imshow(R.reshape((grid_size, grid_size, -1))[:, :, a])
        axs[a, 1].imshow(R_reconstructed.reshape((grid_size, grid_size, -1))[:, :, a])
    plt.show()