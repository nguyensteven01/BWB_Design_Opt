"""Plot the eigenvectors of a 4x4 reachability Gramian.

Eigenvectors are projected onto the first two state axes (V, alpha) and the
last two state axes (q, theta), matching the state ordering used throughout
this repo (see utils/interpolatory_model.py:build_AB).
"""

import numpy as np
import matplotlib.pyplot as plt
from utils.read_hdf5 import extract_dataset_from_results
from utils.gramian_metrics import gramian

def gramian_eigs(gram):
    """Compute eigenvalues/eigenvectors of a symmetric 4x4 Gramian, sorted descending."""
    gram = np.asarray(gram, dtype=float)
    if gram.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 Gramian, got shape {gram.shape}")
    if not np.allclose(gram, gram.T, atol=1e-8):
        raise ValueError("Gramian must be symmetric.")

    eigvals, eigvecs = np.linalg.eigh(gram)
    order = np.argsort(eigvals)[::-1]
    return eigvals[order], eigvecs[:, order]


def plot_gramian_eigenvectors(gram, state_labels=(r'Velocity, $V$', r'Angle of Attack, $\alpha$', r'Pitch Rate, $q$', r'Pitch Angle, $\theta$'),
                               scale_by_eigenvalue=True):
    """
    Plot eigenvectors of a 4x4 Gramian projected onto axes (0,1) and (2,3).

    If scale_by_eigenvalue, each eigenvector is drawn with length sqrt(eigenvalue),
    which is the semi-axis length of the reachable ellipsoid {x : x^T W^-1 x <= 1}
    along that eigenvector direction.
    """
    eigvals, eigvecs = gramian_eigs(gram)
    n = len(eigvals)
    colors = plt.cm.viridis(np.linspace(0, 0.85, n))

    plt.rcParams.update({'font.size': 15})

    # fig, axes = plt.subplots(1, 2, figsize=(10, 8))
    # pairs = [(0, 1), (2, 3)]

    lengths = np.sqrt(eigvals) if scale_by_eigenvalue else np.ones(n)

    # Only plot one

    i = 2
    j = 3
    fig, ax = plt.subplots()
    for k in range(n):
        v = eigvecs[:, k]
        vx, vy = v[i] * lengths[k], v[j] * lengths[k]

        ax.annotate('', xy=(vx, vy), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='-|>', color=colors[k], lw=3))
        
        # uncomment this to plot arrows both directions
        # ax.annotate('', xy=(-vx, -vy), xytext=(0, 0),
        #             arrowprops=dict(arrowstyle='-|>', color=colors[k], lw=1, alpha=0.4))
        ax.plot([], [], color=colors[k], label=fr'$\lambda_{k+1}$ = {eigvals[k]:.3g}')

        # annotate() arrows aren't picked up by autoscale, so set limits manually
        extent = np.max(np.abs(eigvecs[[i, j], :] * lengths)) * 1.2
        ax.set_xlim(-extent, extent)
        ax.set_ylim(-extent, extent)

        ax.axhline(0, color='k', lw=0.5, alpha=0.3)
        ax.axvline(0, color='k', lw=0.5, alpha=0.3)
        ax.set_xlabel(state_labels[i])
        ax.set_ylabel(state_labels[j])
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')

    fig.tight_layout()
    return fig, ax


if __name__ == '__main__':

    # opt_path = 'Output/aero_7dim_asym/N6_initial'
    # x_opt = extract_dataset_from_results(f'{opt_path}/opt_vars.hdf5', 'x')
    # x_opt = np.asarray([5, 12])
    x_opt = np.asarray([5, 12, 35, 4.9, 10])
    gram = gramian(x_opt, desired_vel=200)

    print('Gramian:')
    print(gram)

    eigvals, eigvecs = gramian_eigs(gram)
    print('Eigenvalues:', eigvals)
    print('Eigenvectors (columns):')
    print(eigvecs)

    scale_plot = False

    fig, axes = plot_gramian_eigenvectors(gram, scale_by_eigenvalue=scale_plot)

    axes.set_title(f'Eigenvector projection')
    plt.show()
