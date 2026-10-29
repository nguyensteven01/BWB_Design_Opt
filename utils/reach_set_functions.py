import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from utils.reach_set_data_read import unpack_set_data
import pandas as pd

def compute_hull(final_positions, cost_threshold = 1, control_cost = 1):
    ''' This function computes convex hull of our final positions, only considering points associated with control cost under cost_threshold
    INPUT: control_cost, final_positions, max value of control: cost_threshold
    OUTPUT: volume of convex hull'''

    if final_positions.shape == (6,6,6,6,4):
        filtered_positions = final_positions[control_cost <= cost_threshold]
    else:
        filtered_positions = final_positions.reshape((-1,4))
    
    if len(filtered_positions) < 4:
        print("Not enough points to compute a convex hull. Must recompute p-norm reach set.")
        return 0
    
    # Compute the convex hull of the filtered positions
    hull = ConvexHull(filtered_positions)
    
    # Return the volume of the convex hull
    return hull

def plot_hull_projection(hull, col_x, col_y, hull_plot_label="Convex Hull", point_label = "Projected Points", hull_color = "r",
                         x_label=None, y_label=None, title=None, hatchstyle=None, ls=None, points = False):
    """
    Plots the projection of a 4D convex hull onto two specified dimensions.

    Parameters:
    hull: A ConvexHull object representing the 4D convex hull.
    col_x: Index of the dimension to plot on the x-axis.
    col_y: Index of the dimension to plot on the y-axis.
    """
    # Extract the vertices of the convex hull
    vertices = hull.points[hull.vertices]

    # Project the vertices onto the specified dimensions
    x = vertices[:, col_x]
    y = vertices[:, col_y]

    # Create a 2D convex hull of the projected points
    # This is useful so that we can pull out its vertices and draw boundaries of the projection.
    # If we only plotted (x,y), we would get messy zig zag since some points are only vertices in higher dimensions.
    projected_hull = ConvexHull(np.column_stack((x, y)))

    # Extract the vertices of the projected hull and close the hull by adding the first vertex to the end
    hull_vertices = np.append(projected_hull.vertices, projected_hull.vertices[0])

    # Plot the projected convex hull
    # plt.figure(figsize=(8, 6))
    plt.plot(x[hull_vertices], y[hull_vertices], f'{hull_color}', label=f"{hull_plot_label}", linestyle=ls)
    plt.fill(x[hull_vertices], y[hull_vertices], f'{hull_color}', alpha=0.2, hatch=hatchstyle)
    if points:
        plt.scatter(x, y, s=35, label=f"{point_label}")

    plt.xlabel(x_label if x_label else f"Dimension {col_x}")
    plt.ylabel(y_label if y_label else f"Dimension {col_y}")
    # plt.title(title if title else f"Projection of Convex Hull onto Dimensions {col_x} and {col_y}")
    plt.title(title if title else None)

    # plt.legend()
    plt.grid()
    plt.axis('equal')
    # plt.show()


def sample_ellipsoid_surface(A, gamma, N, rng,savepts=False, filename="Outputs/gramian_set.csv"):
    """
    Draw N points on the ellipsoid surface {x in R^n : x^T A x = gamma}.
    Map the unit sphere through x = sqrt(gamma) * R^{-1} u, A = R^T R (Cholesky),
    ||u|| = 1  =>  x^T A x = gamma.
    """
    n = A.shape[0]
    R = np.linalg.cholesky(A).T                       # A = R^T R
    g = rng.standard_normal((N, n))
    u = g / np.linalg.norm(g, axis=1, keepdims=True)  # uniform on unit sphere
    X =  np.sqrt(gamma) * np.linalg.solve(R, u.T).T  # (N, n) on the surface
    if savepts:
        # save the points
        # Create dynamic column names based on state dimension
        data = {}

        # Add final positions
        for i in range(n):
            data[f"final_x{i+1}"] = X[:, i]

        df = pd.DataFrame(data)

        # Save to CSV
        df.to_csv(filename, index=False)

        print(f"\nData saved to {filename}")
        pass
    return ConvexHull(X)


def plot_projected_boundary(A, gamma, label=None, color="C0", ax=None,
                            N=20000, seed=0, draw_analytic=True):
    """
    Sample the 4D ellipsoid SURFACE, project onto (x1,x2) and (x3,x4).
    The convex hull of the projected surface points is the shadow's boundary.
    Optionally overlay the exact analytic shadow ellipse.
    """
    A = np.asarray(A, dtype=float)
    if not np.allclose(A, A.T, atol=1e-10):
        raise ValueError("A must be symmetric.")
    if np.min(np.linalg.eigvalsh(A)) <= 0:
        raise ValueError("A must be positive definite.")
    if gamma <= 0:
        raise ValueError("gamma must be positive.")

    if ax is None:
        fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    else:
        axes = np.atleast_1d(ax).ravel()
        fig = axes[0].figure

    rng = np.random.default_rng(seed)
    X = sample_ellipsoid_surface(A, gamma, N, rng)

    Ainv = np.linalg.inv(A)
    theta = np.linspace(0, 2 * np.pi, 400)
    circle = np.vstack([np.cos(theta), np.sin(theta)])
    pairs = [(0, 1), (2, 3)]

    for k, (ax_k, (i, j)) in enumerate(zip(axes, pairs)):
        Y = X[:, [i, j]]                              # projected surface points

        # outline = convex hull of the projected points (the shadow boundary)
        from scipy.spatial import ConvexHull
        h = ConvexHull(Y)
        ring = np.append(h.vertices, h.vertices[0])   # close the loop
        ax_k.plot(Y[ring, 0], Y[ring, 1], color=color, lw=1.5,
                  label=label if k == 0 else None)

        if draw_analytic:
            P = Ainv[np.ix_([i, j], [i, j])]
            L = np.linalg.cholesky(P)
            pts = np.sqrt(gamma) * (L @ circle)
            ax_k.plot(pts[0], pts[1], color="k", lw=1.0, ls="--")

        ax_k.axhline(0, color="k", lw=0.5, alpha=0.3)
        ax_k.axvline(0, color="k", lw=0.5, alpha=0.3)
        ax_k.set_xlabel(f"$x_{i+1}$")
        ax_k.set_ylabel(f"$x_{j+1}$")
        ax_k.set_title(f"proj onto ($x_{i+1}$, $x_{j+1}$)")
        # ax_k.set_aspect("equal", adjustable="datalim")
        ax_k.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig, axes



if __name__ == "__main__":
    N_res = 6
    [LX1, LX2, LX3, LX4, control_cost, final_positions] = unpack_set_data("Output/aero_7dim_asym/N6_initial.csv",N_res)
    cost_threshold = 1  # Define your cost threshold here
    hull = compute_hull(control_cost, final_positions, cost_threshold)
    
    # set_fill = compute_percentage_under(control_cost.reshape(-1,1),1)
    print(f"Computed Volume of Reachable Set: {hull.volume}")
    # print(f"Set coverage: {set_fill}%")
    
    # Assuming `hull` is the 4D convex hull computed earlier
    plot_hull_projection(hull, col_x=0, col_y=1)  # Project onto dimensions 0 and 1
    plot_hull_projection(hull, col_x=2, col_y=3)  # Project onto dimensions 2 and 3