import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from p_set_data_read import unpack_p_set_data

def compute_hull(control_cost, final_positions, cost_threshold):
    ''' This function computes convex hull of our final positions, only considering points associated with control cost under cost_threshold
    INPUT: control_cost, final_positions, max value of control: cost_threshold
    OUTPUT: volume of convex hull'''

    if final_positions.shape == (6,6,6,6,4):
        filtered_positions = final_positions[control_cost <= cost_threshold]
    else:
        filtered_positions = final_positions
    
    if len(filtered_positions) < 4:
        print("Not enough points to compute a convex hull. Must recompute p-norm reach set.")
        return 0
    
    # Compute the convex hull of the filtered positions
    hull = ConvexHull(filtered_positions)
    
    # Return the volume of the convex hull
    return hull

# def compute_percentage_under(data,data_bound):
#     ''' This function computes the percentage of elements in the data array
#     that stay under the value of data_bound. In the case of checking control cost,
#     we use this to check how many controllers being sampled are under the cost.
#     INPUT: one dimensional data matrix, data bound
#     OUTPUT: percentage of elements in data under data_bound (in units of percentage)'''

#     valid_pts = np.sum(data <= data_bound)
#     total_elements = data.size
#     percentage_valid = (valid_pts / total_elements) * 100

#     return percentage_valid

# def find_best_lgrid(A, B, p, tf, desired_coverage=55, threshold = 10, l_grid = 1, max_iters = 20, control_max = 1):
#     ''' This computes the best lgrid to compute the p-norm reach sets
#     It will iterate at most max_iters times to find the best one.
#     INPUT: system matrices A,B, 
#             p-norm: p, 
#             final time: tf, 
#             desired percentage of lambdas corresponding to valid controllers: desired_coverage (default 55%)
#             margin of error to hit desired coverage: threshold (default 10%)
#             starting grid: l_grid (default 1)
#             increment to l_grid each iteration: l_grid_increment (default 0.5)
#             maximum number of iterations: max_iters (default 20)
#             maximum control cost to be considered valid: control_max (default 1)
#     OUTPUT: final lgrid'''

#     N_res = 3   # initial search for best lambda grid will use this resolution
#     m = B.shape[1]  # Number of control inputs
#     n = A.shape[0]  # Number of states
#     iters = 0
#     coverage = 0
    
#     # If our initial coverage is too small, we need to DECREASE l_grid
#     # If our initial coverage is too large, we need to INCREASE l_grid
#     init_coverage = control_grid_coverage(A,B,p,tf,l_grid,N_res,control_max)
#     if abs(init_coverage - desired_coverage) <= threshold:
#         l_grid_increment = 0
#     elif desired_coverage-init_coverage > 0:
#         # If we don't have enough coverage, we need to refine
#         l_grid_increment = -0.1
#     else:
#         # If we have more than enough coverage, widen search
#         l_grid_increment = 1

#     while abs(coverage - desired_coverage) > threshold and iters < max_iters and (l_grid + l_grid_increment) > 0:
#         iters+=1
#         l_grid += l_grid_increment
#         print(f"Trying l_grid = {l_grid}")
#         coverage = control_grid_coverage(A,B,p,tf,l_grid,N_res,control_max)
#         print(f"coverage: {coverage}")
#     return l_grid

# def p_norm_control(A, B, t, lambda_0, p):
#     if p % 2 != 0:
#         raise ValueError("Must use an even value of p")
#     b = -B.T @ expm(-A.T * t) @ lambda_0
#     return np.sign(b) * (np.abs(b) ** (1/(p - 1)))

# def control_grid_coverage(A,B,p,tf,l_grid,N_res,control_max=1):
#     # Create the grid for each dimension

#     m = B.shape[1]  # Number of control inputs

#     lx1 = np.linspace(-l_grid, l_grid, N_res)
#     lx2 = np.linspace(-l_grid, l_grid, N_res)
#     lx3 = np.linspace(-l_grid, l_grid, N_res)
#     lx4 = np.linspace(-l_grid, l_grid, N_res)

#     # Create a 4D meshgrid
#     LX1, LX2, LX3, LX4 = np.meshgrid(lx1, lx2, lx3, lx4)

#     # Initialize arrays to store control costs and final positions
#     control_cost = np.zeros((N_res, N_res, N_res, N_res))
#     for i in range(N_res):
#             for j in range(N_res):
#                 for k in range(N_res):
#                     for l in range(N_res):
#                         # Each iteration, pick a new initial condition
#                         lambda1 = LX1[i, j, k, l]
#                         lambda2 = LX2[i, j, k, l]
#                         lambda3 = LX3[i, j, k, l]
#                         lambda4 = LX4[i, j, k, l]
#                         lambda0 = np.array([lambda1, lambda2, lambda3, lambda4])

#                         nt = 1000
#                         t_list = np.linspace(0, tf, nt)
#                         p_control_list = np.zeros((m, nt))

#                         # Compute control costs and simulate the system
#                         for q, t in enumerate(t_list):
#                             p_control_list[:, q] = p_norm_control(A, B, t, lambda0, p)

#                         control_cost[i, j, k, l] = trapezoid(np.sum(np.abs(p_control_list) ** p, axis=0), t_list)
#     coverage = compute_percentage_under(control_cost.reshape(-1,1), control_max)
#     return coverage

# def plot_columns(data, col_x, col_y):
#     """
#     Plots two columns of a 2D array against each other.

#     Parameters:
#     - data: 2D NumPy array of shape (n_samples, n_features)
#     - col_x: Index of the column to plot on the x-axis
#     - col_y: Index of the column to plot on the y-axis
#     """
#     x = data[:, col_x]
#     y = data[:, col_y]

#     plt.figure(figsize=(8, 6))
#     plt.scatter(x, y, s=10, alpha=0.7, label=f"Columns {col_x} vs {col_y}")
#     plt.xlabel(f"Column {col_x}")
#     plt.ylabel(f"Column {col_y}")
#     plt.title(f"Scatter Plot of Column {col_x} vs Column {col_y}")
#     plt.legend()
#     plt.grid()
#     plt.show()

def plot_hull_projection(hull, col_x, col_y, hull_plot_label="Convex Hull", point_label = "Projected Points", hull_color = "r",
                         x_label=None, y_label=None, title=None, hatchstyle=None, ls=None):
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
    plt.plot(x[hull_vertices], y[hull_vertices], f'{hull_color}-', label=f"{hull_plot_label}", linestyle=ls)
    plt.fill(x[hull_vertices], y[hull_vertices], f'{hull_color}', alpha=0.2, hatch=hatchstyle)
    # plt.scatter(x, y, s=35, label=f"{point_label}")

    plt.xlabel(x_label if x_label else f"Dimension {col_x}")
    plt.ylabel(y_label if y_label else f"Dimension {col_y}")
    # plt.title(title if title else f"Projection of Convex Hull onto Dimensions {col_x} and {col_y}")
    plt.title(title if title else None)

    # plt.legend()
    plt.grid()
    plt.axis('equal')
    # plt.show()

if __name__ == "__main__":
    N_res = 6
    [LX1, LX2, LX3, LX4, control_cost, final_positions] = unpack_p_set_data("test_sphere.csv",N_res)
    cost_threshold = 1  # Define your cost threshold here
    hull = compute_hull(control_cost, final_positions, cost_threshold)
    
    # set_fill = compute_percentage_under(control_cost.reshape(-1,1),1)
    print(f"Computed Volume of Reachable Set: {hull.volume}")
    # print(f"Set coverage: {set_fill}%")
    
    # Assuming `hull` is the 4D convex hull computed earlier
    plot_hull_projection(hull, col_x=0, col_y=1)  # Project onto dimensions 0 and 1
    plot_hull_projection(hull, col_x=2, col_y=3)  # Project onto dimensions 2 and 3