import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import pandas as pd

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
    projected_hull = ConvexHull(np.column_stack((x, y)))

    # Extract the vertices of the projected hull and close the hull by adding the first vertex to the end
    hull_vertices = np.append(projected_hull.vertices, projected_hull.vertices[0])

    # Plot the projected convex hull
    plt.plot(x[hull_vertices], y[hull_vertices], f'{hull_color}-', label=f"{hull_plot_label}", linestyle=ls)
    plt.fill(x[hull_vertices], y[hull_vertices], f'{hull_color}', alpha=0.2, hatch=hatchstyle)
    # plt.scatter(x, y, s=35, label=f"{point_label}")

    plt.xlabel(x_label if x_label else f"Dimension {col_x}")
    plt.ylabel(y_label if y_label else f"Dimension {col_y}")
    plt.title(title if title else None)
    plt.grid()
    plt.axis('equal')

def unpack_reach_set_data(csv_input,N_res):
    '''This function reads the CSV file containing the reachability data and 
    outputs: [LX1, LX2, LX3, LX4, control_cost, final_positions]. final_positions[...,i]
    gives the i-th component of the final state corresponding to that initial costate'''

    # Load the data from the CSV file
    df = pd.read_csv(csv_input)

    # Extract the number of resolutions (N_res) from the data
    # N_res = int(round(len(df["lx1"]) ** (1 / 4)))  # Assuming a 4D grid

    # Reshape the LX arrays
    LX1 = df["LX1"].values.reshape(N_res, N_res, N_res, N_res)
    LX2 = df["LX2"].values.reshape(N_res, N_res, N_res, N_res)
    LX3 = df["LX3"].values.reshape(N_res, N_res, N_res, N_res)
    LX4 = df["LX4"].values.reshape(N_res, N_res, N_res, N_res)

    # Reshape the control_cost array
    control_cost = df["control_cost"].values.reshape(N_res, N_res, N_res, N_res)

    # Reshape the final_positions array
    final_positions = np.zeros((N_res, N_res, N_res, N_res, 4))
    final_positions[..., 0] = df["final_x1"].values.reshape(N_res, N_res, N_res, N_res)
    final_positions[..., 1] = df["final_x2"].values.reshape(N_res, N_res, N_res, N_res)
    final_positions[..., 2] = df["final_x3"].values.reshape(N_res, N_res, N_res, N_res)
    final_positions[..., 3] = df["final_x4"].values.reshape(N_res, N_res, N_res, N_res)

    final_positions = final_positions.reshape(N_res,N_res,N_res,N_res,4)

    return LX1, LX2, LX3, LX4, control_cost, final_positions

if __name__ == "__main__":
    N_res = 6
    [LX1, LX2, LX3, LX4, control_cost, final_positions] = unpack_p_set_data("test_sphere.csv",N_res)
    cost_threshold = 1  # Define your cost threshold here
    hull = compute_hull(control_cost, final_positions, cost_threshold)

    print(f"Computed Volume of Reachable Set: {hull.volume}")
    
    # Assuming `hull` is the 4D convex hull computed earlier
    plot_hull_projection(hull, col_x=0, col_y=1)  # Project onto dimensions 0 and 1
    plot_hull_projection(hull, col_x=2, col_y=3)  # Project onto dimensions 2 and 3