import pandas as pd
import numpy as np

def unpack_set_data(csv_input,N_res):
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

    # Reshape the final_positions array
    final_positions = np.zeros((N_res, N_res, N_res, N_res, 4))
    final_positions[..., 0] = df["final_x1"].values.reshape(N_res, N_res, N_res, N_res)
    final_positions[..., 1] = df["final_x2"].values.reshape(N_res, N_res, N_res, N_res)
    final_positions[..., 2] = df["final_x3"].values.reshape(N_res, N_res, N_res, N_res)
    final_positions[..., 3] = df["final_x4"].values.reshape(N_res, N_res, N_res, N_res)

    # final_positions = final_positions.reshape(N_res,N_res,N_res,N_res,4)

    return LX1, LX2, LX3, LX4, final_positions


if __name__ == "__main__":
    N_res = 5
    [LX1,LX2,LX3,LX4,fp] = unpack_set_data("reachability_data.csv",N_res)
    print(LX1)
    print(fp)