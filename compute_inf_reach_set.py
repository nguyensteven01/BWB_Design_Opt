import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from scipy.integrate import trapezoid
import pandas as pd
from progress_bar import printProgressBar
from p_set_functions import compute_hull, find_best_lgrid, plot_columns
from p_set_data_read import unpack_p_set_data
import matplotlib.pyplot as plt

'''This script will sample costate initial conditions and then simulate trajectories
corresponding to controls of sufficiently low cost. The final states, control costs,
and costate initial condition grid will be output into a csv file.'''

def compute_inf_reach_set_sphere(A,B,tf,N_res,u_ub = [1,1], u_lb = [-1,-1],save_data=False,filename="Output/reachability_data.csv"):
    '''This function computes the reachable set of a continuous-time LTI system
    , where the control is chosen to satisfy some magnitude bound from [0,T].

    Inputs:
    A, B: System matrices
    tf: Final time
    N_res: Resolution along each axis for the grid of costate initial conditions
    u_ub,u_lb: arrays that contain upper and lower bounds for inputs thrust and elevator. u_ub = [thrust_max, elev_max]. u_lb = [thrust_min, elev_min].
    save_data: optionally choose to write out data to csv
    filename: Name of file to which reachability data is saved

    Outputs:
    control_cost: Array of control costs corresponding to each initial condition
    final_positions: Array of final states corresponding to each initial condition'''

    np.random.seed(50)

    x0 = np.array([0, 0, 0, 0])  # Initial condition
    m = B.shape[1]  # Number of control inputs
    n = A.shape[0]  # Number of states

    # Function to simulate the continuous-time LTI system
    def sys_sim(t, x, u, A, B):
        x = x.reshape(-1,1)
        u = u.reshape(-1,1)
        x_prime = A @x + B @ u
        return x_prime.reshape(-1)     # Requires reshaping x because solve_ivp wants that...
    
    def inf_norm_control(A,B,t,lambda_0):
        b = -B.T@expm(-A.T*t)@lambda_0
        u = np.zeros(m)
        for i,bi in enumerate(b):
            u[i] = u_ub[i] if np.sign(bi) >= 0 else u_lb[i]
        return u
    
    def sample_sphere_boundary(num_samples):
        n = 4
        x = np.random.randn(num_samples, n)
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        return x
    
    sample_num = N_res**n
    lx = sample_sphere_boundary(sample_num)

    # Create the grid for each dimension
    # lx1 = np.linspace(-l_grid, l_grid, N_res)
    # lx2 = np.linspace(-l_grid, l_grid, N_res)
    # lx3 = np.linspace(-l_grid, l_grid, N_res)
    # lx4 = np.linspace(-l_grid, l_grid, N_res)

    # # Create a 4D meshgrid
    # LX1, LX2, LX3, LX4 = np.meshgrid(lx1, lx2, lx3, lx4)

    # Initialize arrays to store control costs and final positions
    # control_cost = np.zeros((N_res, N_res, N_res, N_res))
    # final_positions = np.zeros((N_res, N_res, N_res, N_res, 4))  # 4D positions

    control_cost = np.zeros((sample_num,1))
    final_positions = np.zeros((sample_num,n))

    for i in range(sample_num):
        lambda0 = np.array(lx[i,:])
        control_cost[i] = 1 # placeholder
        # Simulate the system using solve_ivp
        def ode_func(t, x):
            u = inf_norm_control(A, B, t, lambda0)
            return sys_sim(t, x, u, A, B)

        sol = solve_ivp(ode_func, [0, tf], x0, t_eval=[tf])  # This value of t_eval makes it so we only store final point of traj.
        final_positions[i,:] = sol.y[:, -1]
        printProgressBar(i, sample_num-1, prefix = 'Progress:', suffix = 'Complete', length = 50)

    if save_data:
        # Flatten the 4D arrays for saving
        fp1 = final_positions[:,0]
        fp2 = final_positions[:,1]
        fp3 = final_positions[:,2]
        fp4 = final_positions[:,3]
        control_cost_flat = control_cost.flatten()
        LX1_flat = lx[:,0]
        LX2_flat = lx[:,1]
        LX3_flat = lx[:,2]
        LX4_flat = lx[:,3]

        # Combine all data into a DataFrame
        data = {
            "LX1": LX1_flat,
            "LX2": LX2_flat,
            "LX3": LX3_flat,
            "LX4": LX4_flat,
            "control_cost": control_cost_flat,
            "final_x1": fp1,
            "final_x2": fp2,
            "final_x3": fp3,
            "final_x4": fp4,
        }

        df = pd.DataFrame(data)

        # Save to CSV
        df.to_csv(filename, index=False)

        print(f"\nData saved to {filename}")

    return control_cost, final_positions


if __name__ == "__main__":

    A = np.identity(4)
    A= np.array([[-0.00352121133,  5.87480073,  0, -9.81],
                 [-0.000681553965, -0.520980253,  1, 0],
                 [0, -2.63577136,  0,  0],
                 [ 0,  0,  1,  0]])
    B = np.array([[2.84071645, 0.32619862],
                  [0.        , 0.06223328],
                  [0.        , 1.75401816],
                  [0.        ,0.        ]])

    tf = 1
    N_res = 6
    l_grid = 5

    control_cost, final_positions = compute_inf_reach_set_sphere(A, B, tf, N_res, save_data=True,filename='test_sphere.csv')
    # LX1, LX2, LX3, LX4, control_cost, final_positions = unpack_p_set_data("reach_data_opt.csv",N_res)

    # hull = compute_hull(control_cost, final_positions, 1)
    # print(f"Computed Volume of Reachable Set: {hull.volume}")



