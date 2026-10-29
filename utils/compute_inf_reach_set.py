import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from scipy.integrate import trapezoid
import pandas as pd
from utils.progress_bar import printProgressBar
from utils.reach_set_functions import compute_hull, plot_hull_projection
from utils.reach_set_data_read import unpack_set_data
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

    
    m = B.shape[1]  # Number of control inputs
    n = A.shape[0]  # Number of states
    x0 = np.zeros(n)  # Initial condition

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
    
    def sample_sphere_boundary(num_samples,n):
        x = np.random.randn(num_samples, n)
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        return x
    
    sample_num = N_res**n
    lx = sample_sphere_boundary(sample_num,n)

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
        # Create dynamic column names based on state dimension
        data = {}
        
        # Add costate initial conditions
        for i in range(n):
            data[f"LX{i+1}"] = lx[:, i]
        
        # Add final positions
        for i in range(n):
            data[f"final_x{i+1}"] = final_positions[:, i]
            

        df = pd.DataFrame(data)

        # Save to CSV
        df.to_csv(filename, index=False)

        print(f"\nData saved to {filename}")

    return control_cost, final_positions

if __name__ == "__main__":

    A = np.array([[2, 3],[-1, 1]])
    B = np.array([[0],[1]])
    # A= np.array([[-0.00352121133,  5.87480073,  0, -9.81],
    #              [-0.000681553965, -0.520980253,  1, 0],
    #              [0, -2.63577136,  0,  0],
    #              [ 0,  0,  1,  0]])
    # B = np.array([[2.84071645, 0.32619862],
    #               [0.        , 0.06223328],
    #               [0.        , 1.75401816],
    #               [0.        ,0.        ]])

    tf = 2
    N_res = 20

    control_cost, final_positions = compute_inf_reach_set_sphere(A, B, tf, N_res, save_data=False,filename='2d_mag.csv')
    hull = compute_hull(control_cost, final_positions, 1)
    plt.figure()
    fsize = 18
    plot_hull_projection(hull, col_x = 0, col_y = 1,hull_plot_label='Convex Hull', point_label='Sampled Trajectories')
    plt.xlabel("$x_1$", fontsize=fsize)
    plt.ylabel('$x_2$', fontsize=fsize)
    plt.legend(fontsize=0.6*fsize,loc='best')
    plt.show()

    # plt.savefig("/Users/stevennguyen/Library/CloudStorage/OneDrive-Personal/UCSD/Research/ACC2026/Presentation/test_set.png",dpi=300)
    # print(f"Computed Volume of Reachable Set: {hull.volume}")



