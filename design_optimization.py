from scipy.linalg import expm
from pyslsqp import optimize
from pyslsqp.postprocessing import print_dict_as_table
from interpolatory_model import *
import numpy as np
import csv
from compute_inf_reach_set import compute_inf_reach_set
from reach_set_functions import compute_hull
from functools import partial
from read_hdf5 import extract_dataset_from_results
from trim_aircraft import compute_trim_eom

''' This script solves the actual design optimization problem. It loads aircraft parameters from a CSV file and on each iteration,
builds the A,B matrices while computing aerodynamic force/moment coefficients as a function of physical/operating parameters.'''

def load_parameters(csv_file_path):
    """
    Reads a CSV file containing parameter names and values, ignoring the first row (header),
    and loads the parameters into a dictionary.
    """
    parameters = {}
    with open(csv_file_path, mode="r") as csv_file:
        reader = csv.reader(csv_file)
        next(reader)  # Skip the header row
        for row in reader:
            param_name = row[0]
            param_value = float(row[1])  # Convert the value to float
            parameters[param_name] = param_value
    return parameters

def one_norm(x):
    return x[0] + x[1]

def volume_obj(x, flight_point):
    ''' This objective computes the negative of the volume of the p-norm reach set for design variable x. Negative so we
    can maximize it.'''

    hull = set_hull(x,flight_point)
    print(f"Design variables: {x}")
    vol = hull.volume
    print(f"Volume of convex hull: {vol}")
    return -vol  # Negative for maximization

def scaled_volume_obj(x, flight_point):
    ''' This objective just computes volume_obj, but scales it up to make optimization easier'''
    hull_vol = volume_obj(x, flight_point)
    return hull_vol * 1e2  # Scale to make optimization easier

def set_hull(x,desired_vel):
    ''' This function computes the A,B matrices of the system, computes the reach set, and returns the convex hull of the points.
    It will also save the data from the reach set computation into a csv if desired

    Input: x: array of design variables, where x[0] = center halfspan of bwb, x[1] = wing halfspan of bwb, 
    desired_vel is the desired trim velocity. 
    
    Output: reachable set hull: scipy.spatial.ConvexHull object'''

    # First construct A,B matrices
    data_file = 'Aero_data/aero_7dim2.npz'
    interp_dict = aero_interps(data_file)

    # Given new design, recompute trim point
    chs = x[0]
    whs = x[1]
    alpha0, deltae0, deltaT0, pitch0 = compute_trim_eom(np.asarray([desired_vel,chs,whs]), data_file,climb_angle=0)
    print(f'Trimmed alpha: {np.rad2deg(alpha0)}, deltae: {np.rad2deg(deltae0)}, deltaT: {deltaT0}, pitch: {np.rad2deg(pitch0)}')

    # Compute input bounds
    thrust_ub = 1-deltaT0
    thrust_lb = -deltaT0
    elev_ub = np.deg2rad(30) - deltae0
    elev_lb = np.deg2rad(-30) - deltae0

    u_upper = np.array([thrust_ub, elev_ub])
    u_lower = np.array([thrust_lb, elev_lb])

    flight_point = np.array([desired_vel, alpha0, deltae0])

    full_flight_point = np.concatenate((flight_point[0:2], x, flight_point[2:]))
    A,B = build_AB(interp_dict,full_flight_point)

    tf = 1
    N_res = 6     # Resolution for the costate sampling

    # With this, actually compute the set
    control_cost, final_positions = compute_inf_reach_set(A,B,tf,N_res,u_ub = u_upper, u_lb = u_lower, \
                                                                 save_data=False,filename='Output/bwb_inf_data.csv')

    # Next, take the convex hull of the points in final_positions, computing inclusion based on control cost.
    hull = compute_hull(control_cost, final_positions, cost_threshold=1)

    return hull

def vol_constraint(x,flight_point):
    '''This is a second constraint that we can use to test the optimization.
    Instead of the trace of the reach gramian, we can use volume of p-norm set.
    This will be much slower, and requires iterating over computation of p-norm reach sets'''
    
    hull = set_hull(x,flight_point)
    print(f"Volume of convex hull: {hull.volume}")

    init_vol = 0.1408 # Initial volume for inf set at 
    init_vol = init_vol * 100

    return np.array([100*hull.volume - 5*init_vol]) 

def compute_axis_extent(x,v,fp):
    ''' This constraint computes the length of the p-norm reach set projected along a given direction.'''
    hull = set_hull(x,fp)

    # Define axis for projection
    axis = v
    axis = axis/np.linalg.norm(axis)

    hull_points = hull.points[hull.vertices]
    projections = hull_points @ axis

    # Compute min and max position of projections along axis
    proj_min, proj_max = projections.min(), projections.max()
    extent = proj_max - proj_min
    print(f"Extent along direction {v}: {extent}")
    return extent

def direction_constraint(x,v,extent0,fp):
    ''' This is a wrapper for compute_axis_extent to be used as a constraint in the optimization.
    Projects onto fourth axis [0,0,0,1]'''
    extent = compute_axis_extent(x,v,fp)
    return np.array([extent - 1.15 * extent0]) # Want to increase extent by at least 15%.

def direction_obj(x,v,fp):
    print(f"Design variables: {x}")
    extent = compute_axis_extent(x,v,fp)
    return -(extent) # Want to increase extent.

if __name__ == '__main__':
    '''Now, the design variable is only going to be center-halfspan and wing-halfspan.
    we can use partial to pass the other arguments to building A,B.
    At each iteration, we will build A,B, compute the Lp norm reach set, compute its volume, and then optimize.

    flight_point will contain [alt, V, pitch, er], the full one should contain [alt, V, pitch, CHS, WHS, er].
    chs and whs were separated out because they are the design variables that we will alter, the others will remain fixed for now.
    '''

    chs = 5 # m
    whs = 12 # m
    x0 = np.array([chs, whs])

    V0 = 200 # m/s

    flight_params = V0

    # Axis on which to project reachable set during some optimization problems
    v = np.array([[0,0,np.cos(np.deg2rad(110)),np.sin(np.deg2rad(110))]]).T

    init_extent = compute_axis_extent(x0,v,flight_params)

    # Partial function for objective functions
    fixed_Lp_objective = partial(volume_obj,flight_point=V0)
    fixed_scaled_Lp_obj = partial(scaled_volume_obj,flight_point=V0)

    # Partial function for direction objective
    fixed_direction_obj = partial(direction_obj,v=v,fp=V0)
    fixed_direction_con = partial(direction_constraint,v=v,extent0=init_extent, fp=V0)

    # Set up optimization
    num_eqcon = 0  # No equality constraints
    x_low = np.array([3, 10])
    x_upp = np.array([7,20])            # Use the range that I swept across.

    output_string = 'Output/aero_7dim_asym'

    '''This will purely maximize the volume of reach set with some bounds on design variables. No constraints.'''
    results = optimize(x0, obj=fixed_scaled_Lp_obj, meq = num_eqcon, xl=x_low, xu = x_upp, \
                       visualize = True, save_figname=f"{output_string}/N6_volmax/optimization_progress.png", summary_filename=f"{output_string}/N6_volmax/design_optimization.out", \
                        save_itr = 'major', save_filename = f'{output_string}/N6_volmax/opt_vars.hdf5')
    print_dict_as_table(results)

    '''Optimize some direction'''
    results = optimize(x0, obj=fixed_direction_obj, meq = num_eqcon, con=fixed_direction_con, xl=x_low, xu = x_upp, \
                        visualize = True, save_figname=f"{output_string}/N6_110max/optimization_progress.png", summary_filename=f"{output_string}/N6_110max/design_optimization.out", \
                            save_itr = 'major', save_filename = f'{output_string}/N6_110max/opt_vars.hdf5') 
    print_dict_as_table(results)

    '''Optimized for volume with direction constraint'''
    results = optimize(x0, obj=fixed_scaled_Lp_obj, meq = num_eqcon, con=fixed_direction_con, xl=x_low, xu = x_upp, \
                        visualize = True, save_figname=f"{output_string}/N6_volmax_110con/optimization_progress.png", summary_filename=f"{output_string}/N6_volmax_110con/design_optimization.out", \
                            save_itr = 'major', save_filename = f'{output_string}/N6_volmax_110con/opt_vars.hdf5')
    print_dict_as_table(results)
    
