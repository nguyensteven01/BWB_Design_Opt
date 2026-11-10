from pyslsqp import optimize
from pyslsqp.postprocessing import print_dict_as_table
from utils.interpolatory_model import *
import numpy as np
import csv
from utils.compute_inf_reach_set import compute_inf_reach_set_sphere
from utils.reach_set_functions import compute_hull
from functools import partial
from utils.read_hdf5 import extract_dataset_from_results
from utils.trim_aircraft import compute_trim_eom
import time
import pandas as pd

from utils.gramian_metrics import *

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

def volume_obj(x, flight_point):
    ''' This objective computes the negative of the volume of the p-norm reach set for design variable x. Negative so we
    can maximize it.'''

    hull = set_hull(x,flight_point)
    # hull = set_rate_hull(x, flight_point)
    print(f"Design variables: {x}")
    vol = hull.volume
    print(f"Volume of convex hull: {vol}")
    return -vol  # Negative for maximization

def scaled_volume_obj(x, flight_point):
    ''' This objective just computes volume_obj, but scales it up to make optimization easier'''
    hull_vol = volume_obj(x, flight_point)
    return hull_vol * 1e2

def lift_drag(x, desired_vel):
    '''Computes lift over drag value from interpolator.'''
    data_file = 'Aero_data/aero_7dim2.npz'
    interp_dict = aero_interps(data_file)
    chs = x[0]
    whs = x[1]
    alpha0, deltae0, deltaT0, pitch0 = compute_trim_eom(
        np.asarray([desired_vel, chs, whs]), data_file, climb_angle=0
    )
    flight_point = np.array([desired_vel, alpha0, deltae0])
    full_flight_point = np.concatenate((flight_point[0:2], x, flight_point[2:]))

    lift = interp_dict['CL'](full_flight_point)
    drag = interp_dict['Cdi'](full_flight_point)
    print(f'lift:{lift}')
    print(f'drag:{drag}')
    c = -lift/drag
    print(c)
    return c[0]

def set_hull(x,desired_vel, already_trimmed = False, trim_data = None,save_set = False):
    ''' This function computes the A,B matrices of the system, computes the p-norm reach set, and returns the convex hull of the points.
    It will also save the data from the p-set computation into Outputs/reach_data.csv if desired

    Input: x: array of design variables, where x[0] = center halfspan of bwb, x[1] = wing halfspan of bwb, 
    flight_point is the trim [V, pitch, er]. We will add chs and whs into this array. 
    But maybe now flight_point is just going to become velocity.
    
    Output: reachable set hull: scipy.spatial.ConvexHull object'''

    if already_trimmed:
        tp, flight_point, data_file = trim_data
    else:
        tp, flight_point, data_file, _cost = design_trim(x,desired_vel)
    deltaT0 = tp[2]
    deltae0 = tp[1]
    alpha0 = tp[0]
    pitch0 = tp[3]

    interp_dict = aero_interps(data_file)

    # print(f'Trimmed alpha: {np.rad2deg(alpha0)}, deltae: {np.rad2deg(deltae0)}, deltaT: {deltaT0}, pitch: {np.rad2deg(pitch0)}')

    # Compute input bounds
    thrust_ub = 1-deltaT0
    thrust_lb = -deltaT0
    elev_ub = np.deg2rad(30) - deltae0
    elev_lb = np.deg2rad(-30) - deltae0

    u_upper = np.array([thrust_ub, elev_ub])
    u_lower = np.array([thrust_lb, elev_lb])

    # flight_point = np.array([desired_vel, alpha0, deltae0])

    full_flight_point = np.concatenate((np.array([desired_vel, alpha0]), x[0:2], np.array([deltae0]),x[2:]))
    A,B = build_AB(interp_dict,full_flight_point)

    # If we want to see the A,B matrices
    # print(f'A: {A}')
    # print(f'B: {B}')

    tf = 1

    N_res = 6     # Resolution for the costate sampling

    # With this, actually compute the set
    control_cost, final_positions = compute_inf_reach_set_sphere(A,B,tf,N_res,u_ub = u_upper, u_lb = u_lower, \
                                                                 save_data=save_set,filename='Output/bwb_inf_data.csv')

    # Next, take the convex hull of the points in final_positions, computing inclusion based on control cost.
    hull = compute_hull(final_positions)

    return hull


def compute_axis_extent(x,v,desired_vel, already_trimmed = False, trim_data = None):
    ''' This constraint computes the length of the p-norm reach set projected along a given direction.
    Initial length along [0,0,0,1] is...
    
    Going to also try checking along [0,1,0,0]...'''
    hull = set_hull(x,desired_vel, already_trimmed = already_trimmed, trim_data = trim_data)

    # Define axis for projection
    axis = v
    # ensure unit vector
    axis = axis/np.linalg.norm(axis)

    hull_points = hull.points[hull.vertices]
    projections = hull_points @ axis

    # Compute min and max position of projections along axis
    proj_min, proj_max = projections.min(), projections.max()
    
    # Compute the actual vectors in case of plot
    line_points = np.concatenate((proj_min * axis, proj_max * axis),axis=1) # Saves min and max as [x_min, x_max]
    extent = proj_max - proj_min
    print(f"Design variables: {x}")
    print(f"Extent along direction {v}: {extent}")
    return extent

def direction_constraint(x,v,extent0,desired_vel):
    ''' This is a wrapper for compute_axis_extent to be used as a constraint in the optimization.
    Projects onto fourth axis [0,0,0,1]'''
    extent = compute_axis_extent(x,v,desired_vel)
    con = extent - 1.5 * extent0
    print(f'Direction Constraint: {con}')
    return np.array([con]) # Want to increase extent by at least 50%.

def direction_obj(x,v,desired_vel):
    print(f"Design variables: {x}")
    extent = compute_axis_extent(x,v,desired_vel)
    return -(extent) # Want to increase extent.

def design_trim(x,desired_vel,climb_angle=0, give_cost = False):
    chs = x[0]
    whs = x[1]
    if x.size < 3:
        data_file = 'Aero_data/aero_7dim2.npz'
        alpha0, deltae0, deltaT0, pitch0, cost = compute_trim_eom(np.asarray([desired_vel,chs,whs]), data_file,climb_angle=climb_angle)
    else:
        data_file = 'Aero_data/aero_merged_cg_17.npz'     # MAKE SURE TO PASS THE CORRECT NUMBER OF FLIGHT_POINTS TO compute_trim_eom WHEN USING THIS
        wsl = x[2]
        thsl = x[3]
        wdl = x[4]
        
        alpha0, deltae0, deltaT0, pitch0, cost = compute_trim_eom(np.asarray([desired_vel,chs,whs,wsl,thsl,wdl]), data_file,climb_angle=climb_angle)

    flight_point = np.concatenate((np.array([desired_vel, alpha0]), x[0:2], np.array([deltae0]),x[2:]))

    return np.array([alpha0, deltae0, deltaT0, pitch0]), flight_point, data_file, cost

def static_stability(x,desired_vel, already_trimmed = False, trim_data = None):
    if already_trimmed:
        tp, flight_point, data_file = trim_data
    else:
        tp, flight_point, data_file, _cost = design_trim(x,desired_vel)
    alpha0 = tp[0]

    interp_dict = aero_interps(data_file)
    M_alpha = interp_dict['malpha'](flight_point)
    Z_alpha = interp_dict['zalpha'](flight_point)
    # stat_margin = M_alpha/(Z_alpha * np.cos(alpha0))
    stat_margin = M_alpha/(Z_alpha)
    print(f'Static Margin:{stat_margin}')
    return stat_margin

def collective_constraints(x,desired_vel,v = None, extent0 = None,extent_bool = True, stab_bool = True, trim_bool = True, desired_sm = 0):

    cons = np.array([])
    tp, flight_point, data_file, cost = design_trim(x,desired_vel,give_cost = trim_bool)

    if extent_bool:
        extent = compute_axis_extent(x,v,desired_vel, already_trimmed = True, trim_data = (tp, flight_point, data_file))
        cons = np.block([cons, extent - 1.275 * extent0])
    if stab_bool:
        stability = static_stability(x,desired_vel,already_trimmed = True, trim_data = (tp, flight_point, data_file))
        cons = np.block([cons, stability.item() - desired_sm])
    if trim_bool:
        thrust_force = 96000 # lb
        maxT = thrust_force * 4.44822 * 1.5 # convert to N
        weight = 331522 # lb
        weight = weight * 4.44822 # convert to N
        mass = weight / 9.81  # kg
        g = 9.81

        interp_dict = aero_interps(data_file)
        M = interp_dict['M']
        Di = interp_dict['Di']
        L = interp_dict['L']

        deltaT = tp[2]
        alpha = tp[0]
        theta = tp[3]
        # full_flight_point = np.concatenate((flight_point[0:1],tp[0:1],flight_point[1:3],tp[1:2],flight_point[3:]))
        full_flight_point = flight_point
        Ft = deltaT * maxT

        # Velocity constraint
        cons = np.block([ Ft/mass * np.cos(alpha) - Di(full_flight_point)/mass - g * np.sin(theta - alpha), cons])

        # AOA constraint
        cons = np.block([(-Ft/(mass * V0) * np.sin(alpha) - L(full_flight_point)/(mass * V0) + g/V0 * np.cos(theta - alpha) ) * 1e3, cons])

        # Q constraint
        cons = np.block([M(full_flight_point) * 1e-5, cons])

        # Gamma constraint
        climb_angle = 0
        # cons = np.block([theta - alpha - np.deg2rad(climb_angle), cons])
    print(cons)

    return cons

if __name__ == '__main__':
    '''Now, the design variable is only going to be center-halfspan and wing-halfspan.
    we can use partial to pass the other arguments to building A,B.
    At each iteration, we will build A,B, compute the Lp norm reach set, compute its volume, and then optimize.

    flight_point will contain [alt, V, pitch, er], the full one should contain [alt, V, pitch, CHS, WHS, er].
    chs and whs were separated out because they are the design variables that we will alter, the others will remain fixed for now.
    '''

    # This dictates if we have additional decision variables or not.

    tb = False
    sb = True
    angle = 115
    stat_margin = 0

    chs = 5 # m
    whs = 12 # m
    ws = 35
    ths = 4.9
    wd = 10
    x0 = np.array([chs,whs,ws,ths,wd])

    V0 = 200 # m/s

    flight_params = V0
    
    # Axis on which to project reachable set during some optimization problems
    v = np.array([[0,0,np.cos(np.deg2rad(angle)),np.sin(np.deg2rad(angle))]]).T

    init_extent = compute_axis_extent(x0,v,flight_params)

    ''' Partial functions for objective and constraint functions '''
    fixed_scaled_volume_obj = partial(scaled_volume_obj,flight_point=V0)
    fixed_direction_obj = partial(direction_obj,v=v,desired_vel=V0)
    fixed_stab_con = partial(collective_constraints,desired_vel = V0, extent_bool = False, stab_bool = sb,trim_bool = tb, desired_sm = stat_margin)
    fixed_direction_con = partial(direction_constraint,v=v,extent0=init_extent, desired_vel=V0)
    fixed_direction_stab_con = partial(collective_constraints,v=v,extent0=init_extent,desired_vel=V0, extent_bool = True, stab_bool = sb, trim_bool = tb, desired_sm = stat_margin)
    fixed_gram_direction_obj = partial(gramian_direction_obj, v=v, desired_vel = V0 )
    fixed_gram_eig_obj = partial(gramian_eig_obj, desired_vel=V0, eigs=np.array([2,3]) )

    # Set up optimization
    num_eqcon = 3 if tb else 0  # Number of equality constraints. We have 4 from trim. 3 if not using climb angle.
    x_low = np.array([3, 10, 15, 3, 10])
    x_upp = np.array([7, 20, 50, 8, 12])

    opt_version = input("Which optimization problem would you like to run? Choose from: VM, DM, VMDC, or Gramian (VM = volume maximization, DM = direction maximization, VMDC = volume maximization with direction constraint, Gramian = maximizing eigenvalues as in Gupta et al.\n")
    opt_version = opt_version.upper()

    if opt_version == "VM":
        '''This will purely maximize the volume of reach set.'''
        results = optimize(x0, obj=fixed_scaled_volume_obj, meq = num_eqcon, con=fixed_stab_con, xl=x_low, xu = x_upp, \
                           visualize = True, save_figname="Output/optimization_progress.png", summary_filename="Output/design_optimization.out", \
                            save_itr = 'major', save_filename = 'Output/opt_vars.hdf5')
        print_dict_as_table(results)

    elif opt_version == 'DM':
        '''Optimize some direction w/o direction constraint'''
        results = optimize(x0, obj=fixed_direction_obj, meq = num_eqcon, con=fixed_stab_con, xl=x_low, xu = x_upp, \
                            visualize = True, save_figname="Output/optimization_progress.png", summary_filename="Output/design_optimization.out", \
                                save_itr = 'major', save_filename = 'Output/opt_vars.hdf5') 
        print_dict_as_table(results)

    elif opt_version == 'VMDC':
        '''Optimized for volume with direction constraint'''
        results = optimize(x0, obj=fixed_scaled_volume_obj, meq = num_eqcon, con=fixed_direction_stab_con, xl=x_low, xu = x_upp, \
                            visualize = True, save_figname="Output/optimization_progress.png", summary_filename="Output/design_optimization.out", \
                                save_itr = 'major', save_filename = 'Output/opt_vars.hdf5', finite_diff_abs_step=np.array([1e-3, 1e-3, 1e-3, 1e-3, 1e-3]), \
                                    acc = 1e-4)
        print_dict_as_table(results)

    elif opt_version == 'Gramian':
        '''Optimize some eigenvalue of the reachability Gramian'''
        results = optimize(x0, obj=fixed_gram_eig_obj, meq = num_eqcon, con=fixed_stab_con, xl=x_low, xu = x_upp, \
                            visualize = True, save_figname="Output/optimization_progress.png", summary_filename="Output/design_optimization.out", \
                                save_itr = 'major', save_filename = 'Output/opt_vars.hdf5') 
        print_dict_as_table(results)
    else:
        print('Choose from one of the listed options.')

    # output_hdf5_file = "Output/opt_vars.hdf5"
    # x0 = extract_dataset_from_results(output_hdf5_file, "x")
    # hull = set_hull(x0,V0,save_set = True)