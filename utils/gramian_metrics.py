from scipy import integrate
from scipy.linalg import expm
from utils.interpolatory_model import *
import numpy as np
import csv
import math
from utils.read_hdf5 import extract_dataset_from_results
from utils.trim_aircraft import compute_trim_eom
from utils.reach_set_functions import plot_projected_boundary, sample_ellipsoid_surface

def gramian(x,desired_vel,tf=1):
    # First construct A,B matrices

    print(f"Design variables: {x}")

    chs = x[0]
    whs = x[1]
    if x.size < 3:
        data_file = 'Aero_data/aero_7dim2.npz'
        alpha0, deltae0, deltaT0, pitch0, _ = compute_trim_eom(np.asarray([desired_vel,chs,whs]), data_file,climb_angle=0)
    else:
        data_file = 'Aero_data/aero_merged_cg_17.npz'     # MAKE SURE TO PASS THE CORRECT NUMBER OF FLIGHT_POINTS TO compute_trim_eom WHEN USING THIS
        wsl = x[2]
        thsl = x[3]
        wdl = x[4]
        
        alpha0, deltae0, deltaT0, pitch0, _ = compute_trim_eom(np.asarray([desired_vel,chs,whs,wsl,thsl,wdl]), data_file,climb_angle=0)

    interp_dict = aero_interps(data_file)

    # print(f'Trimmed alpha: {np.rad2deg(alpha0)}, deltae: {np.rad2deg(deltae0)}, deltaT: {deltaT0}, pitch: {np.rad2deg(pitch0)}')

    # Compute input bounds
    thrust_ub = 1-deltaT0
    thrust_lb = -deltaT0
    elev_ub = np.deg2rad(30) - deltae0
    elev_lb = np.deg2rad(-30) - deltae0

    u_upper = np.array([thrust_ub, elev_ub])
    u_lower = np.array([thrust_lb, elev_lb])

    flight_point = np.array([desired_vel, alpha0, deltae0])

    full_flight_point = np.concatenate((flight_point[0:2], x[0:2], flight_point[2:],x[2:]))
    A,B = build_AB(interp_dict,full_flight_point)
    # print(f'A: {A}')
    # print(f'B: {B}')
    # Use A, B matrices to compute the reach gramian
    def gram_integrand(t,A,B,tf):
        return expm(A * (tf-t)) @ B @ np.transpose(B) @ expm(np.transpose(A) * (tf-t))
    gram, err = integrate.quad_vec(gram_integrand, 0, tf, args=(A,B,tf)) # Integrate the gramian integrand from 0 to 1
    return gram

def gramian_hull(x,desired_vel, tf=1,seed=50,savepts=False,filename="Output/gramian_set.csv"):
    gram = gramian(x,desired_vel,tf)
    myrng = np.random.default_rng(seed)
    X = sample_ellipsoid_surface(np.linalg.inv(gram),gamma=1,N=1000,rng=myrng,savepts=savepts,filename=filename) # This pulls the convexhull associated with sampled reachableset for L2 norm.
    return X


def gramian_eig_obj(x,desired_vel,eigs):
    '''This objective will maximize the eigenvalues of reach gramian.
    Input eigs gives a np array of the eigenvalues we care about.'''
    gram = gramian(x,desired_vel,tf=1)

    # eigenvalues are:
    vals, vecs = np.linalg.eig(gram)

    print(f'Eigenvalues: {vals}')
    print(f'Eigenvectors: {vecs}')
    # If we assume that the first 2 eigenvalues should be minimized and we ignore the last 2.
    # This is because the last two eigenvectors align most with the pitch angle axis.
    print(f'The sum: {np.sum(vals[eigs]) - np.sum(np.delete(vals,eigs))}')

    return -np.sum(vals[eigs]) + np.sum(np.delete(vals,eigs))

def gramian_extent(x,v,desired_vel):
    hull = gramian_hull(x,desired_vel,tf=1)
    axis = v/np.linalg.norm(v)
    hull_points = hull.points[hull.vertices]
    projections = hull_points @ axis

    # Compute min and max position of projections along axis
    proj_min, proj_max = projections.min(), projections.max()
    
    # Compute the actual vectors in case of plot
    line_points = np.concatenate((proj_min * axis, proj_max * axis),axis=1) # Saves min and max as [x_min, x_max]
    extent = proj_max - proj_min
    print(f"Extent along direction {axis}: {extent}")
    return extent

def gramian_direction_obj(x,v,fp):
    print(f'Design variables: {x}')
    extent = gramian_extent(x,v,fp)
    return -extent