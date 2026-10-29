import numpy as np
from utils.reach_set_functions import compute_hull, plot_hull_projection
from utils.reach_set_data_read import unpack_set_data
import matplotlib.pyplot as plt

''' This script is for processing some of the data and making some plots'''

def plot_the_sets(opt_path,init_csv_file,saveit,projection_line=True, scatteron=False):

    N_res = 6
    plt.rc('xtick', labelsize=14)
    plt.rc('ytick', labelsize=14)

    model_type = opt_path.split('/')[-1][3:]
    print(f'Showing {model_type}')

    angle = np.deg2rad(90)
    v = np.array([[0,0,np.cos(angle),np.sin(angle)]]).T
    # v = np.array([[np.cos(np.deg2rad(angle)),np.sin(np.deg2rad(angle)),0,0]]).T

    opt_csv_file = f'{opt_path}/bwb_inf_data.csv'

    l10, l20, l30, l40, fp0 = unpack_set_data(init_csv_file,N_res)
    l11, l21, l31, l41, fp1 = unpack_set_data(opt_csv_file,N_res)

    hull0 = compute_hull(fp0.reshape(-1,4))
    hull1 = compute_hull(fp1.reshape(-1,4))
    # hull2 = compute_hull(fp2)

    hull0v = hull0.volume
    hull1v = hull1.volume
    diff_volume = hull1v - hull0v

    hull_points0 = hull0.points[hull0.vertices]
    projections0 = hull_points0 @ v

    # Compute min and max position of projections along v
    proj_min, proj_max = projections0.min(), projections0.max()
    
    # Compute the actual vectors in case of plot
    extent0 = proj_max - proj_min

    hull_points1 = hull1.points[hull1.vertices]
    projections1 = hull_points1 @ v
    proj_min, proj_max = projections1.min(), projections1.max()
    extent1 = proj_max - proj_min
    print(f"Extent along direction {v} changes from: {extent0} to {extent1}")
    print(f'Percent increase: {100*(extent1-extent0)/extent0}')

    print(f"Volume of Reachable Set: {hull0v} to {hull1v}")
    print(f"Percent increase: {100*diff_volume/hull0v} %")

    states = ["Airspeed (m/s)","Angle of Attack (rad)","Pitch Rate (rad/s)","Pitch Angle (rad)"]
    states_abbrev = ["speed","aoa","pitch_rate","pitch"]
    fsize = 18

    col1 = 2
    col2 = 3

    plt.figure(figsize=(8,6))
    plot_hull_projection(hull0,col_x=col1,col_y=col2,hull_plot_label="Initial Set",point_label="Initial Interior",hull_color="b", ls='dashed',points=scatteron)
    plot_hull_projection(hull1,col_x=col1,col_y=col2,hull_plot_label="Optimized Set",point_label="Optimized Interior",hull_color="r", ls='solid',points=scatteron)

    plt.xlabel(states[col1],fontsize=fsize)
    plt.ylabel(states[col2],fontsize=fsize)
    # plt.title(f"Projection of Reachable Set onto {states[col1]} and {states[col2]}",fontsize=fsize)

    if projection_line:
        hull = hull1
        
        # Choose projection axis
        axis = v
        # axis = np.asarray([[0,0,0.5,-0.8]]).T
        axis = axis/np.linalg.norm(axis)

        # Get the vertices of convex hull and project onto axis
        hull_points = hull.points[hull.vertices]
        projections = hull_points @ axis

        # Compute min and max position of projections along axis
        proj_min, proj_max = projections.min(), projections.max()

        # Compute the actual vectors in case of plot
        line_points = np.concatenate((proj_min * axis, proj_max * axis),axis=1) # Saves min and max as [[x_min, y_min],[x_max, y_max]]
        line_points = line_points.T
        plt.plot(line_points[:,col1], line_points[:,col2], 'r', lw=5, label='Optimized Set Projection')

        '''Do same for hull0'''
        hull = hull0
        hull_points = hull.points[hull.vertices]
        projections = hull_points @ axis

        # Compute min and max position of projections along axis
        proj_min, proj_max = projections.min(), projections.max()

        # Compute the actual vectors in case of plot
        line_points = np.concatenate((proj_min * axis, proj_max * axis),axis=1) # Saves min and max as [[x_min, y_min],[x_max, y_max]]
        line_points = line_points.T
        plt.plot(line_points[:,col1], line_points[:,col2], 'b', linestyle='dashed', lw=3, label='Initial Set Projection')

    if col1 == 2:
        plt.legend(fontsize=fsize*0.8,loc='upper left')

    plt.grid(True)
    plt.axis('auto')
    if saveit:
        plt.axis('equal')
        plt.savefig(f"{opt_path}/{model_type}_reach_proj_{states_abbrev[col1]}_{states_abbrev[col2]}.png",dpi=300)
        path_to_paper = '/Users/stevennguyen/Library/CloudStorage/OneDrive-Personal/UCSD/Research/AIAA 2026/Reachability Paper/Figures/Reach Sets'
        plt.savefig(f'{path_to_paper}/{model_type}_reach_proj_{states_abbrev[col1]}_{states_abbrev[col2]}.png',dpi=300)

    else:
        plt.axis('equal')
        plt.show()

    col1 = 0
    col2 = 1

    plt.figure(figsize=(8,6))
    plot_hull_projection(hull0,col_x=col1,col_y=col2,hull_plot_label="Initial Set",point_label="Initial Interior",hull_color="b", ls='dashed', points=scatteron)
    plot_hull_projection(hull1,col_x=col1,col_y=col2,hull_plot_label="Optimized Set",point_label="Optimized Interior",hull_color="r", ls='solid', points=scatteron)

    plt.xlabel(states[col1],fontsize=fsize)
    plt.ylabel(states[col2],fontsize=fsize)

    if projection_line:
        hull = hull1
        
        # Choose projection axis
        axis = v
        axis = axis/np.linalg.norm(axis)

        # Get the vertices of convex hull and project onto axis
        hull_points = hull.points[hull.vertices]
        projections = hull_points @ axis

        # Compute min and max position of projections along axis
        proj_min, proj_max = projections.min(), projections.max()

        # Compute the actual vectors in case of plot
        line_points = np.concatenate((proj_min * axis, proj_max * axis),axis=1) # Saves min and max as [[x_min, y_min],[x_max, y_max]]
        line_points = line_points.T
        plt.plot(line_points[:,col1], line_points[:,col2], 'r', lw=5, label='Optimized Set Projection')

        '''Do same for hull0'''
        hull = hull0
        hull_points = hull.points[hull.vertices]
        projections = hull_points @ axis

        # Compute min and max position of projections along axis
        proj_min, proj_max = projections.min(), projections.max()

        # Compute the actual vectors in case of plot
        line_points = np.concatenate((proj_min * axis, proj_max * axis),axis=1) # Saves min and max as [[x_min, y_min],[x_max, y_max]]
        line_points = line_points.T
        plt.plot(line_points[:,col1], line_points[:,col2], 'b', linestyle='dashed', lw=3, label='Initial Set Projection')

    if col1 == 2:
        plt.legend(fontsize=fsize,location='bottom left')

    plt.grid(True)
    plt.axis('auto')
    if saveit:
        plt.axis('equal')
        plt.savefig(f"{opt_path}/{model_type}_reach_proj_{states_abbrev[col1]}_{states_abbrev[col2]}.png",dpi=300)
        path_to_paper = '/Users/stevennguyen/Library/CloudStorage/OneDrive-Personal/UCSD/Research/AIAA 2026/Reachability Paper/Figures/Reach Sets'
        plt.savefig(f'{path_to_paper}/{model_type}_reach_proj_{states_abbrev[col1]}_{states_abbrev[col2]}.png',dpi=300)

    else:
        plt.axis('equal')
        plt.show()

if __name__ == "__main__":
    # opt_path1 = "Output/5_dec_vars/N6_volmax"
    # opt_path2 = "Output/5_dec_vars/N6_115max"
    # opt_path3 = "Output/5_dec_vars/N6_volmax_115con"

    # models = [opt_path1, opt_path2, opt_path3]

    sm = 0.04
    if sm == 0:
        opt_path1 = 'Output/5_dec_vars/N6_volmax'
        opt_path2 = 'Output/5_dec_vars/N6_115max'
        opt_path3 = 'Output/5_dec_vars/N6_volmax_115con'
        models = [opt_path1, opt_path2, opt_path3]
    elif sm == 0.04:
        opt_path1 = 'Output/5_dec_vars/10-22-26/N6_volmax_sm4'
        opt_path2 = 'Output/5_dec_vars/10-22-26/N6_90max_sm4'
        opt_path3 = 'Output/5_dec_vars/10-29-26/N6_volmax_90con_sm4'
        models = [opt_path1, opt_path2, opt_path3]
    elif sm == 0.02:
        opt_path1 = 'Output/5_dec_vars/10-23-26/N6_volmax_sm2'
        opt_path2 = 'Output/5_dec_vars/10-23-26/N6_115max_sm2'
        opt_path3 = 'Output/5_dec_vars/10-23-26/N6_90max_sm2'
        models = [opt_path1, opt_path2, opt_path3]
    elif sm == 0.01:
        opt_path1 = 'Output/5_dec_vars/10-21-26/N6_volmax_sm1'
        opt_path2 = 'Output/5_dec_vars/10-21-26/N6_115max_sm1'
        models = [opt_path1, opt_path2]
    elif sm == 0.03:
        opt_path1 = 'Output/5_dec_vars/10-19-26/N6_volmax_sm3'
        opt_path2 = 'Output/5_dec_vars/10-19-26/N6_115max_sm3'
        models = [opt_path1, opt_path2]
    elif sm == 0.01:
        opt_path1 = 'Output/5_dec_vars/10-21-26/N6_volmax_sm1'
        opt_path2 = 'Output/5_dec_vars/10-21-26/N6_115max_sm1'
        models = [opt_path1, opt_path2]

    init_file = "Output/5_dec_vars/N6_initial/bwb_inf_data.csv"

    saveit = False
    proj = True
    show_points = False
    for file in models:
        plot_the_sets(file,init_file,saveit,projection_line = proj, scatteron = show_points)