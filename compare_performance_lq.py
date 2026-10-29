# This code will take the optimized and initial system matrices from the 
# design optimization problem and compare the two linear systems in terms of 
# how well they can complete reference tracking/stabilization tasks. These will
# be evaluated in terms of required control effort and tracking error.

from design_optimization import *
import numpy as np
from scipy.integrate import trapezoid
from utils.read_hdf5 import extract_dataset_from_results
from utils.ref_trackers import lqr_ref_track
from utils.plotters import linear_sim_plots

#%% Build system matrices, cost matrices, time horizon

def simulate_lq(design, case, show_error, do_meas_noise=False, do_proc_noise=False,more_data = False):
    if more_data:
        aero_data_file = 'Aero_data/aero_merged_cg_17.npz'
    else:
        aero_data_file = 'Aero_data/aero_7dim2.npz'
    interp_dict = aero_interps(aero_data_file)

    # # Flight point
    V0 = 200 # m/s
    Tf = 5

    if more_data:
        chs, whs, ws, ths, wd = design
        alpha0, er0, deltaT0, pitch0, _ = compute_trim_eom(np.asarray([V0,chs,whs,ws,ths,wd]), aero_data_file,climb_angle=0)
    else:
        chs, whs = design
        alpha0, er0, deltaT0, pitch0, _ = compute_trim_eom(np.asarray([V0,chs,whs]), aero_data_file,climb_angle=0)

    flight_point = np.array([V0, alpha0, er0])
    # print(f'Optimized trimmed alpha: {np.rad2deg(alpha0)}, deltae: {np.rad2deg(er0)}, deltaT: {deltaT0}, pitch: {np.rad2deg(pitch0)}')
    trim_point = np.array([V0, alpha0, 0, pitch0])

    # First construct A,B matrices
    full_flight_point = np.concatenate((flight_point[0:2], design[0:2], flight_point[2:],design[2:]))
    A,B = build_AB(interp_dict,full_flight_point)

    # Compute input bounds
    thrust_ub = 1-deltaT0
    thrust_lb = -deltaT0
    elev_ub = np.deg2rad(30) - er0
    elev_lb = np.deg2rad(-30) - er0

    u_upper = np.array([thrust_ub, elev_ub])
    u_lower = np.array([thrust_lb, elev_lb])

    # Simulate the system's LQ tracking
    t_opt, x_opt, xref_opt, u_opt, error_opt = lqr_ref_track(A,B,case,Tf=Tf,u_ub = u_upper,u_lb=u_lower, do_meas_noise=do_meas_noise, do_proc_noise=do_proc_noise)

    return t_opt, x_opt, u_opt, np.tile(trim_point.reshape(-1,1),t_opt.size), error_opt

if __name__ == "__main__":
    

    cases = ['pitch','velocity']
    # cases = ['pitch']
    error = False
    saveit = False
    do_meas_noise = False
    do_proc_noise = False  # This one seems to make for more interesting results. Might need to debug the other case if desired.


    opt_path1 = 'Output/5_dec_vars/N6_volmax'
    opt_path2 = 'Output/5_dec_vars/N6_115max'
    opt_path3 = 'Output/5_dec_vars/N6_volmax_115con'

    models = [opt_path1, opt_path2, opt_path3]

    sim_data = {}
    
    # Initial case
    chs = 5
    whs = 12
    ws = 35
    ths = 4.9
    wd = 10
    design = [chs,whs,ws,ths,wd]

    for case in cases:
        t,x,u,tp, error_init = simulate_lq(design,case,error,do_meas_noise,do_proc_noise,True)

        u_l2_norm_init = np.sqrt(trapezoid(np.sum(u**2,axis=0),t))
        elev_l2_norm_init = np.sqrt(trapezoid(u[1,:]**2,t))
        thrust_l2_norm_init = np.sqrt(trapezoid(u[0,:]**2,t))

        sim_data[f'initial_{case}'] = {
            't': t,
            'x': x,
            'u': u,
            'tp': tp,
            'style': 'dashed',
            'color': 'orange',
            'name': 'initial',
            'ul2':u_l2_norm_init,
            'elev_l2':elev_l2_norm_init,
            'state_l2':error_init
        }

    for file in models:
        for case in cases:
            output_hdf5_file = f"{file}/opt_vars.hdf5"
            design = extract_dataset_from_results(output_hdf5_file, "x")
            model_type = file.split('/')[-1][3:]

            t,x,u,tp, error_opt = simulate_lq(design,case,error,do_meas_noise,do_proc_noise,True)
            u_l2_norm_opt = np.sqrt(trapezoid(np.sum(u**2,axis=0),t))
            elev_l2_norm_opt = np.sqrt(trapezoid(u[1,:]**2,t))
            thrust_l2_norm_opt = np.sqrt(trapezoid(u[0,:]**2,t))

            if model_type == 'volmax':
                key = 'vm'
                color = 'red'
            elif model_type == '115max':
                key = 'dm'
                color = 'blue'
            elif model_type == 'volmax_115con':
                key = 'vmdc'
                color = 'green'
            elif model_type == 'inf_set':
                key = model_type
                color = 'salmon'
            elif model_type == 'gramian_eigs':
                key = model_type
                color = 'teal'

            sim_data[f'{model_type}_{case}'] = {
                't': t,
                'x': x,
                'u': u,
                'tp': tp,
                'style': 'solid',
                'color': color,
                'name':key,
                'ul2': u_l2_norm_opt,
                'elev_l2':elev_l2_norm_opt,
                'state_l2':error_opt
            }
            print('-----------------------------------------------')
            print(f'Showing {model_type} for case {case}')
            error_init = sim_data[f'initial_{case}']['state_l2']
            u_l2_init = sim_data[f'initial_{case}']['ul2']
            elev_l2_init = sim_data[f'initial_{case}']['elev_l2']
            print(f'Optimized tracking error: {error_opt:.4f}, initial tracking error: {error_init:.4f}, change of: {(error_init-error_opt)/error_init*100:.2f}%')
            print(f"Opt control signal: {u_l2_norm_opt:.4f}, init control signal: {u_l2_init:.4f},          change of: {(u_l2_init-u_l2_norm_opt)/u_l2_init*100:.2f}%")
            print(f"Opt elev signal: {elev_l2_norm_opt:.4f}, init: {elev_l2_init:.4f},                            change of: {(elev_l2_init - elev_l2_norm_opt)/elev_l2_init*100:.2f}%")
    for case in cases:
        linear_sim_plots(sim_data,['initial','volmax','115max','volmax_115con'],
                         case,saveit)


