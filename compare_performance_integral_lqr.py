# This code will take the optimized and initial system matrices from the 
# design optimization problem and compare the two linear systems in terms of 
# how well they can complete reference tracking/stabilization tasks. These will
# be evaluated in terms of required control effort and tracking error.

from design_optimization import *
import numpy as np
from scipy.integrate import  trapezoid
import matplotlib.pyplot as plt
from read_hdf5 import extract_dataset_from_results
from ref_trackers import integral_ref_track

#%% Build system matrices, cost matrices, time horizon
def make_integral_plots(opt_path, case):

    # Import Aero Data
    aero_data_file = 'Aero_data/aero_7dim2.npz'
    interp_dict = aero_interps(aero_data_file)

    # Initial design
    x = np.array([5, 12])
    chs = x[0]
    whs = x[1]

    # Flight point
    V0 = 200 # m/s
    alpha0, er0, deltaT0, pitch0 = compute_trim_eom(np.asarray([V0,chs,whs]), aero_data_file,climb_angle=0)
    flight_point = np.array([V0, alpha0, er0])

    Tf = 5

    full_flight_point = np.concatenate((flight_point[0:2], x, flight_point[2:]))
    A,B = build_AB(interp_dict,full_flight_point)
    # Compute input bounds
    thrust_ub = 1-deltaT0
    thrust_lb = -deltaT0
    elev_ub = np.deg2rad(30) - er0
    elev_lb = np.deg2rad(-30) - er0

    u_upper = np.array([thrust_ub, elev_ub])
    u_lower = np.array([thrust_lb, elev_lb])

    t_init, x_init, xref_init, u_init, error_init = integral_ref_track(A,B,case,Tf=Tf,u_lb = u_lower, u_ub = u_upper)

    # Optimized design
    output_hdf5_file = f"{opt_path}/opt_vars.hdf5"
    x = extract_dataset_from_results(output_hdf5_file, "x")
    chs = x[0]
    whs = x[1]
    alpha0, er0, deltaT0, pitch0 = compute_trim_eom(np.asarray([V0,chs,whs]), aero_data_file,climb_angle=0)
    flight_point = np.array([V0, alpha0, er0])

    model_type = opt_path.split('/')[-1][3:]

    full_flight_point = np.concatenate((flight_point[0:2], x, flight_point[2:]))
    A,B = build_AB(interp_dict,full_flight_point)

    # Compute input bounds
    thrust_ub = 1-deltaT0
    thrust_lb = -deltaT0
    elev_ub = np.deg2rad(30) - er0
    elev_lb = np.deg2rad(-30) - er0

    u_upper = np.array([thrust_ub, elev_ub])
    u_lower = np.array([thrust_lb, elev_lb])

    t_opt, x_opt, xref_opt, u_opt, error_opt = integral_ref_track(A,B,case,Tf=Tf,u_lb = u_lower,u_ub = u_upper)

    u_l2_norm_opt = np.sqrt(trapezoid(np.sum(u_opt**2,axis=0),t_opt))
    u_l2_norm_init = np.sqrt(trapezoid(np.sum(u_init**2,axis=0),t_init))

    #%% Plot

    plt.figure(figsize=(9, 6))  # Adjust figure size to accommodate more subplots (width, height)
    plt.rcParams.update({'font.size': 15})
    plt.rcParams["legend.handlelength"] = 3.5
    plt.rcParams['lines.markersize'] = 12

    init_color = 'orange'
    if model_type == 'volmax':
        opt_color = 'red'
    elif model_type == '110max':
        opt_color = 'blue'
    elif model_type == 'volmax_110con':
        opt_color = 'darkgreen'
    ref_color = 'purple'

    plt.subplot(2, 1, 1)
    if case == 'pitch':
        plt.plot(t_opt, x_opt[3, :], label='Optimized System', lw=4, linestyle='solid',color=opt_color)
        plt.plot(t_init,x_init[3,:], label='Initial System', lw=4, linestyle='dashed',color=init_color)
        plt.plot(t_opt, xref_opt[:], linestyle='dotted', label = 'Reference', lw=4,color=ref_color)
        plt.ylabel('Pitch angle [rad]')
    elif case == 'velocity':
        plt.plot(t_opt, x_opt[0, :], label='Optimized System', lw=4, linestyle='solid',color=opt_color)
        plt.plot(t_init,x_init[0,:], label='Initial System', lw=4, linestyle='dashed',color=init_color)
        plt.plot(t_opt, xref_opt[:], label = 'Reference', lw=4, linestyle='dotted',color=ref_color)
        plt.ylabel('Airspeed [m/s]')
    plt.legend(loc='best')  # Set legend location to center right
    plt.grid(True)

    # if show_error:
    #     error_caption = f"Optimal tracking error: {error_opt}, initial tracking error: {error_init}"
    #     plt.figtext(0.5, 0.8, error_caption, wrap=True, horizontalalignment='center', fontsize=15)

    # Plot 5: Control input
    plt.subplot(2, 1, 2)
    plt.plot(t_opt, u_opt[0, :], label='$\delta_{th}$ Optimized', color=opt_color, lw=4, linestyle='dashdot')
    plt.plot(t_opt, u_opt[1, :], label='$\delta_e$ Optimized [rad]', color=opt_color, lw=4, linestyle='solid',marker='o',markevery=0.1)
    plt.plot(t_init, u_init[0, :], label='$\delta_{th}$ Initial', color=init_color, lw=4, linestyle='dashdot')
    plt.plot(t_init, u_init[1, :], label='$\delta_e$ Initial [rad]', color=init_color, lw=4, linestyle='solid',marker='o',markevery=0.1)
    plt.xlabel('Time [s]')
    plt.ylabel('Control')
    plt.legend(loc='center right')  # Set legend location to center right
    plt.grid(True)

    # Adjust layout and show the figure
    plt.tight_layout()

    print(f'Showing {model_type} for case {case}')
    print(f'Optimized tracking error: {error_opt}, initial tracking error: {error_init}, change of: {(error_init-error_opt)/error_init*100:.2f}%')
    print(f"Opt control signal: {u_l2_norm_opt:.4f}, init control signal: {u_l2_norm_init:.4f}, change of: {(u_l2_norm_init-u_l2_norm_opt)/u_l2_norm_init*100:.2f}%")
    plt.show()

if __name__ == "__main__":
    opt_path2 = "Output/aero_7dim_asym/N6_110max"
    opt_path1 = "Output/aero_7dim_asym/N6_volmax"
    opt_path3 = "Output/aero_7dim_asym/N6_volmax_110con"
    models = [opt_path1, opt_path2, opt_path3]

    cases = ['pitch','velocity']

    for file in models:
        for case in cases:
            make_integral_plots(file,case)
