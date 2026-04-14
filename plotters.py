import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import trapezoid

def niceplot_one_at_a_time(sim_dictionary,model,save=False):
    ''' This function makes a nicer looking plot for the simulation trajectories
    between the initial and an optimized model that is fed in.
    No error bar plots since we only look at one model at a time.'''
    plt.figure(figsize=(12,12))
    plt.rcParams.update({'font.size': 15})
    plt.rcParams['lines.linewidth'] = 4
    plt.rcParams["legend.handlelength"] = 3.5
    plt.rcParams['lines.markersize'] = 12

    t_init = sim_dictionary['initial']['t']
    x_initial = sim_dictionary['initial']['x']
    u_init = sim_dictionary['initial']['u']
    tp_init = sim_dictionary['initial']['tp']
    style_init = sim_dictionary['initial']['style']
    color_init = sim_dictionary['initial']['color']
    
    # t_ref = sim_dictionary['ref']['t']
    # x_ref = sim_dictionary['ref']['x']
    # style_ref = sim_dictionary['ref']['style']
    # color_ref = sim_dictionary['ref']['color']

    t_opt = sim_dictionary[model]['t']
    x_opt = sim_dictionary[model]['x']
    u_opt = sim_dictionary[model]['u']
    tp_opt = sim_dictionary[model]['tp']
    style_opt = sim_dictionary[model]['style']
    color_opt = sim_dictionary[model]['color']

    name = sim_dictionary[model]['name']

    x_opt[0:4,:] = x_opt[0:4,:] - tp_opt
    x_init = x_initial.copy()[0:4,:] - tp_init

    # Plot 1: Airspeed
    plt.subplot(5, 1, 1)
    plt.plot(t_init, x_init[0, :], label='Initial', color=color_init, linestyle=style_init)
    plt.plot(t_opt, x_opt[0, :], label=name.upper(), color = color_opt, linestyle=style_opt)
    # plt.plot(t_ref, x_ref[0,:], label='Reference', linestyle=style_ref, color=color_ref)
    plt.legend(loc='best')  # Set legend location to center right
    plt.grid(True)
    plt.ylabel('Perturbed Airspeed\n [m/s] ($V - V_0$)')

    # Plot 2: Angle of Attack
    plt.subplot(5,1,2)
    plt.plot(t_init, np.rad2deg(x_init[1, :]), label='Initial', color=color_init, linestyle=style_init)
    plt.plot(t_opt,   np.rad2deg(x_opt[1, :]), label=name.upper(), color = color_opt, linestyle=style_opt)
    # plt.plot(t_ref,  np.rad2deg(x_ref[1,:]), label='Reference', linestyle=style_ref, color=color_ref)
    plt.grid(True)
    plt.ylabel('Perturbed Angle of\n Attack [deg]\n ($\\alpha - \\alpha_0$)')

    # Plot 3: Pitch Rate
    plt.subplot(5, 1, 3)
    plt.plot(t_init, np.rad2deg(x_init[2, :]), label='Initial', color=color_init, linestyle=style_init)
    plt.plot(t_opt,   np.rad2deg(x_opt[2, :]), label=name.upper(), color = color_opt, linestyle=style_opt)
    # plt.plot(t_ref,  np.rad2deg(x_ref[2,:]), label='Reference', linestyle=style_ref, color=color_ref)
    plt.grid(True)
    plt.ylabel('Perturbed Pitch Rate\n [deg/s] ($\dot{\\theta} - \dot{\\theta}_0$)')

    # Plot 4: Pitch angle
    plt.subplot(5, 1, 4)
    plt.plot(t_init, np.rad2deg(x_init[3, :]), label='Initial', color=color_init, linestyle=style_init)
    plt.plot(t_opt,   np.rad2deg(x_opt[3, :]), label=name.upper(), color = color_opt, linestyle=style_opt)
    # plt.plot(t_ref,  np.rad2deg(x_ref[3,:]), label='Reference', linestyle=style_ref, color=color_ref)
    plt.ylabel('Perturbed Pitch angle\n [deg] ($\\theta - \\theta_0$)')
    plt.grid(True)

    # Plot 5: Control input
    # only one input...
    plt.subplot(5, 1, 5)
    plt.plot(t_init, u_init[0, :], label='$\delta_{th}$ Initial', color=color_init, linestyle='dashdot'         )#,marker='^',markevery=0.1)
    plt.plot(t_init, u_init[1, :], label='$\delta_e$ Initial [rad]', color=color_init, linestyle='solid',marker='o',markevery=0.1  )#,marker='o',markevery=0.1)
    plt.plot(t_opt, u_opt[0, :], label='$\delta_{th}$ Optimized', color=color_opt, linestyle='dashdot'          )#,marker='^',markevery=0.1)
    plt.plot(t_opt, u_opt[1, :], label='$\delta_e$ Optimized [rad]', color=color_opt, linestyle='solid',marker='o',markevery=0.1   )#,marker='o',markevery=0.1)
    plt.xlabel('time [s]')
    plt.ylabel('Control', wrap=True)
    plt.legend(loc='center right')  # Set legend location to center right
    plt.grid(True)

    # Adjust layout and show the figure
    plt.tight_layout()

    if save:
        figname = f'{model}_nonlinear_track'
        plt.savefig(f'Output/aero_7dim_asym/Nonlinear_Pitch_Integrator/{figname}.png', dpi=300)
        path_to_paper = '/Users/stevennguyen/Library/CloudStorage/OneDrive-Personal/UCSD/Research/AIAA 2026/Reachability Paper/Figures/Tracking'
        plt.savefig(f'{path_to_paper}/{figname}')
    else:
        plt.show()


def error_bars(sim_dictionary,save=False):
    ''' This function makes error bars to compare the models'''

    plt.figure(figsize=(8,6))
    plt.rcParams.update({'font.size': 15})
    plt.rcParams['lines.linewidth'] = 4
    plt.rcParams['lines.markersize'] = 12

    # Error computation for all trajectories
    print("\n" + "="*70)
    print("TRACKING ERROR ANALYSIS")
    print("="*70)
    
    for model, data in sim_dictionary.items():
        if model == 'ref':
            continue
        t = data['t']
        x = data['x']
        tp = data['tp']
        u = data['u']
        
        # Compute errors
        error = x[0:4, :] - tp
        vel_err =   np.sqrt(trapezoid(error[0, :]**2, t))
        aoa_err =   np.sqrt(trapezoid(error[1, :]**2, t))
        pr_err =    np.sqrt(trapezoid(error[2, :]**2, t))
        pitch_err = np.sqrt(trapezoid(error[3, :]**2, t))
        error_norm = np.sqrt(trapezoid(np.sum(error**2, 0), t))
        u_norm = np.sqrt(trapezoid(np.sum(u**2,axis=0),t))

        data['error'] = {
                'vel_err': vel_err,
                'aoa_err': aoa_err,
                'pr_err': pr_err,
                'pitch_err': pitch_err,
                'error_norm': error_norm,
                'u_norm':u_norm
        }
        print(f"\n{model.upper()} Model:")
        print(f"  Total error (L2 norm): {error_norm:.6f}")
        print(f"  Velocity error:        {vel_err:.6f}")
        print(f"  AOA error:             {aoa_err:.6f}")
        print(f"  Pitch rate error:      {pr_err:.6f}")
        print(f"  Pitch angle error:     {pitch_err:.6f}")
        print(f"  Input L2 norm:         {u_norm:.6f}")

        if model == 'initial':
            init_vel = vel_err
            init_aoa = aoa_err
            init_pr = pr_err
            init_pitch = pitch_err
            init_total = error_norm
            init_u = u_norm
            continue   # This skips the rest of the for loop and starts again.

        vel_rel_err = (init_vel - vel_err)/init_vel
        aoa_rel_err = (init_aoa - aoa_err)/init_aoa
        pr_rel_err = (init_pr - pr_err)/init_pr
        pitch_rel_err = (init_pitch - pitch_err)/init_pitch
        total_rel_err = (init_total - error_norm)/init_total
        u_rel_norm = (init_u - u_norm)/init_u

        print(f"  Velocity relative error:        {vel_rel_err*100:.2f}")
        print(f"  AOA relative error:             {aoa_rel_err*100:.2f}")
        print(f"  Pitch rate relative error:      {pr_rel_err*100:.2f}")
        print(f"  Pitch angle relative error:     {pitch_rel_err*100:.2f}")
        print(f"  Input relative L2 norm:         {u_rel_norm*100:.2f}")


        data['error']['relative'] = {
            'vel':vel_rel_err     * 100 ,
            'aoa':aoa_rel_err     * 100 ,
            'pr':pr_rel_err       * 100 ,
            'pitch':pitch_rel_err * 100 ,
            'total':total_rel_err * 100 ,
            'u':u_rel_norm
        }

    print("="*70 + "\n")
    
    models_list_label = ['vm','dm','vmdc']
    models_list_actual = [key for key in sim_dictionary.keys() if key not in ['initial', 'ref']]
    error_types = ['vel', 'aoa', 'pr', 'pitch', 'u']
    error_labels = ['Velocity', 'AOA', 'Pitch Rate', 'Pitch Angle', 'Control']
    
    # Define bar patterns/hatches
    hatches = ['//', '\\', 'o', '*', 'xxx']
    
    # Set up bar positions
    x_pos = np.arange(len(models_list_actual))
    bar_width = 0.15
    
    # Plot bars for each error type
    for i, (error_type, error_label, hatch) in enumerate(zip(error_types, error_labels, hatches)):
        offset = (i - 2) * bar_width
        values = [sim_dictionary[model]['error']['relative'][error_type] for model in models_list_actual]
        plt.bar(x_pos + offset, values, bar_width, label=error_label, hatch=hatch, edgecolor='black', linewidth=1.5)
    
    # Customize the plot
    plt.xlabel('Model Type')
    plt.ylabel('Decrease in Error (%)')
    # plt.title('Tracking Error Comparison Across Models')
    plt.xticks(x_pos, [m.upper() for m in models_list_label])
    plt.legend(loc='best')
    plt.grid(True, axis='y')
    if save:
        figname = 'error_nonlinear'
        plt.savefig(f'Output/aero_7dim_asym/Nonlinear_Pitch_Integrator/{figname}.png', dpi=300)
        path_to_paper = '/Users/stevennguyen/Library/CloudStorage/OneDrive-Personal/UCSD/Research/AIAA 2026/Reachability Paper/Figures/Tracking'
        plt.savefig(f'{path_to_paper}/{figname}')
    else:
        plt.show()