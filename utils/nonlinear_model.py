''' This script will be a nonlinear model from Stevens/Lewis'''
from utils.interpolatory_model import *
import numpy as np
from utils.trim_aircraft import compute_trim_eom
import control as ct
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid
from utils.read_hdf5 import extract_dataset_from_results

# def xref(t):
#     '''This function will return a reference trajectory to be followed.
#     Did sims for reference of 204 m/s, an increase from 200 m/s initial.
#     Works for up to 208 m/s. At 209, some states go out of bound (I think AOA...)'''
#     v_ref = 204  # m/s
#     return v_ref

# def run_nonlinear(fp,data_file,tf):
#     ''' This code will simulate nonlinear steady level flight. However, the input is not coded to actually
#     do any special control. It simply holds the trimmed inputs. Not very useful...
#     Inputs: fp,             [velocity, chs, whs] flight point
#             data_file,      file from which to take aero data interpolators
#             tf,             final simulation time
#     Outputs: t,x,u_hist'''

#     # Using given flight point, compute trimmed angle of attack and inputs.
#     # alpha0, deltae0, deltaT0 = compute_trim_forces(fp, data_file)
#     # pitch0 = alpha0

#     alpha0, deltae0, deltaT0, pitch0 = compute_trim_eom(fp,data_file,climb_angle=0)

#     # this input simply holds the inputs at their trimmed values.
#     u = np.array([deltaT0,deltae0])

#     # Trim point aka initial condition of simulation
#     x0 = np.array([v0, alpha0, 0, pitch0])     # initial state, with integrator state appended

#     # some constants
#     g = 9.81  # m/s^2
#     weight = 331522 # lb
#     weight = weight * 4.44822 # convert to N
#     mass = weight / g  # kg
#     Iy = 3759014 # slug ft^2
#     Iy = Iy * 14.5939 * 0.3048**2 # kg m^2
#     thrust_force = 96000 # lb
#     maxT = thrust_force * 4.44822 * 1.5 # convert to N

#     interp_dict = aero_interps(data_file)
#     M_interp = interp_dict['M']
#     Di_interp = interp_dict['Di']
#     L_interp = interp_dict['L']

#     def controller(t,x):
#         # if t > 2:
#         #     return u + np.array([0.5,0])
#         return u

#     def nonlinear_model(t, x,u):
#         v = x[0]
#         alpha = x[1]
#         q = x[2]
#         theta = x[3]

#         deltae = u[1] # if t < 2 else u[1] + np.deg2rad(2) # add 5 deg elevator deflection after 2 seconds
#         deltaT = u[0] # if t < 2 else u[0] + 0.5

#         flight_point = np.asarray([v, alpha, chs, whs, deltae])

#         M = M_interp(flight_point)[0]
#         Di = Di_interp(flight_point)[0]
#         L = L_interp(flight_point)[0]

#         Ft = deltaT * maxT

#         x_dot = np.zeros_like(x)
#         x_dot[0] = Ft/mass * np.cos(alpha) - Di/mass - g * np.sin(theta - alpha)
#         x_dot[1] = -Ft/(mass * v) * np.sin(alpha) - L/(mass * v) + g/v * np.cos(theta - alpha) + q
#         x_dot[2] = M / Iy
#         x_dot[3] = q

#         return x_dot

#     sol_fwd = solve_ivp(lambda t,x: nonlinear_model(t,x,controller(t,x))  , (0.0, tf), x0, method='BDF')
#     if not sol_fwd.success:
#         raise RuntimeError("Forward integration failed: " + str(sol_fwd.message))

#     t = sol_fwd.t
#     x = sol_fwd.y

#     u_hist = np.zeros((2,len(t)))

#     u_hist = np.asarray([controller(ti,x[:,i]) for i,ti in enumerate(t)])
#     u_hist = u_hist.T

#     return t,x,u_hist

# def run_linear(fp,data_file,tf):
#     ''' Run steady level flight.'''
#     v0 = fp[0]
#     chs = fp[1]
#     whs = fp[2]
#     alpha0, deltae0, deltaT0, pitch0 = compute_trim_eom(fp, data_file,climb_angle=10)
#     full_flight_point = np.array([v0,alpha0,chs,whs,deltae0])
#     trim_point = np.array([v0,alpha0,0,pitch0])
#     trim_inputs = np.array([deltaT0,deltae0])
#     interp_dict = aero_interps(data_file)
#     A,B = build_AB(interp_dict,full_flight_point)

#     def linear_model(t,x,control):
#         x = x-trim_point
#         u = control - trim_inputs
#         x_dot = np.zeros((4,1))
#         x_dot = A@x + B@u
#         return x_dot
    
#     u = np.array([deltaT0,deltae0])
#     sol = solve_ivp(lambda t,x: linear_model(t,x,u)  , (0.0, tf), trim_point)
#     t = sol.t
#     x = sol.y
#     u_hist = np.zeros((2,len(t)))
#     u_hist[0,:] = np.ones(len(t)) * deltaT0
#     u_hist[1,:] = np.ones(len(t)) * np.rad2deg(deltae0)
#     xr_hist = np.zeros((4,len(t)))

#     return t,x,u_hist, xr_hist

# def linear_track(fp,data_file,tf):
#     '''This function will do linearized integral tracking control in velocity.
#     This code is not as robust as the other one for comparing performance, but use this to compare with the nonlinear model.
#     The input bounding is more correct here and we apply the linear model more "corectly" since we treat it as a perturbation
#     model.'''

#     v0 = fp[0]
#     chs = fp[1]
#     whs = fp[2]
#     alpha0, deltae0, deltaT0, pitch0 = compute_trim_eom(fp, data_file,climb_angle=0)
#     full_flight_point = np.array([v0,alpha0,chs,whs,deltae0])
#     trim_point = np.array([v0,alpha0,0,pitch0])
#     print(trim_point)
#     trim_inputs = np.array([deltaT0,deltae0])
#     interp_dict = aero_interps(data_file)
#     A,B = build_AB(interp_dict,full_flight_point)
#     C = np.array([[1,0,0,0]])

#     # Take integral reference tracking code but modify slightly.

#     k = C.shape[0]
#     n = A.shape[0]
#     m = B.shape[1]

#     Abar = np.concatenate((np.concatenate((A,np.zeros((n,k))),axis=1),np.concatenate((C,np.zeros((k,k))),axis=1)),axis=0)
#     Bbar = np.concatenate((B,np.zeros((k,m))),axis=0)

#     Q = np.diag([1,1,1,1,1])
#     R = np.diag([0.1,1000])

#     K = ct.lqr(Abar,Bbar,Q,R)[0]
#     def control_signal(x):
#         x = x.reshape((n+1,1))
#         u = -K@x

#         # In order to make sure we have the right input bounds, we add the value of trimmed inputs
#         # apply the bounds, then remove the trimmed value again to get the perturbed inputs.
        
#         u_actual = u + trim_inputs.reshape(2,-1)
#         u_actual[0] = np.clip(u_actual[0],0,1)
#         u_actual[1] = np.clip(u_actual[1],np.deg2rad(-30),np.deg2rad(30))
#         u = u_actual - trim_inputs.reshape(2,-1)
        
#         return u
    
#     t_eval_fwd = np.linspace(0.0, tf, 1001)

#     def closed_loop_forward(t, x_flat):
#         x = x_flat.reshape((n+1,1))
#         u = control_signal(x)
#         # xr = xref(t) - trim_point[0]
#         xr = 0
#         dx = Abar @ x + Bbar @ u - np.concatenate((np.zeros((4,1)),np.asarray([[xr]])),axis=0)

#         if x[4] > 50:      # prevent wind up, if integrator state too large, keep constant...
#             dx[4] = min(dx[4],0)
#         elif x[4] < -50:
#             dx[4] = max(dx[4],0)
#         return dx.flatten()

#     x0 = np.zeros((n+1,1)).flatten()
#     x0 = x0 + np.array([-10,0,0,0,0])
#     sol_fwd = solve_ivp(closed_loop_forward, (0.0, tf), x0, t_eval=t_eval_fwd)

#     t = sol_fwd.t
#     y = sol_fwd.y
#     x = y[:n,:]
#     # x[0,:] = x[0,:] + np.ones(y.shape[1])*trim_point[0]
#     # x[1,:] = x[1,:] + np.ones(y.shape[1])*trim_point[1]
#     # x[2,:] = x[2,:] + np.ones(y.shape[1])*trim_point[2]
#     # x[3,:] = x[3,:] + np.ones(y.shape[1])*trim_point[3]

#     u_hist = np.zeros((m,t.size))
#     for i, ti in enumerate(t):
#         u = control_signal(y[:,i].reshape((n+1,1)))
#         u_hist[:,i] = u.flatten() + trim_inputs
#     print('here')
    
#     return t,y, u_hist

def plotit(t,x,u,ref=None,bound=False,save=False,model=None):
    plt.figure(figsize=(12, 12))  # Adjust figure size to accommodate more subplots (width, height)
    plt.rcParams.update({'font.size': 15})
    plt.rcParams['lines.linewidth'] = 4
    color_ref = 'orange'

    # Plot 1: Airspeed
    plt.subplot(5, 1, 1)
    plt.plot(t, x[0, :], label='System trajectory')
    plt.ylabel('Airspeed [m/s]')
    plt.legend(loc='best')  # Set legend location to center right
    if bound:
        plt.ylim((220,260))
    if ref is not None:
        plt.plot(t, ref[0,:], label='Reference trajectory', linestyle='dotted', color=color_ref)
        plt.legend(loc='best')  # Set legend location to center right
    plt.grid(True)

    # Plot 2: Angle of Attack
    plt.subplot(5,1,2)
    plt.plot(t, np.rad2deg(x[1, :]))
    if bound:
        plt.ylim((-5,15))
    if ref is not None:
        plt.plot(t, ref[1,:], label='Reference trajectory', linestyle='dotted', color=color_ref)
    plt.ylabel('Angle of Attack [deg]')
    plt.grid(True)

    # Plot 3: Pitch Rate
    plt.subplot(5, 1, 3)
    plt.plot(t, np.rad2deg(x[2, :]))
    if bound:
        plt.ylim((-2,2))
    if ref is not None:
        plt.plot(t, ref[2,:], label='Reference trajectory', linestyle='dotted', color=color_ref)
    plt.ylabel('Pitch Rate [deg/s]')
    plt.grid(True)

    # Plot 4: Pitch angle
    plt.subplot(5, 1, 4)
    plt.plot(t, np.rad2deg(x[3, :]))
    if bound:
        plt.ylim((0,20))
    if ref is not None:
        plt.plot(t, ref[3,:], label='Reference trajectory', linestyle='dotted', color=color_ref)
    plt.ylabel('Pitch angle [deg]')
    plt.grid(True)

    # Plot 5: Control input
    # only one input...
    plt.subplot(5, 1, 5)
    plt.plot(t, u[0, :], label='thrust', color='red')
    plt.plot(t, u[1, :], label='elevator [rad]', color='purple')

    # plt.plot(u_list.t_list, u_hist[0, :], label='thrust', color='red')
    # plt.plot(u_list.t_list, u_hist[1, :], label='elevator [deg]', color='purple')
    plt.xlabel('time [ts]')
    plt.ylabel('Control')
    # plt.title('Control')
    plt.legend(loc='center right')  # Set legend location to center right
    plt.grid(True)


    error = x[0:4,:] - ref
    vel_err =   np.sqrt(trapezoid(error[0,:]**2,t))
    aoa_err =   np.sqrt(trapezoid(error[1,:]**2,t))
    pr_err =    np.sqrt(trapezoid(error[2,:]**2,t))
    pitch_err = np.sqrt(trapezoid(error[3,:]**2,t))

    error_norm = np.sqrt(trapezoid(np.sum(error**2,0),t))
    print(f'Total error: {error_norm}')
    print(f'Velocity error: {vel_err}, AOA error: {aoa_err}, Pitch rate error: {pr_err}, Pitch angle error: {pitch_err}')

    # Adjust layout and show the figure
    plt.tight_layout()
    # plt.text(0.5, -0.5, f'L2 norm of error is {error_norm:.3f}')
    if save:
        plt.savefig(f'Output/aero_7dim_asym/{model}_nonlinear_track.png', dpi=300)
    else:
        plt.show()

# def niceplot(sim_dictionary,save=False,model=None):
#     ''' This function makes a nicer looking plot for the simulation trajectories with all models
#     and also plots error bars to compare'''

#     plt.rcParams.update({'font.size': 15})
#     plt.rcParams['lines.linewidth'] = 4
#     plt.rcParams["legend.handlelength"] = 3.5

#     t_init = sim_dictionary['initial']['t']
#     x_init = sim_dictionary['initial']['x']
#     u_init = sim_dictionary['initial']['u']
#     tp_init = sim_dictionary['initial']['tp']
#     style_init = sim_dictionary['initial']['style']
#     color_init = sim_dictionary['initial']['color']
    
#     t_ref = sim_dictionary['ref']['t']
#     x_ref = sim_dictionary['ref']['x']
#     style_ref = sim_dictionary['ref']['style']
#     color_ref = sim_dictionary['ref']['color']

#     t_vm = sim_dictionary['vm']['t']
#     x_vm = sim_dictionary['vm']['x']
#     u_vm = sim_dictionary['vm']['u']
#     tp_vm = sim_dictionary['vm']['tp']
#     style_vm = sim_dictionary['vm']['style']
#     color_vm = sim_dictionary['vm']['color']

#     t_dm = sim_dictionary['dm']['t']
#     x_dm = sim_dictionary['dm']['x']
#     u_dm = sim_dictionary['dm']['u']
#     tp_dm = sim_dictionary['dm']['tp']
#     style_dm = sim_dictionary['dm']['style']
#     color_dm = sim_dictionary['dm']['color']

#     t_vmdc = sim_dictionary['vmdc']['t']
#     x_vmdc = sim_dictionary['vmdc']['x']
#     u_vmdc = sim_dictionary['vmdc']['u']
#     tp_vmdc = sim_dictionary['vmdc']['tp']
#     style_vmdc = sim_dictionary['vmdc']['style']
#     color_vmdc = sim_dictionary['vmdc']['color']

#     # Plot 1: Airspeed
#     plt.subplot(5, 1, 1)
#     plt.plot(t_init, x_init[0, :], label='Initial', color=color_init, linestyle=style_init)
#     plt.plot(t_vm, x_vm[0, :], label='VM', color = color_vm, linestyle=style_vm)
#     plt.plot(t_dm, x_dm[0, :], label='DM', color=color_dm, linestyle=style_dm)
#     plt.plot(t_vmdc, x_vmdc[0, :], label='VMDC', color=color_vmdc, linestyle=style_vmdc)
#     plt.legend(loc='best')  # Set legend location to center right
#     plt.plot(t_ref, x_ref[0,:], label='Reference trajectory', linestyle=style_ref, color=color_ref)
#     plt.legend(loc='best')  # Set legend location to center right
#     plt.grid(True)
#     plt.ylabel('Airspeed [m/s]')

#     # Plot 2: Angle of Attack
#     plt.subplot(5,1,2)
#     plt.plot(t_init, np.rad2deg(x_init[1, :]), label='Initial', color=color_init, linestyle=style_init)
#     plt.plot(t_vm,   np.rad2deg(x_vm[1, :]), label='VM', color = color_vm, linestyle=style_vm)
#     plt.plot(t_dm,   np.rad2deg(x_dm[1, :]), label='DM', color=color_dm, linestyle=style_dm)
#     plt.plot(t_vmdc, np.rad2deg(x_vmdc[1, :]), label='VMDC', color=color_vmdc, linestyle=style_vmdc)
#     plt.plot(t_ref,  x_ref[1,:], label='Reference trajectory', linestyle=style_ref, color=color_ref)
#     plt.grid(True)
#     plt.ylabel('Angle of Attack [deg]')

#     # Plot 3: Pitch Rate
#     plt.subplot(5, 1, 3)
#     plt.plot(t_init, np.rad2deg(x_init[2, :]), label='Initial', color=color_init, linestyle=style_init)
#     plt.plot(t_vm,   np.rad2deg(x_vm[2, :]), label='VM', color = color_vm, linestyle=style_vm)
#     plt.plot(t_dm,   np.rad2deg(x_dm[2, :]), label='DM', color=color_dm, linestyle=style_dm)
#     plt.plot(t_vmdc, np.rad2deg(x_vmdc[2, :]), label='VMDC', color=color_vmdc, linestyle=style_vmdc)
#     plt.plot(t_ref,  x_ref[2,:], label='Reference trajectory', linestyle=style_ref, color=color_ref)
#     plt.grid(True)
#     plt.ylabel('Pitch Rate [deg/s]')

#     # Plot 4: Pitch angle
#     plt.subplot(5, 1, 4)
#     plt.plot(t_init, np.rad2deg(x_init[3, :]), label='Initial', color=color_init, linestyle=style_init)
#     plt.plot(t_vm,   np.rad2deg(x_vm[3, :]), label='VM', color = color_vm, linestyle=style_vm)
#     plt.plot(t_dm,   np.rad2deg(x_dm[3, :]), label='DM', color=color_dm, linestyle=style_dm)
#     plt.plot(t_vmdc, np.rad2deg(x_vmdc[3, :]), label='VMDC', color=color_vmdc, linestyle=style_vmdc)
#     plt.plot(t_ref,  x_ref[3,:], label='Reference trajectory', linestyle=style_ref, color=color_ref)
#     plt.ylabel('Pitch angle [deg]')
#     plt.grid(True)

#     # Plot 5: Control input
#     # only one input...
#     plt.subplot(5, 1, 5)
#     plt.plot(t_init, u_init[0, :], label='thrust', color=color_init, linestyle=style_init)
#     plt.plot(t_init, u_init[1, :], label='elevator [rad]', color=color_init, linestyle=style_init)
#     plt.plot(t_vm, u_vm[0, :], label='thrust', color=color_vm, linestyle=style_vm)
#     plt.plot(t_vm, u_vm[1, :], label='elevator [rad]', color=color_vm, linestyle=style_vm)
#     plt.plot(t_dm, u_dm[0, :], label='thrust', color=color_dm, linestyle=style_dm)
#     plt.plot(t_dm, u_dm[1, :], label='elevator [rad]', color=color_dm, linestyle=style_dm)
#     plt.plot(t_vmdc, u_vmdc[0, :], label='thrust', color=color_vmdc, linestyle=style_vmdc)
#     plt.plot(t_vmdc, u_vmdc[1, :], label='elevator [rad]', color=color_vmdc, linestyle=style_vmdc)
#     plt.xlabel('time [s]')
#     plt.ylabel('Control')
#     plt.legend(loc='center right')  # Set legend location to center right
#     plt.grid(True)

#     # Adjust layout and show the figure
#     plt.tight_layout()

#     if save:
#         plt.savefig(f'Output/aero_7dim_fixedmoment/{model}_nonlinear_track.png', dpi=300)
#     else:
#         plt.show()

#     trajectories = {
#         'initial': {'t': t_init, 'x': x_init, 'color': color_init, 'ref':tp_init},
#         'vm': {'t': t_vm, 'x': x_vm, 'color': color_vm, 'ref':tp_vm},
#         'dm': {'t': t_dm, 'x': x_dm, 'color': color_dm, 'ref':tp_dm},
#         'vmdc': {'t': t_vmdc, 'x': x_vmdc, 'color': color_vmdc, 'ref':tp_vmdc}
#     }
#     # Error computation for all trajectories
#     print("\n" + "="*70)
#     print("TRACKING ERROR ANALYSIS")
#     print("="*70)
    
#     error_data = {}
#     for traj_name, traj_data in trajectories.items():
#         t = traj_data['t']
#         x = traj_data['x']
#         tp = traj_data['ref']
        
#         # # Interpolate reference trajectory to match current trajectory's time points
#         # x_ref_interp = np.zeros_like(x)
#         # for i in range(4):
#         #     x_ref_interp[i, :] = np.interp(t, t_ref, x_ref[i, :])
        
#         # Compute errors
#         error = x[0:4, :] - tp
#         vel_err =   np.sqrt(trapezoid(error[0, :]**2, t))
#         aoa_err =   np.sqrt(trapezoid(error[1, :]**2, t))
#         pr_err =    np.sqrt(trapezoid(error[2, :]**2, t))
#         pitch_err = np.sqrt(trapezoid(error[3, :]**2, t))
#         error_norm = np.sqrt(trapezoid(np.sum(error**2, 0), t))

#         error_data[traj_name] = {
#             'vel_err': vel_err,
#             'aoa_err': aoa_err,
#             'pr_err': pr_err,
#             'pitch_err': pitch_err,
#             'error_norm': error_norm
#         }
        
#         print(f"\n{traj_name.upper()} Model:")
#         print(f"  Total error (L2 norm): {error_norm:.6f}")
#         print(f"  Velocity error:        {vel_err:.6f}")
#         print(f"  AOA error:             {aoa_err:.6f}")
#         print(f"  Pitch rate error:      {pr_err:.6f}")
#         print(f"  Pitch angle error:     {pitch_err:.6f}")
    
#     print("="*70 + "\n")

#     # Start new figure for error plot
#     plt.figure()
    
#     models_list = list(trajectories.keys())
#     error_types = ['vel_err', 'aoa_err', 'pr_err', 'pitch_err', 'error_norm']
#     error_labels = ['Velocity', 'AOA', 'Pitch Rate', 'Pitch Angle', 'Total']
    
#     # Define bar patterns/hatches
#     hatches = ['///', '\\\\\\', '|||', '---', 'xxx']
    
#     # Set up bar positions
#     x_pos = np.arange(len(models_list))
#     bar_width = 0.15
    
#     # Plot bars for each error type
#     for i, (error_type, error_label, hatch) in enumerate(zip(error_types, error_labels, hatches)):
#         offset = (i - 2) * bar_width
#         values = [error_data[model][error_type] for model in models_list]
#         plt.bar(x_pos + offset, values, bar_width, label=error_label, hatch=hatch, edgecolor='black', linewidth=1.5)
    
#     # Customize the plot
#     plt.xlabel('Model Type', fontsize=12)
#     plt.ylabel('Error Magnitude', fontsize=12)
#     plt.title('Tracking Error Comparison Across Models', fontsize=14, fontweight='bold')
#     plt.xticks(x_pos, [m.upper() for m in models_list])
#     plt.legend(loc='upper right', fontsize=10)
#     plt.grid(True, axis='y', alpha=0.3)
#     plt.show()
    
def nonlinear_track_velocity(design,fp1,fp2,data_file,tf,switch_time=1e6):
    '''This code will simulate the nonlinear tracking problem. It constructs a linear controller that uses LQR + integrator
    to track the reference.
    THE INTEGRATOR TRACKS VELOCITY
    Inputs:
        design is an array [chs, whs] that gives the design parameters we are testing
        fp1,fp2 are arrays [v0,climb_ang] that give the first and second desired velocities and climb angles
        data_file is the aero data file we are using for all aerodynamic data
        tf is final time of simulation
        switch_time is the time where we command aircraft to switch from tracking fp1 to fp2. By default, this is very large to prevent switching.
        
    Outputs:
        time, trajectory, control history, reference trajectory'''

    chs = design[0]
    whs = design[1]

    v01 = fp1[0]
    climb_ang1 = fp1[1]

    # Using given flight point, compute trimmed angle of attack and inputs.
    alpha01, deltae01, deltaT01, pitch01 = compute_trim_eom(np.array([v01,chs,whs]), data_file,climb_angle=climb_ang1)

    full_flight_point = np.array([v01,alpha01,chs,whs,deltae01])
    trim_point1 = np.array([v01,alpha01,0,pitch01])
    trim_inputs1 = np.array([deltaT01,deltae01])
    tp1 = trim_point1.copy()
    tp1[1] = np.rad2deg(tp1[1])
    tp1[2] = np.rad2deg(tp1[2])
    tp1[3] = np.rad2deg(tp1[3])
    # print(f'trim point 1:{tp1}')
    interp_dict = aero_interps(data_file)
    A1,B1 = build_AB(interp_dict,full_flight_point)
    C = np.array([[1,0,0,0]])
    C1 = C

    k = C1.shape[0]
    n = A1.shape[0]
    m = B1.shape[1]

    Abar1 = np.concatenate((np.concatenate((A1,np.zeros((n,k))),axis=1),np.concatenate((C1,np.zeros((k,k))),axis=1)),axis=0)
    Bbar1 = np.concatenate((B1,np.zeros((k,m))),axis=0)

    Q = np.diag([1, 1, 1, 100, 1])
    R = np.diag([1,100]) # penalize elevator usage more to avoid too much variation in AOA.

    # This is the control gain we want to use, we need to create an integrator state in the nonlinear model to use this.
    # Then, with each application, we simply subtract from the nonlinear state the trim state and multiply with K...
    K1 = ct.lqr(Abar1,Bbar1,Q,R)[0]

    v02 = fp2[0]
    climb_ang2 = fp2[1]

    alpha02, deltae02, deltaT02, pitch02 = compute_trim_eom(np.array([v02,chs,whs]), data_file,climb_angle=climb_ang2)

    full_flight_point = np.array([v02,alpha02,chs,whs,deltae02])
    trim_point2 = np.array([v02,alpha02,0,pitch02])
    tp2 = trim_point2.copy()
    tp2[1] = np.rad2deg(tp2[1])
    tp2[2] = np.rad2deg(tp2[2])
    tp2[3] = np.rad2deg(tp2[3])
    # print(f'trim point 2: {tp2}')
    trim_inputs2 = np.array([deltaT02,deltae02])
    A2,B2 = build_AB(interp_dict,full_flight_point)
    C2 = C

    k = C2.shape[0]
    n = A2.shape[0]
    m = B2.shape[1]

    Abar2 = np.concatenate((np.concatenate((A2,np.zeros((n,k))),axis=1),np.concatenate((C2,np.zeros((k,k))),axis=1)),axis=0)
    Bbar2 = np.concatenate((B2,np.zeros((k,m))),axis=0)

    K2 = ct.lqr(Abar2,Bbar2,Q,R)[0]

    # Now we have 2 controllers, K1 and K2...

    # some constants
    g = 9.81  # m/s^2
    weight = 331522 # lb
    weight = weight * 4.44822 # convert to N
    mass = weight / g  # kg
    Iy = 3759014 # slug ft^2
    Iy = Iy * 14.5939 * 0.3048**2 # kg m^2
    thrust_force = 96000 # lb
    maxT = thrust_force * 4.44822 * 1.5 # convert to N

    M_interp = interp_dict['M']
    Di_interp = interp_dict['Di']
    L_interp = interp_dict['L']

    def linear_controller1(t,x):
        # take current state, subtract from it the trimmed state and multiply with gain K.
        # send back the resulting controller after applying proper bounds.

        x_bar = x - np.concatenate((trim_point1.reshape(4,-1),np.array([[0]])),axis=0).flatten()
        u = -K1 @ x_bar
        u = u + trim_inputs1
        u[0] = np.clip(u[0],0,1)
        u[1] = np.clip(u[1],np.deg2rad(-30),np.deg2rad(30))
        
        return u
    
    def linear_controller2(t,x):
        x_bar = x - np.concatenate((trim_point2.reshape(4,-1),np.array([[0]])),axis=0).flatten()
        u = -K2 @ x_bar
        u = u + trim_inputs2
        u[0] = np.clip(u[0],0,1)
        u[1] = np.clip(u[1],np.deg2rad(-30),np.deg2rad(30))
        return u

    def nonlinear_model(t, x):
        # with an integrator

        v = x[0]
        alpha = x[1]
        q = x[2]
        theta = x[3]

        # solve for control using integral controller
        
        # switch based on fixed time
        if t < switch_time:
            u = linear_controller1(t,x)
        else:
            u = linear_controller2(t,x)
        xr = v01 if t < switch_time else v02

        deltaT = u[0]
        deltae = u[1]

        flight_point = np.asarray([v, alpha, chs, whs, deltae])
        # print(flight_point)
        
        M = M_interp(flight_point)[0]
        Di = Di_interp(flight_point)[0]
        L = L_interp(flight_point)[0]

        Ft = deltaT * maxT

        x_dot = np.zeros_like(x)
        x_dot[0] = Ft/mass * np.cos(alpha) - Di/mass - g * np.sin(theta - alpha)
        x_dot[1] = -Ft/(mass * v) * np.sin(alpha) - L/(mass * v) + g/v * np.cos(theta - alpha) + q
        x_dot[2] = M / Iy
        x_dot[3] = q
        x_dot[4] = v - xr           # integrator state

        threshold = 5
        if x[4] > threshold:      # prevent wind up, if integrator state too large, keep constant...
            x_dot[4] = min(x_dot[4],0)
        elif x[4] < -threshold:
            x_dot[4] = max(x_dot[4],0)

        # print(f'derivative: {x_dot}')

        return x_dot

    # Build initial condition using trim point around slower state
    fp0 = np.asarray([v01 - 10, chs, whs])
    alpha0, deltae0, deltaT0, pitch0 = compute_trim_eom(fp0, data_file,climb_angle=0)
    trim_point0 = np.array([fp0[0],alpha0,0,pitch0])

    x0 = np.concatenate((trim_point0.reshape(4,-1),np.array([[0]])),axis=0)
    x0 = x0.flatten()

    sol_fwd = solve_ivp(lambda t,x: nonlinear_model(t,x)  , (0.0, tf), x0, method='BDF')
    if not sol_fwd.success:
        raise RuntimeError("Forward integration failed: " + str(sol_fwd.message))

    t = sol_fwd.t
    x = sol_fwd.y

    u_hist = np.zeros((m,t.size))
    tp_hist = np.zeros((n,t.size))

    for i, ti in enumerate(t):
        if ti <= switch_time:
            u = linear_controller1(ti,x[:,i])
            tp_hist[:,i] = trim_point1
        else:
            u = linear_controller2(ti,x[:,i])
            tp_hist[:,i] = trim_point2
        u_hist[:,i] = u.flatten()

    return t,x, u_hist, tp_hist

def nonlinear_track_pitch(design,fp1,fp2,data_file,tf,switch_time=100):
    '''This code will simulate the nonlinear tracking problem. It constructs a linear controller that uses LQR + integrator
    to track the reference.
    THE INTEGRATOR TRACKS PITCH ANGLE
    Inputs:
        design is an array [chs, whs] that gives the design parameters we are testing
        fp1,fp2 are arrays [v0,climb_ang] that give the first and second desired velocities and climb angles
        data_file is the aero data file we are using for all aerodynamic data
        tf is final time of simulation
        switch_time is the time where we command aircraft to switch from tracking fp1 to fp2. By default, this is very large to prevent switching.
        
    Outputs:
        time, trajectory, control history, trim point trajectory'''

    chs = design[0]
    whs = design[1]

    v01 = fp1[0]
    climb_ang1 = fp1[1]

    # Using given flight point, compute trimmed angle of attack and inputs.
    if design.size < 3:
        data_file = 'Aero_data/aero_7dim2.npz'
        alpha01, deltae01, deltaT01, pitch01, _cost = compute_trim_eom(np.asarray([v01,chs,whs]), data_file,climb_angle=climb_ang1)
        full_flight_point = np.array([v01,alpha01,chs,whs,deltae01])
    else:
        data_file = 'Aero_data/aero_merged_cg_17.npz'     # MAKE SURE TO PASS THE CORRECT NUMBER OF FLIGHT_POINTS TO compute_trim_eom WHEN USING THIS
        ws = design[2]
        ths= design[3]
        wd = design[4]
        
        alpha01, deltae01, deltaT01, pitch01, _cost = compute_trim_eom(np.asarray([v01,chs,whs,ws,ths,wd]), data_file,climb_angle=climb_ang1)
        full_flight_point = np.array([v01,alpha01,chs,whs,deltae01,ws,ths,wd])

    
    trim_point1 = np.array([v01,alpha01,0,pitch01])
    trim_inputs1 = np.array([deltaT01,deltae01])
    tp1 = trim_point1.copy()
    tp1[1] = np.rad2deg(tp1[1])
    tp1[2] = np.rad2deg(tp1[2])
    tp1[3] = np.rad2deg(tp1[3])
    print(f'trim point 1:{tp1}')
    # print(f'trim inputs 1: {trim_inputs1}')
    interp_dict = aero_interps(data_file)
    A1,B1 = build_AB(interp_dict,full_flight_point)
    C = np.array([[0,0,0,1]])
    C1 = C

    k = C1.shape[0]
    n = A1.shape[0]
    m = B1.shape[1]

    Abar1 = np.concatenate((np.concatenate((A1,np.zeros((n,k))),axis=1),np.concatenate((C1,np.zeros((k,k))),axis=1)),axis=0)
    Bbar1 = np.concatenate((B1,np.zeros((k,m))),axis=0)

    # I think this is main
    Q = np.diag([1,100,1,100,50])
    R = np.diag([0.1,1000]) # penalize elevator usage more to avoid too much variation in AOA.

    # This is the control gain we want to use, we need to create an integrator state in the nonlinear model to use this.
    # Then, with each application, we simply subtract from the nonlinear state the trim state and multiply with K...
    K1 = ct.lqr(Abar1,Bbar1,Q,R)[0]

    v02 = fp2[0]
    climb_ang2 = fp2[1]

    if design.size < 3:
        data_file = 'Aero_data/aero_7dim2.npz'
        alpha02, deltae02, deltaT02, pitch02, _cost = compute_trim_eom(np.asarray([v02,chs,whs]), data_file,climb_angle=climb_ang2)
        full_flight_point = np.array([v02,alpha02,chs,whs,deltae02])
    else:
        data_file = 'Aero_data/aero_merged_cg_17.npz'     # MAKE SURE TO PASS THE CORRECT NUMBER OF FLIGHT_POINTS TO compute_trim_eom WHEN USING THIS
        ws = design[2]
        ths= design[3]
        wd = design[4]
        
        alpha02, deltae02, deltaT02, pitch02, _cost = compute_trim_eom(np.asarray([v02,chs,whs,ws,ths,wd]), data_file,climb_angle=climb_ang2, verbose=False)
        full_flight_point = np.array([v02,alpha02,chs,whs,deltae02,ws,ths,wd])

    trim_point2 = np.array([v02,alpha02,0,pitch02])
    tp2 = trim_point2.copy()
    tp2[1] = np.rad2deg(tp2[1])
    tp2[2] = np.rad2deg(tp2[2])
    tp2[3] = np.rad2deg(tp2[3])
    print(f'trim point 2: {tp2}')
    # print(f'Inputs: {deltaT02} and {deltae02}')
    trim_inputs2 = np.array([deltaT02,deltae02])
    A2,B2 = build_AB(interp_dict,full_flight_point)

    # eigvals = np.linalg.eigvals(A2)
    # print("Eigenvalues:", eigvals)
    # print("Real parts:", eigvals.real)
    # stiffness_ratio = np.max(np.abs(eigvals.real)) / np.min(np.abs(eigvals.real[eigvals.real != 0]))
    # print("Stiffness ratio:", stiffness_ratio)
    C2 = C

    k = C2.shape[0]
    n = A2.shape[0]
    m = B2.shape[1]

    Abar2 = np.concatenate((np.concatenate((A2,np.zeros((n,k))),axis=1),np.concatenate((C2,np.zeros((k,k))),axis=1)),axis=0)
    Bbar2 = np.concatenate((B2,np.zeros((k,m))),axis=0)

    K2 = ct.lqr(Abar2,Bbar2,Q,R)[0]

    # Now we have 2 controllers, K1 and K2...

    # some constants
    g = 9.81  # m/s^2
    weight = 331522 # lb
    weight = weight * 4.44822 # convert to N
    mass = weight / g  # kg
    Iy = 3759014 # slug ft^2
    Iy = Iy * 14.5939 * 0.3048**2 # kg m^2
    thrust_force = 96000 # lb
    maxT = thrust_force * 4.44822 * 1.5 # convert to N

    M_interp = interp_dict['M']
    Di_interp = interp_dict['Di']
    L_interp = interp_dict['L']

    def linear_controller1(t,x):
        # take current state, subtract from it the trimmed state and multiply with gain K.
        # send back the resulting controller after applying proper bounds.

        x_bar = x - np.concatenate((trim_point1.reshape(4,-1),np.array([[0]])),axis=0).flatten()
        u = -K1 @ x_bar
        u = u + trim_inputs1
        u[0] = np.clip(u[0],0,1)
        u[1] = np.clip(u[1],np.deg2rad(-30),np.deg2rad(30))
        
        return u
    
    def linear_controller2(t,x):
        x_bar = x - np.concatenate((trim_point2.reshape(4,-1),np.array([[0]])),axis=0).flatten()
        u = -K2 @ x_bar
        u = u + trim_inputs2
        u[0] = np.clip(u[0],0,1)
        u[1] = np.clip(u[1],np.deg2rad(-30),np.deg2rad(30))
        return u

    def nonlinear_model(t, x, mode):
        # with an integrator

        v = x[0]
        alpha = x[1]
        q = x[2]
        theta = x[3]

        # solve for control using integral controller
        
        # switch based on fixed time
        if mode == 1:
            u = linear_controller1(t,x)
            xr = pitch01
        elif mode == 2:
            u = linear_controller2(t,x)
            xr = pitch02

        deltaT = u[0]
        deltae = u[1]

        if design.size < 3:
            flight_point = np.asarray([v, alpha, chs, whs, deltae])
        else:
            flight_point = np.asarray([v,alpha,chs,whs,deltae,ws,ths,wd])
        
        M = M_interp(flight_point)[0]
        Di = Di_interp(flight_point)[0]
        L = L_interp(flight_point)[0]

        Ft = deltaT * maxT

        x_dot = np.zeros_like(x)
        x_dot[0] = Ft/mass * np.cos(alpha) - Di/mass - g * np.sin(theta - alpha)
        x_dot[1] = -Ft/(mass * v) * np.sin(alpha) - L/(mass * v) + g/v * np.cos(theta - alpha) + q
        x_dot[2] = M / Iy
        x_dot[3] = q
        x_dot[4] = theta - xr           # integrator state

        threshold = 0.3
        if x[4] > threshold:      # prevent wind up, if integrator state too large, keep constant...
            x_dot[4] = min(x_dot[4],0)
        elif x[4] < -threshold:
            x_dot[4] = max(x_dot[4],0)

        # print(f'derivative: {x_dot}')

        return x_dot
    
    # Build initial condition using trim point around slower state
    fp0 = np.asarray([v01 - 5, chs, whs])
    if design.size < 3:
        data_file = 'Aero_data/aero_7dim2.npz'
        alpha0, deltae0, deltaT0, pitch0, _cost = compute_trim_eom(np.asarray([v01-5,chs,whs]), data_file,climb_angle=climb_ang2)
        full_flight_point = np.array([v02,alpha02,chs,whs,deltae02])
    else:
        data_file = 'Aero_data/aero_merged_cg_17.npz'  
        ws = design[2]
        ths= design[3]
        wd = design[4]
        
        alpha0, deltae0, deltaT0, pitch0, _cost = compute_trim_eom(np.asarray([v01-5,chs,whs,ws,ths,wd]), data_file,climb_angle=climb_ang1 - 5)
        full_flight_point = np.array([v02,alpha02,chs,whs,deltae02,ws,ths,wd])
    trim_point0 = np.array([fp0[0],alpha0,0,pitch0])

    tp0 = trim_point0.copy()
    tp0[1] = np.rad2deg(tp0[1])
    tp0[2] = np.rad2deg(tp0[2])
    tp0[3] = np.rad2deg(tp0[3])
    print(f'trim point 0: {tp0}')

    x0 = np.concatenate((trim_point0.reshape(4,-1),np.array([[0]])),axis=0)
    x0 = x0.flatten()

    #--------

    t_eval1 = np.linspace(0,switch_time,600)
    sol_fwd = solve_ivp(lambda t,x: nonlinear_model(t,x, 1)  , (0.0, switch_time), x0, method='BDF', t_eval = t_eval1)
    # sol_fwd = solve_ivp(lambda t,x: nonlinear_model(t,x, 1)  , (0.0, switch_time), x0, method='BDF')
    t1 = sol_fwd.t
    x1 = sol_fwd.y
    # Take new initial condition at end of previous and set integrator state to 0
    new_init = x1[:,-1]
    new_init[4] = 0
    t_eval2 = np.linspace(switch_time, tf, 600)
    sol_fwd = solve_ivp(lambda t,x: nonlinear_model(t,x, 2)  , (switch_time, tf), new_init, method='BDF', t_eval = t_eval2)
    # sol_fwd = solve_ivp(lambda t,x: nonlinear_model(t,x, 2)  , (switch_time, tf), new_init, method='BDF')
    t2 = sol_fwd.t
    x2 = sol_fwd.y

    t = np.concatenate((t1,t2),axis=0)
    x = np.concatenate((x1,x2),axis=1)
    

    #---------

    # sol_fwd = solve_ivp(lambda t,x: nonlinear_model(t,x)  , (0.0, tf), x0, method='BDF')
    # if not sol_fwd.success:
    #     raise RuntimeError("Forward integration failed: " + str(sol_fwd.message))

    # t = sol_fwd.t
    # x = sol_fwd.y

    u_hist = np.zeros((m,t.size))
    tp_hist = np.zeros((n,t.size))

    for i, ti in enumerate(t):
        if ti <= switch_time:
            u = linear_controller1(ti,x[:,i])
            tp_hist[:,i] = trim_point1
        else:
            u = linear_controller2(ti,x[:,i])
            tp_hist[:,i] = trim_point2
        u_hist[:,i] = u.flatten()

    return t,x, u_hist, tp_hist


def nonlinear_track_pitch_air_force(design,fp1,fp2,data_file,tf,switch_time=100):
    '''This code will simulate the nonlinear tracking problem. It constructs a linear controller that uses LQR + integrator
    to track the reference.
    THE INTEGRATOR TRACKS PITCH ANGLE
    Inputs:
        design is an array [chs, whs] that gives the design parameters we are testing
        fp1,fp2 are arrays [v0,climb_ang] that give the first and second desired velocities and climb angles
        data_file is the aero data file we are using for all aerodynamic data
        tf is final time of simulation
        switch_time is the time where we command aircraft to switch from tracking fp1 to fp2. By default, this is very large to prevent switching.
        
    Outputs:
        time, trajectory, control history, trim point trajectory'''

    chs = design[0]
    whs = design[1]

    v01 = fp1[0]
    climb_ang1 = fp1[1]

    # Using given flight point, compute trimmed angle of attack and inputs.
    alpha01, deltae01, deltaT01, pitch01 = compute_trim_eom(np.array([v01,chs,whs]), data_file,climb_angle=climb_ang1)

    full_flight_point = np.array([v01,alpha01,chs,whs,deltae01])
    trim_point1 = np.array([v01,alpha01,0,pitch01])
    trim_inputs1 = np.array([deltaT01,deltae01])
    tp1 = trim_point1.copy()
    tp1[1] = np.rad2deg(tp1[1])
    tp1[2] = np.rad2deg(tp1[2])
    tp1[3] = np.rad2deg(tp1[3])
    print(f'trim point 1:{tp1}')
    interp_dict = aero_interps(data_file)
    A1,B1 = build_AB(interp_dict,full_flight_point)
    C = np.array([[0,0,0,1]])
    C1 = C

    k = C1.shape[0]
    n = A1.shape[0]
    m = B1.shape[1]

    Abar1 = np.concatenate((np.concatenate((A1,np.zeros((n,k))),axis=1),np.concatenate((C1,np.zeros((k,k))),axis=1)),axis=0)
    Bbar1 = np.concatenate((B1,np.zeros((k,m))),axis=0)

    # I think this is main
    # Q = np.diag([1,100,1,100,100])
    # R = np.diag([0.1,1000]) # penalize elevator usage more to avoid too much variation in AOA.

    # Trying from frequency analysis
    Q = np.diag([100,1,1,1000,10])
    R = np.diag([0.1,10])

    # Another one
    # Q = np.diag([100, 1, 1, 1000, 10])
    # R = np.diag([0.1, 10])

    # This is the control gain we want to use, we need to create an integrator state in the nonlinear model to use this.
    # Then, with each application, we simply subtract from the nonlinear state the trim state and multiply with K...
    K1 = ct.lqr(Abar1,Bbar1,Q,R)[0]

    v02 = fp2[0]
    climb_ang2 = fp2[1]

    alpha02, deltae02, deltaT02, pitch02 = compute_trim_eom(np.array([v02,chs,whs]), data_file,climb_angle=climb_ang2)
    # print(f'Trimmed alpha: {np.rad2deg(alpha02)}, deltae: {np.rad2deg(deltae02)}, deltaT: {deltaT02}, pitch: {np.rad2deg(pitch02)}')

    full_flight_point = np.array([v02,alpha02,chs,whs,deltae02])
    trim_point2 = np.array([v02,alpha02,0,pitch02])
    tp2 = trim_point2.copy()
    tp2[1] = np.rad2deg(tp2[1])
    tp2[2] = np.rad2deg(tp2[2])
    tp2[3] = np.rad2deg(tp2[3])
    print(f'trim point 2: {tp2}')
    trim_inputs2 = np.array([deltaT02,deltae02])
    A2,B2 = build_AB(interp_dict,full_flight_point)

    # eigvals = np.linalg.eigvals(A2)
    # print("Eigenvalues:", eigvals)
    # print("Real parts:", eigvals.real)
    # stiffness_ratio = np.max(np.abs(eigvals.real)) / np.min(np.abs(eigvals.real[eigvals.real != 0]))
    # print("Stiffness ratio:", stiffness_ratio)
    C2 = C

    k = C2.shape[0]
    n = A2.shape[0]
    m = B2.shape[1]

    Abar2 = np.concatenate((np.concatenate((A2,np.zeros((n,k))),axis=1),np.concatenate((C2,np.zeros((k,k))),axis=1)),axis=0)
    Bbar2 = np.concatenate((B2,np.zeros((k,m))),axis=0)

    K2 = ct.lqr(Abar2,Bbar2,Q,R)[0]

    # Now we have 2 controllers, K1 and K2...

    # some constants
    g = 9.81  # m/s^2
    weight = 331522 # lb
    weight = weight * 4.44822 # convert to N
    mass = weight / g  # kg
    Iy = 3759014 # slug ft^2
    Iy = Iy * 14.5939 * 0.3048**2 # kg m^2
    thrust_force = 96000 # lb
    maxT = thrust_force * 4.44822 * 1.5 # convert to N

    M_interp = interp_dict['M']
    Di_interp = interp_dict['Di']
    L_interp = interp_dict['L']

    def linear_controller1(t,x):
        # take current state, subtract from it the trimmed state and multiply with gain K.
        # send back the resulting controller after applying proper bounds.

        x_bar = x - np.concatenate((trim_point1.reshape(4,-1),np.array([[0]])),axis=0).flatten()
        u = -K1 @ x_bar
        u = u + trim_inputs1
        u[0] = np.clip(u[0],0,1)
        u[1] = np.clip(u[1],np.deg2rad(-30),np.deg2rad(30))
        
        return u
    
    def linear_controller2(t,x):
        x_bar = x - np.concatenate((trim_point2.reshape(4,-1),np.array([[0]])),axis=0).flatten()
        u = -K2 @ x_bar
        u = u + trim_inputs2
        u[0] = np.clip(u[0],0,1)
        u[1] = np.clip(u[1],np.deg2rad(-30),np.deg2rad(30))
        return u

    def nonlinear_model(t, x, mode):
        # with an integrator

        v = x[0]
        alpha = x[1]
        q = x[2]
        theta = x[3]

        # solve for control using integral controller
        
        # switch based on fixed time
        if mode == 1:
            u = linear_controller1(t,x)
            xr = pitch01
        elif mode == 2:
            u = linear_controller2(t,x)
            xr = pitch02

        deltaT = u[0]
        deltae = u[1]

        flight_point = np.asarray([v, alpha, chs, whs, deltae])
        # print(x)
        # print(flight_point)
        
        M = M_interp(flight_point)[0]
        Di = Di_interp(flight_point)[0]
        L = L_interp(flight_point)[0]

        Ft = deltaT * maxT

        x_dot = np.zeros_like(x)
        x_dot[0] = Ft/mass * np.cos(alpha) - Di/mass - g * np.sin(theta - alpha)
        x_dot[1] = -Ft/(mass * v) * np.sin(alpha) - L/(mass * v) + g/v * np.cos(theta - alpha) + q
        x_dot[2] = M / Iy
        x_dot[3] = q
        x_dot[4] = theta - xr           # integrator state

        threshold = 0.3
        if x[4] > threshold:      # prevent wind up, if integrator state too large, keep constant...
            x_dot[4] = min(x_dot[4],0)
        elif x[4] < -threshold:
            x_dot[4] = max(x_dot[4],0)

        # print(f'derivative: {x_dot}')

        return x_dot
    
    # Build initial condition using trim point around slower state
    fp0 = np.asarray([v01, chs, whs])
    alpha0, deltae0, deltaT0, pitch0 = compute_trim_eom(fp0, data_file,climb_angle=climb_ang1)
    # trim_point0 = np.array([fp0[0] - 5,alpha0,0,pitch0 - np.deg2rad(5)])
    trim_point0 = np.array([fp0[0],alpha0,0,pitch0])

    tp0 = trim_point0.copy()
    tp0[1] = np.rad2deg(tp0[1])
    tp0[2] = np.rad2deg(tp0[2])
    tp0[3] = np.rad2deg(tp0[3])
    print(f'trim point 0: {tp0}')

    x0 = np.concatenate((trim_point0.reshape(4,-1),np.array([[0]])),axis=0)
    x0 = x0.flatten()

    #--------
    sol_fwd = solve_ivp(lambda t,x: nonlinear_model(t,x, 1)  , (0.0, switch_time), x0, method='BDF')
    t1 = sol_fwd.t
    x1 = sol_fwd.y
    # Take new initial condition at end of previous and set integrator state to 0
    new_init = x1[:,-1]
    new_init[4] = 0
    sol_fwd = solve_ivp(lambda t,x: nonlinear_model(t,x, 2)  , (switch_time, tf), new_init, method='BDF')
    t2 = sol_fwd.t
    x2 = sol_fwd.y

    t = np.concatenate((t1,t2),axis=0)
    x = np.concatenate((x1,x2),axis=1)
    

    #---------

    # sol_fwd = solve_ivp(lambda t,x: nonlinear_model(t,x)  , (0.0, tf), x0, method='BDF')
    # if not sol_fwd.success:
    #     raise RuntimeError("Forward integration failed: " + str(sol_fwd.message))

    # t = sol_fwd.t
    # x = sol_fwd.y

    u_hist = np.zeros((m,t.size))
    tp_hist = np.zeros((n,t.size))

    for i, ti in enumerate(t):
        if ti <= switch_time:
            u = linear_controller1(ti,x[:,i])
            tp_hist[:,i] = trim_point1
        else:
            u = linear_controller2(ti,x[:,i])
            tp_hist[:,i] = trim_point2
        u_hist[:,i] = u.flatten()

    return t,x, u_hist, tp_hist



if __name__ == "__main__":
    # Test the nonlinear model by integrating it forward in time.
    v0 = 235 # speed in m/s

    # Initial
    chs = 5 # center halfspan in m
    whs = 12 # wing halfspan in m
    model_type = 'initial'

    opt_path3 = 'Output/aero_7dim_asym/N6_volmax_110con'
    opt_path1 = 'Output/aero_7dim_asym/N6_volmax'
    opt_path2 = 'Output/aero_7dim_asym/N6_110max'
    models = [opt_path1, opt_path2, opt_path3]

    speed1 = 190
    climb_angle1 = 10
    speed2 = 210
    climb_angle2 = 0
    
    data_file = 'Aero_data/aero_7dim2.npz'
    flight_point = np.array([v0,chs,whs])
    tf = 40 # seconds
    st = 20 # switching time

    savefig = False


    #%% Velocity integrator tracker for initial and 3 models

    model_type = 'initial'
    t,x,u_hist,tp_hist = nonlinear_track_velocity(np.array([chs,whs]),np.array([speed1,climb_angle1]), 
                                          np.array([speed2,climb_angle2]),data_file, tf,switch_time=st)
    plotit(t,x,u_hist,ref=tp_hist,bound=False,model=model_type, save=savefig)

    for opt_path in models:

        model_type = opt_path.split('/')[-1][3:]
        print(model_type)
        output_hdf5_file = f"{opt_path}/opt_vars.hdf5"
        x = extract_dataset_from_results(output_hdf5_file, "x")
        chs = x[0]
        whs = x[1]

        t,x,u_hist,tp_hist = nonlinear_track_velocity(np.array([chs,whs]),np.array([speed1,climb_angle1]), 
                                          np.array([speed2,climb_angle2]),data_file, tf,switch_time=st)
        plotit(t,x,u_hist,ref=tp_hist,bound=False,model=model_type,save=savefig)

    #%%


    # t,x,u_hist = run_nonlinear(flight_point,data_file,tf)

    # t,x,u_hist, tp_hist = run_linear(flight_point,data_file,tf)
    # print(x[:,-1])

    # t,x,u_hist = linear_track(flight_point,data_file,tf)

    # t,x,u_hist = nonlinear_track(flight_point,data_file,tf)

    # print out final state for convenience.
    y = x.copy()
    y[1,:] = np.rad2deg(y[1,:])
    y[2,:] = np.rad2deg(y[2,:])
    y[3,:] = np.rad2deg(y[3,:])
    # print(f'final state: {y[:,-1]}')

    #%% Plot all
    # plotit(t,x,u_hist,ref=tp_hist,bound=False)

    # Show integrator state
    plt.plot(t,x[4,:])
    plt.show()

