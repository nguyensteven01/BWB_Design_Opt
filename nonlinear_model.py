''' This script will be a nonlinear model from Stevens/Lewis'''
from interpolatory_model import *
import numpy as np
from trim_aircraft import compute_trim_eom
import control as ct
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid
from read_hdf5 import extract_dataset_from_results


def plotit(t,x,u,ref=None,model=None):
    plt.figure(figsize=(12, 12))  # Adjust figure size to accommodate more subplots (width, height)
    plt.rcParams.update({'font.size': 15})
    plt.rcParams['lines.linewidth'] = 4
    color_ref = 'orange'

    # Plot 1: Airspeed
    plt.subplot(5, 1, 1)
    plt.plot(t, x[0, :], label='System trajectory')
    plt.ylabel('Airspeed [m/s]')
    plt.legend(loc='best')  # Set legend location to center right
    if ref is not None:
        plt.plot(t, ref[0,:], label='Reference trajectory', linestyle='dotted', color=color_ref)
        plt.legend(loc='best')  # Set legend location to center right
    plt.grid(True)

    # Plot 2: Angle of Attack
    plt.subplot(5,1,2)
    plt.plot(t, np.rad2deg(x[1, :]))
    if ref is not None:
        plt.plot(t, ref[1,:], label='Reference trajectory', linestyle='dotted', color=color_ref)
    plt.ylabel('Angle of Attack [deg]')
    plt.grid(True)

    # Plot 3: Pitch Rate
    plt.subplot(5, 1, 3)
    plt.plot(t, np.rad2deg(x[2, :]))
    if ref is not None:
        plt.plot(t, ref[2,:], label='Reference trajectory', linestyle='dotted', color=color_ref)
    plt.ylabel('Pitch Rate [deg/s]')
    plt.grid(True)

    # Plot 4: Pitch angle
    plt.subplot(5, 1, 4)
    plt.plot(t, np.rad2deg(x[3, :]))
    if ref is not None:
        plt.plot(t, ref[3,:], label='Reference trajectory', linestyle='dotted', color=color_ref)
    plt.ylabel('Pitch angle [deg]')
    plt.grid(True)

    # Plot 5: Control input
    # only one input...
    plt.subplot(5, 1, 5)
    plt.plot(t, u[0, :], label='thrust', color='red')
    plt.plot(t, u[1, :], label='elevator [rad]', color='purple')
    plt.xlabel('time [t]')
    plt.ylabel('Control')
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

    plt.tight_layout()

    plt.show()
    
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

def nonlinear_track_pitch(design,fp1,fp2,data_file,tf,switch_time=1e6):
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
    interp_dict = aero_interps(data_file)
    A1,B1 = build_AB(interp_dict,full_flight_point)
    C = np.array([[0,0,0,1]])
    C1 = C

    k = C1.shape[0]
    n = A1.shape[0]
    m = B1.shape[1]

    Abar1 = np.concatenate((np.concatenate((A1,np.zeros((n,k))),axis=1),np.concatenate((C1,np.zeros((k,k))),axis=1)),axis=0)
    Bbar1 = np.concatenate((B1,np.zeros((k,m))),axis=0)

    Q = np.diag([1,100,1,100,100])
    R = np.diag([0.1,1000]) # penalize elevator usage more to avoid too much variation in AOA.

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
        xr = pitch01 if t < switch_time else pitch02

        deltaT = u[0]
        deltae = u[1]

        flight_point = np.asarray([v, alpha, chs, whs, deltae])        
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

        return x_dot
    
    # Build initial condition using trim point around slower state
    fp0 = np.asarray([v01, chs, whs])
    alpha0, deltae0, deltaT0, pitch0 = compute_trim_eom(fp0, data_file,climb_angle=climb_ang1)
    trim_point0 = np.array([fp0[0] - 5,alpha0,0,pitch0 - np.deg2rad(5)])

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

    #%% Velocity integrator tracker for initial and 3 models

    model_type = 'initial'
    t,x,u_hist,tp_hist = nonlinear_track_velocity(np.array([chs,whs]),np.array([speed1,climb_angle1]), 
                                          np.array([speed2,climb_angle2]),data_file, tf,switch_time=st)
    plotit(t,x,u_hist,ref=tp_hist,model=model_type)

    for opt_path in models:

        model_type = opt_path.split('/')[-1][3:]
        print(model_type)
        output_hdf5_file = f"{opt_path}/opt_vars.hdf5"
        x = extract_dataset_from_results(output_hdf5_file, "x")
        chs = x[0]
        whs = x[1]

        t,x,u_hist,tp_hist = nonlinear_track_velocity(np.array([chs,whs]),np.array([speed1,climb_angle1]), 
                                          np.array([speed2,climb_angle2]),data_file, tf,switch_time=st)
        plotit(t,x,u_hist,ref=tp_hist,model=model_type)

