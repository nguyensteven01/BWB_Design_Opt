''' This script will compute trim '''

import numpy as np
from interpolatory_model import aero_interps
import math
from functools import partial
from scipy import optimize


def compute_trim_forces(flight_point,data_file, verbose=False):
    '''This function computes the trim condition for a given flight point.
    As an input, the flight_point will contain velocity, center halfspan, and wing halfspan.
    It will also take the aero data file as an input.
    
    It will output the pitch angle, elevator deflection, and thrust throttle for trim.'''

    # data_file = 'Aero_data/aero_7dim2.npz'
    interp_dict = aero_interps(data_file)

    Fx = interp_dict['Fx']
    Fz = interp_dict['Fz']
    M = interp_dict['M']

    thrust_force = 96000 # lb
    maxT = thrust_force * 4.44822 * 1.5 # convert to N
    weight = 331522 # lb
    weight = weight * 4.44822 # convert to N
    mass = weight / 9.81  # kg

    x0 = np.asarray([np.deg2rad(6),np.deg2rad(10),0.3]) # initial guess: 6 deg pitch, 10 deg elev, 30% thrust

    # Each of these force/moment functions will take x = [pitch, elev, thrust] as inputs...
    def x_con(x):
        deltaT = x[2]
        full_flight_point = np.concatenate((flight_point[0:1],x[0:1],flight_point[1:],x[1:2]))
        return (Fx(full_flight_point) + deltaT * maxT * math.cos(x[0])) * 1e-5  

    def z_con(x):
        deltaT = x[2]
        full_flight_point = np.concatenate((flight_point[0:1],x[0:1],flight_point[1:],x[1:2]))
        return (-Fz(full_flight_point) - weight + deltaT * maxT * math.sin(x[0])) * 1e-4
    
    def m_con(x):
        full_flight_point = np.concatenate((flight_point[0:1],x[0:1],flight_point[1:],x[1:2]))
        return  M(full_flight_point) * 1e-5
    
    def full_obj(x):
        forces = np.asarray([x_con(x), z_con(x), m_con(x)])
        return forces.flatten()
    
    sol = optimize.fsolve(func = full_obj, x0 = x0)

    if np.max(np.abs(full_obj(sol))) > 1e-2:
        print('Warning: Trim solution may not have converged well enough!')

    if verbose:
        print(f'Forces in x: {x_con(sol) * 1e5} N')
        print(f'Forces in z: {z_con(sol) * 1e4} N')
        print(f'Pitching moment: {m_con(sol) * 1e5} Nm')
        y = [np.rad2deg(sol[0]), np.rad2deg(sol[1]), sol[2]]
        print(f'Trim solution: pitch = {y[0]:.2f} deg, elev = {y[1]:.2f} deg, throttle = {y[2]:.2f}')

    return sol


def compute_trim_eom(flight_point,data_file,climb_angle,verbose=False):
    ''' This function computes trim using nonlinear EOM. It sets V_dot, alpha_dot, q_dot to 0
    and also constrains the flight path angle. Trim variables are aoa, elevator, thrust, pitch.
    Note that bounds on the values of these quantities are not set.
    Inputs: flight_point is [velocity, chs, whs]
            data_file is the aero data file
            climb_angle is the desired climb angle in degrees (positive for climb, negative for descent)
            verbose is for printing out constraint values
    Outputs: trimmed [aoa, elev, thrust, pitch]'''
    interp_dict = aero_interps(data_file)
    V0 = flight_point[0]
    M = interp_dict['M']
    Di = interp_dict['Di']
    L = interp_dict['L']

    # h0 = 10      # alt in km
    # full flight point will modify fp argument so it's [vel, aoa, chs, whs, elevator]

    thrust_force = 96000 # lb
    maxT = thrust_force * 4.44822 * 1.5 # convert to N
    weight = 331522 # lb
    weight = weight * 4.44822 # convert to N
    mass = weight / 9.81  # kg
    g = 9.81

    x0 = np.asarray([np.deg2rad(5),np.deg2rad(5),0.5, np.deg2rad(5)]) # initial guess: 5 deg aoa, 5 deg elev, half thrust, 5 deg pitch

    # Constraints based on EOM instead of force/moment balance.
    def vdot_con(x):
        # x is [alpha, elev, thrust, pitch]
        deltaT = x[2]
        Ft = deltaT * maxT
        full_flight_point = np.concatenate((flight_point[0:1],x[0:1],flight_point[1:],x[1:2]))
        return ( Ft/mass * np.cos(x[0]) - Di(full_flight_point)/mass - g * np.sin(x[3] - x[0]) ) * 1

    def alpha_con(x):
        deltaT = x[2]
        alpha = x[0]
        theta = x[3]
        Ft = deltaT * maxT
        full_flight_point = np.concatenate((flight_point[0:1],x[0:1],flight_point[1:],x[1:2]))
        return (-Ft/(mass * V0) * np.sin(alpha) - L(full_flight_point)/(mass * V0) + g/V0 * np.cos(theta - alpha) ) * 1e3

    def q_con(x):
        deltaT = x[2]
        alpha = x[0]
        theta = x[3]
        Ft = deltaT * maxT
        full_flight_point = np.concatenate((flight_point[0:1],x[0:1],flight_point[1:],x[1:2]))
        return M(full_flight_point) * 1e-5

    def pathangle_con(x):
        return np.asarray([x[3] - x[0] - np.deg2rad(climb_angle)])

    def full_obj(x):
        forces = np.asarray([vdot_con(x), alpha_con(x), q_con(x), pathangle_con(x)])
        return forces.flatten()

    bnds = ((np.deg2rad(-5),np.deg2rad(15)), # AOA bounds
        (np.deg2rad(-30),np.deg2rad(30)), # elev bounds
        (0.0,1.0),                       # thrust bounds
        (np.deg2rad(-20),np.deg2rad(60)))  # pitch bounds
    
    cons = ({'type':'eq','fun':vdot_con},
        {'type':'eq','fun':alpha_con},
        {'type':'eq','fun':q_con},
        {'type':'eq','fun':pathangle_con})
    
    def penalty_obj(x):
        return vdot_con(x)**2 + alpha_con(x)**2 + q_con(x)**2 + pathangle_con(x)**2
    
    best = {"x": None, "cost": np.inf}

    def callback(x):
        cost = penalty_obj(x)
        if cost < best["cost"]:
            best["cost"] = cost
            best["x"] = x.copy() 
    
    ops = {'maxiter':200}
    sol = optimize.minimize(fun=penalty_obj, x0=x0, constraints=cons, bounds=bnds, options=ops, callback=callback)
    if not sol.success:
        print('Warning, failed to trim!')

    if verbose:
        y = [np.rad2deg(sol.x[0]), np.rad2deg(sol.x[1]), sol.x[2], np.rad2deg(sol.x[3])]
        print(y)
        print(vdot_con(sol.x))
        print(alpha_con(sol.x))
        print(q_con(sol.x))
        print(pathangle_con(sol.x))

    # Solving with fsolve

    # sol = optimize.fsolve(func = full_obj, x0 = x0)
    
    # if verbose:
    #     y = [np.rad2deg(sol[0]), np.rad2deg(sol[1]), sol[2], np.rad2deg(sol[3])]
    #     print(y)
    #     print(vdot_con(sol))
    #     print(alpha_con(sol))
    #     print(q_con(sol))
    #     print(pathangle_con(sol))

    return sol.x

if __name__ == "__main__":
    v0 = 200 # speed in m/s
    chs = 5 # center halfspan in m
    whs = 12 # wing halfspan in m

    # chs = 5
    # whs = 16
    

    flight_point = np.asarray([v0,chs,whs])
    data_file = 'Aero_data/aero_7dim2.npz'
    trimx = compute_trim_eom(flight_point, data_file, 0, verbose=False)
    print(trimx)