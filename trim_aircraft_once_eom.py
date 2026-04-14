# New file to compute trim point.
# Script should take in a desired altitude and speed

import numpy as np
from interpolatory_model import aero_interps
import math
from functools import partial
from scipy import optimize
from pyslsqp.postprocessing import print_dict_as_table

''' This script will attempt to trim the aircraft.
Given desired altitude, speed, and the design parameters, it will solve for the pitch angle and
elevator deflection angle that trim the plane.'''

# data_file = 'Aero_data/aero_5dim_fixedcoord.npz'
# data_file = 'Aero_data/aero_7dim.npz'
data_file = 'Aero_data/aero_7dim2.npz'
interp_dict = aero_interps(data_file)

Xv = interp_dict['xv']
Xalpha = interp_dict['xalpha']
Xdeltae = interp_dict['xdeltae']
Zv = interp_dict['zv']
Zalpha = interp_dict['zalpha']
Zdeltae = interp_dict['zdeltae']
Mv = interp_dict['mv']
Malpha = interp_dict['malpha']
Mdeltae = interp_dict['mdeltae']
Fx = interp_dict['Fx']
Fz = interp_dict['Fz']
M = interp_dict['M']
CL = interp_dict['CL']
Di = interp_dict['Di']
L = interp_dict['L']
Cdi = interp_dict['Cdi']

# h0 = 10      # alt in km
V0 = 200    # speed in m/s
# chs0 = 5    # center halfspan in m
# whs0 = 12   # wing halfspan in m

chs0 = 4
whs0 = 18
climb_angle = 0

# flight_point = np.asarray([h0,V0,chs0,whs0])
flight_point = np.asarray([V0,chs0,whs0])
# full flight point will modify this so it's [vel, aoa, chs, whs, elevator]

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

cons = ({'type':'eq','fun':vdot_con},
        {'type':'eq','fun':alpha_con},
        {'type':'eq','fun':q_con},
        {'type':'eq','fun':pathangle_con})

bnds = ((np.deg2rad(-5),np.deg2rad(15)), # AOA bounds
        (np.deg2rad(-30),np.deg2rad(30)), # elev bounds
        (0.0,1.0),                       # thrust bounds
        (np.deg2rad(-5),np.deg2rad(15)))  # pitch bounds

def full_obj(x):
    forces = np.asarray([vdot_con(x), alpha_con(x), q_con(x), pathangle_con(x)])
    return forces.flatten()

def penalty_obj(x):
    return vdot_con(x)**2 + alpha_con(x)**2 + q_con(x)**2 + pathangle_con(x)**2

best = {"x": None, "cost": np.inf}

def callback(x):
    cost = penalty_obj(x)
    if cost < best["cost"]:
        best["cost"] = cost
        best["x"] = x.copy() 

ops = {'maxiter':200}
sol = optimize.minimize(fun=penalty_obj, x0=x0, constraints=cons, bounds=bnds, options=ops, callback = callback)
if sol.success == False:
    print('Failed to trim!')
y = [np.rad2deg(sol.x[0]), np.rad2deg(sol.x[1]), sol.x[2], np.rad2deg(sol.x[3])]
print(y)
print(vdot_con(sol.x))
print(alpha_con(sol.x))
print(q_con(sol.x))
print(pathangle_con(sol.x))

# Solving with fsolve
sol = optimize.fsolve(func = full_obj, x0 = x0)
y = [np.rad2deg(sol[0]), np.rad2deg(sol[1]), sol[2], np.rad2deg(sol[3])]
print(y)
print(vdot_con(sol))
print(alpha_con(sol))
print(q_con(sol))
print(pathangle_con(sol))
