''' Computing other aircraft metrics'''

from utils.interpolatory_model import *
import numpy as np
from utils.read_hdf5 import extract_dataset_from_results
from utils.trim_aircraft import compute_trim_eom

names = ['init','volmax','115max','volmax_115con']
for i in names:
    print(i)
    if i == 'init':
        x = np.array([5,12,35,4.9,10])
    else:
        output_hdf5_file = "Output/5_dec_vars/N6_" + i + "/opt_vars.hdf5"
        x = extract_dataset_from_results(output_hdf5_file, "x")
    print(x)
    chs = x[0]
    whs = x[1]
    desired_vel = 200
    data_file = 'Aero_data/aero_merged_cg_17.npz'     # MAKE SURE TO PASS THE CORRECT NUMBER OF FLIGHT_POINTS TO compute_trim_eom WHEN USING THIS
    wsl = x[2]
    thsl = x[3]
    wdl = x[4]
        
    alpha0, deltae0, deltaT0, pitch0, _ = compute_trim_eom(np.asarray([desired_vel,chs,whs,wsl,thsl,wdl]), data_file,climb_angle=0)

    interp_dict = aero_interps(data_file)
    # print(f'Trimmed alpha: {np.rad2deg(alpha0)}, deltae: {np.rad2deg(deltae0)}, deltaT: {deltaT0}, pitch: {np.rad2deg(pitch0)}')
    flight_point = np.array([desired_vel, alpha0, deltae0])
    full_flight_point = np.concatenate((flight_point[0:2], x[0:2], flight_point[2:],x[2:]))
    L = interp_dict['L'](full_flight_point)
    D = interp_dict['Di'](full_flight_point)

    print(f"Aero efficiency: {L/D}")
    print(f"Power required at 200 m/s: {D * desired_vel/1e6}")

    # Computing speed at which minimum drag acts

    # def obj(x):
    #     # For speed x, trim for level flight, then compute Cl^1.5 / Cdi
    #     # The speed at which this quantity is max is minimum drag

    #     alpha0, deltae0, deltaT0, pitch0, _ = compute_trim_eom(np.asarray([x[0],chs,whs]), data_file,climb_angle=0, warn_trim=False)

    #     full_flight_point = np.asarray([x[0], alpha0, chs, whs, deltae0])
    #     Cdi = interp_dict['Cdi'](full_flight_point).item()
    #     Cl = interp_dict['CL'](full_flight_point).item()
    #     quantity = Cl**3/Cdi**2
    #     if np.iscomplexobj(quantity):
    #         raise RuntimeError(f"complex objective at x={x}, J={quantity}. Cdi = {Cdi}, CL = {Cl}")
    #     return -quantity