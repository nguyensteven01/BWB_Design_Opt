'''This script will save all 4 [A,B] models to a .mat file for use in MATLAB.
It considers a trim point about 200 m/s airspeed and 0 climb angle'''

from scipy.io import savemat
import numpy as np
from read_hdf5 import extract_dataset_from_results
from design_optimization import build_AB
from trim_aircraft import compute_trim_eom
from interpolatory_model import *

# Set model and set trim points

V0 = 200
ca = 0 # climb angle

chs = 5 # center halfspan in m
whs = 12 # wing halfspan in m
model_type = 'initial'

data_file = 'Aero_data/aero_7dim2.npz'
interp_dict = aero_interps(data_file)

print(model_type)
alpha0, deltae0, deltaT0, pitch0 = compute_trim_eom(np.asarray([V0,chs,whs]), data_file,climb_angle=ca)
print(f'Trimmed alpha: {np.rad2deg(alpha0)}, deltae: {np.rad2deg(deltae0)}, deltaT: {deltaT0}, pitch: {np.rad2deg(pitch0)}')
flight_point = np.array([V0, alpha0, deltae0])
full_flight_point = np.concatenate((flight_point[0:2], np.array([chs, whs]), flight_point[2:]))
A,B = build_AB(interp_dict,full_flight_point)

my_models = {f'A_{model_type}':A,
             f'B_{model_type}':B}

opt_path3 = 'Output/aero_7dim_asym/N6_volmax_110con'
opt_path1 = 'Output/aero_7dim_asym/N6_volmax'
opt_path2 = 'Output/aero_7dim_asym/N6_110max'
models = [opt_path1, opt_path2, opt_path3]

for opt_path in models:
    model_type = opt_path.split('/')[-1][3:]
    print(model_type)
    output_hdf5_file = f"{opt_path}/opt_vars.hdf5"
    x = extract_dataset_from_results(output_hdf5_file, "x")
    chs = x[0]
    whs = x[1]
    alpha0, deltae0, deltaT0, pitch0 = compute_trim_eom(np.asarray([V0,chs,whs]), data_file,climb_angle=ca)
    print(f'Trimmed alpha: {np.rad2deg(alpha0)}, deltae: {np.rad2deg(deltae0)}, deltaT: {deltaT0}, pitch: {np.rad2deg(pitch0)}')
    flight_point = np.array([V0, alpha0, deltae0])

    full_flight_point = np.concatenate((flight_point[0:2], x, flight_point[2:]))
    A,B = build_AB(interp_dict,full_flight_point)

    my_models[f'A_{model_type}']=A
    my_models[f'B_{model_type}']=B

savemat('Output/MATLAB_files/allmodels.mat',my_models)