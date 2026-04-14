# Pitch integrator nonlinear controller

import numpy as np
from nonlinear_model import nonlinear_track_pitch
from read_hdf5 import extract_dataset_from_results
from plotters import niceplot_one_at_a_time, error_bars

if __name__ == "__main__":
    # Test the nonlinear model by integrating it forward in time.

    # Initial
    chs = 5  # center halfspan in m
    whs = 12  # wing halfspan in m
    model_type = 'initial'

    opt_path3 = 'Output/aero_7dim_asym/N6_volmax_110con'
    opt_path1 = 'Output/aero_7dim_asym/N6_volmax'
    opt_path2 = 'Output/aero_7dim_asym/N6_110max'
    models = [opt_path1, opt_path2, opt_path3]

    speed1 = 190
    climb_angle1 = 10
    speed2 = 210
    climb_angle2 = 0

    # Single consolidated dictionary
    sim_data = {}
    
    data_file = 'Aero_data/aero_7dim2.npz'
    tf = 40  # seconds
    st = 20  # switching time

    #%% Testing pitch angle tracker

    model_type = 'initial'
    t, x, u_hist, tp_hist = nonlinear_track_pitch(np.array([chs, whs]), np.array([speed1, climb_angle1]), 
                                                   np.array([speed2, climb_angle2]), data_file, tf, switch_time=st)

    sim_data['initial'] = {
        't': t,
        'x': x,
        'u': u_hist,
        'tp': tp_hist,
        'style': 'dashed',
        'color': 'orange'
    }

    sim_data['ref'] = {
        't':t,
        'x':tp_hist,
        'style':'dotted',
        'color':'purple'
    }

    for opt_path in models:

        model_type = opt_path.split('/')[-1][3:]
        print(model_type)
        output_hdf5_file = f"{opt_path}/opt_vars.hdf5"
        x_opt = extract_dataset_from_results(output_hdf5_file, "x")
        chs = x_opt[0]
        whs = x_opt[1]

        t, x, u_hist, tp_hist = nonlinear_track_pitch(np.array([chs, whs]), np.array([speed1, climb_angle1]), 
                                                       np.array([speed2, climb_angle2]), data_file, tf, switch_time=st)

        if model_type == 'volmax':
            key = 'vm'
            color = 'red'
        elif model_type == '110max':
            key = 'dm'
            color = 'blue'
        elif model_type == 'volmax_110con':
            key = 'vmdc'
            color = 'green'

        sim_data[model_type] = {
            't': t,
            'x': x,
            'u': u_hist,
            'tp': tp_hist,
            'style': 'solid',
            'color': color,
            'name':key
        }

    error_bars(sim_data)

    for key, item in sim_data.items():
        if key != 'initial' and key != 'ref':
            niceplot_one_at_a_time(sim_data,key)