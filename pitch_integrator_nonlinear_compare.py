# Pitch integrator nonlinear controller

import numpy as np
from utils.nonlinear_model import nonlinear_track_pitch
from utils.read_hdf5 import extract_dataset_from_results
from utils.plotters import niceplot_3_models_alt
from utils.transient_errors import transient_errors

if __name__ == "__main__":
    # Test the nonlinear model by integrating it forward in time.

    angle = 115

    opt_path1 = 'Output/5_dec_vars/Gramian_comparison/inf_set'
    opt_path2 = 'Output/5_dec_vars/Gramian_comparison/gramian_eigs'
    models = [opt_path1, opt_path2]

    speed1 = 190
    climb_angle1 = 10
    speed2 = 210
    climb_angle2 = 0

    # Single consolidated dictionary
    sim_data = {}

    data_file = 'Aero_data/aero_merged_cg_17.npz'
    tf = 20
    st = 10

    # Initial
    model_type = 'initial'
    chs = 5  # center halfspan in m
    whs = 12  # wing halfspan in m
    ws = 35
    ths = 4.9
    wd = 10
    design = np.array([chs,whs,ws,ths,wd])

    t, x, u_hist, tp_hist = nonlinear_track_pitch(design, np.array([speed1, climb_angle1]), 
                                                   np.array([speed2, climb_angle2]), data_file, tf, switch_time=st)

    sim_data['initial'] = {
        't': t,
        'x': x,
        'u': u_hist,
        'tp': tp_hist,
        'style': 'dashed',
        'color': 'orange',
        'name': 'initial'
    }

    sim_data['ref'] = {
        't':t,
        'x':tp_hist,
        'style':'dotted',
        'color':'purple'
    }

    for opt_path in models:
        
        model_type = opt_path.split('/')[-1]
        output_hdf5_file = f"{opt_path}/opt_vars.hdf5"
        x_opt = extract_dataset_from_results(output_hdf5_file, "x")

        print("Below, model type is:")
        print(model_type)
        print(x_opt)
        t, x, u_hist, tp_hist = nonlinear_track_pitch(x_opt, np.array([speed1, climb_angle1]), 
                                                       np.array([speed2, climb_angle2]), data_file, tf, switch_time=st)

        if model_type == 'inf_set':
            key = model_type
            color = 'salmon'
        elif model_type == 'gramian_eigs':
            key = model_type
            color = 'teal'

        sim_data[model_type] = {
            't': t,
            'x': x,
            'u': u_hist,
            'tp': tp_hist,
            'style': 'solid',
            'color': color,
            'name':key
        }

    transient_errors(sim_data, switch_time = st)

    niceplot_3_models_alt(sim_data,['inf_set','gramian_eigs'],save=False,savename='nonlinear_gramian',compare=True)
