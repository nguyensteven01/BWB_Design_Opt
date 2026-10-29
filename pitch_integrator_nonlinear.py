# Pitch integrator nonlinear controller

import numpy as np
from utils.nonlinear_model import nonlinear_track_pitch
from utils.read_hdf5 import extract_dataset_from_results
from utils.plotters import error_bars, niceplot_3_models_alt
from utils.transient_errors import transient_errors

from animate.compare_sim_data import animate_sim_data_comparison

if __name__ == "__main__":
    # Test the nonlinear model by integrating it forward in time.

    savefig = False
    make_video = False
    make_plots = True

    angle = 115

    sm = 0.02
    if sm == 0:
        opt_path1 = 'Output/5_dec_vars/N6_volmax'
        opt_path2 = 'Output/5_dec_vars/N6_115max'
        opt_path3 = 'Output/5_dec_vars/N6_volmax_115con'
        models = [opt_path1, opt_path2, opt_path3]
    elif sm == 0.04:
        opt_path1 = 'Output/5_dec_vars/10-22-26/N6_volmax_sm4'
        opt_path2 = 'Output/5_dec_vars/10-22-26/N6_90max_sm4'
        # opt_path2 = 'Output/5_dec_vars/10-29-26/N6_115max_sm4'
        models = [opt_path1, opt_path2]
    elif sm == 0.02:
        opt_path1 = 'Output/5_dec_vars/10-23-26/N6_volmax_sm2'
        opt_path2 = 'Output/5_dec_vars/10-23-26/N6_115max_sm2'
        # opt_path3 = 'Output/5_dec_vars/10-23-26/N6_90max_sm2'
        models = [opt_path1, opt_path2]
    elif sm == 0.01:
        opt_path1 = 'Output/5_dec_vars/10-21-26/N6_volmax_sm1'
        opt_path2 = 'Output/5_dec_vars/10-21-26/N6_115max_sm1'
        models = [opt_path1, opt_path2]
    elif sm == 0.03:
        opt_path1 = 'Output/5_dec_vars/10-19-26/N6_volmax_sm3'
        opt_path2 = 'Output/5_dec_vars/10-19-26/N6_115max_sm3'
        models = [opt_path1, opt_path2]
    elif sm == 0.01:
        opt_path1 = 'Output/5_dec_vars/10-21-26/N6_volmax_sm1'
        opt_path2 = 'Output/5_dec_vars/10-21-26/N6_115max_sm1'
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
        
        model_type = opt_path.split('/')[-1][3:]
        output_hdf5_file = f"{opt_path}/opt_vars.hdf5"
        x_opt = extract_dataset_from_results(output_hdf5_file, "x")

        print("Below, model type is:")
        print(model_type)
        print(x_opt)
        t, x, u_hist, tp_hist = nonlinear_track_pitch(x_opt, np.array([speed1, climb_angle1]), 
                                                       np.array([speed2, climb_angle2]), data_file, tf, switch_time=st)

        if model_type == f'volmax_sm2':
            key = 'vm'
            color = 'red'
        elif model_type == f'{angle}max_sm2':
            key = 'dm'
            color = 'blue'
        elif model_type == f'volmax_{angle}con':
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

    transient_errors(sim_data, switch_time = st)

    # error_bars(sim_data,save=savefig)
    if make_plots:
        print(sim_data.keys())
        # niceplot_3_models_alt(sim_data, ['initial',f'volmax',f'{angle}max',f'volmax_{angle}con'], save=savefig)
        niceplot_3_models_alt(sim_data, ['initial',f'volmax_sm2',f'{angle}max_sm2'], save=savefig)




