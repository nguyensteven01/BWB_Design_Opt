import numpy as np
import matplotlib.pyplot as plt
from textwrap import wrap
from scipy.interpolate import RegularGridInterpolator
import math


#%% Build function to construct interpolators

def aero_interps(data_file):
    ''' This function will take in the name of the data .npz file and output 
    interpolators to query the tabulated data at various points'''

    aero_arrays = np.load(data_file)

    PL = aero_arrays['PL']
    WHSL = aero_arrays['WHSL']
    CHSL = aero_arrays['CHSL']
    # AL = aero_arrays['AL']
    ERL = aero_arrays['ERL']
    ML = aero_arrays['ML']
    if 'WSL' in aero_arrays:
        extra_dec_vars = True
    else:
        extra_dec_vars = False
    
    if extra_dec_vars:
        WSL = aero_arrays['WSL'] # wing sweep list
        THSL = aero_arrays['THSL'] # transition half span list
        WDL = aero_arrays['WDL'] # wing dihedral list

    X_v = aero_arrays['X_v']
    X_alpha = aero_arrays['X_alpha']
    X_deltae = aero_arrays['X_deltae']
    Z_v = aero_arrays['Z_v']
    Z_alpha = aero_arrays['Z_alpha']
    Z_deltae = aero_arrays['Z_deltae']
    M_v = aero_arrays['M_v']
    M_alpha = aero_arrays['M_alpha']
    M_deltae = aero_arrays['M_deltae']

    Fx = aero_arrays['Fx_mat']
    Fz = aero_arrays['Fz_mat']
    M = aero_arrays['M_mat']
    CL = aero_arrays['CL_mat']
    Di = aero_arrays['Di']
    L = aero_arrays['L']
    Cdi = aero_arrays['Cdi_mat']

    # Generate the 6 vectors passed into meshgrid.
    shape = PL.shape
    # print(shape)
    nparams = len(shape)
    sweep_resolution = shape[0] #Assume grid same resolution all axes
    # print(sweep_resolution)
    # print(extra_dec_vars)

    if extra_dec_vars:
        ''' Using additional 3 design variables, all have same resolution except for wing dihedral.'''
        speed_list = np.linspace(ML[0,0,0,0,0,0,0,0],  ML[-1,0,0,0,0,0,0,0],sweep_resolution)
        pitch_list = np.linspace(PL[0,0,0,0,0,0,0,0],  PL[0,-1,0,0,0,0,0,0],sweep_resolution)
        chs_list = np.linspace(CHSL[0,0,0,0,0,0,0,0],CHSL[0,0,-1,0,0,0,0,0],sweep_resolution)
        whs_list = np.linspace(WHSL[0,0,0,0,0,0,0,0],WHSL[0,0,0,-1,0,0,0,0],sweep_resolution)
        er_list = np.linspace(  ERL[0,0,0,0,0,0,0,0], ERL[0,0,0,0,-1,0,0,0],sweep_resolution)
        ws_list = np.linspace(  WSL[0,0,0,0,0,0,0,0], WSL[0,0,0,0,0,-1,0,0],sweep_resolution)
        ths_list = np.linspace(THSL[0,0,0,0,0,0,0,0],THSL[0,0,0,0,0,0,-1,0],sweep_resolution)
        wd_list = np.linspace(  WDL[0,0,0,0,0,0,0,0], WDL[0,0,0,0,0,0,0,-1],2)
        interp_grid = (speed_list, pitch_list, chs_list, whs_list, er_list, ws_list, ths_list, wd_list)
        
    else:
        ''' Below is the approach used for 2 design variables in the AIAA paper.'''
        speed_list = np.linspace(ML[0,0,0,0,0],ML[-1,0,0,0,0],sweep_resolution)
        pitch_list = np.linspace(PL[0,0,0,0,0],PL[0,-1,0,0,0],sweep_resolution)
        chs_list = np.linspace(CHSL[0,0,0,0,0],CHSL[0,0,-1,0,0],sweep_resolution)
        whs_list = np.linspace(WHSL[0,0,0,0,0],WHSL[0,0,0,-1,0],sweep_resolution)
        er_list = np.linspace(ERL[0,0,0,0,0],  ERL[0,0,0,0,-1],sweep_resolution)
        interp_grid = (speed_list, pitch_list, chs_list, whs_list, er_list)
    # print(interp_grid)
    xv_interp = RegularGridInterpolator(interp_grid, X_v)
    xalpha_interp = RegularGridInterpolator(interp_grid, X_alpha)
    xdeltae_interp = RegularGridInterpolator(interp_grid, X_deltae)
    zv_interp = RegularGridInterpolator(interp_grid, Z_v)
    zalpha_interp = RegularGridInterpolator(interp_grid, Z_alpha)
    zdeltae_interp = RegularGridInterpolator(interp_grid, Z_deltae)
    mv_interp = RegularGridInterpolator(interp_grid, M_v)
    malpha_interp = RegularGridInterpolator(interp_grid, M_alpha)
    mdeltae_interp = RegularGridInterpolator(interp_grid, M_deltae)

    Fx_interp = RegularGridInterpolator(interp_grid, Fx)
    Fz_interp = RegularGridInterpolator(interp_grid, Fz)
    M_interp = RegularGridInterpolator(interp_grid, M)
    CL_interp = RegularGridInterpolator(interp_grid, CL)
    Di_interp = RegularGridInterpolator(interp_grid, Di)
    L_interp = RegularGridInterpolator(interp_grid, L)
    Cdi_interp = RegularGridInterpolator(interp_grid, Cdi)

    interp_dict = {'xv':xv_interp, 'xalpha':xalpha_interp, 'xdeltae':xdeltae_interp,
                   'zv':zv_interp, 'zalpha':zalpha_interp, 'zdeltae':zdeltae_interp,
                   'mv':mv_interp, 'malpha':malpha_interp, 'mdeltae':mdeltae_interp,
                   
                   'Fx':Fx_interp, 'Fz':Fz_interp, 'M':M_interp, 'CL':CL_interp,
                   'Di':Di_interp, 'L':L_interp, 'Cdi':Cdi_interp}

    return interp_dict

def build_AB(interpolators, flight_point):
    ''' This function will take in the interpolators and return the A,B matrices
    in our linear-longitudinal model.
    
    Inputs: interpolators is a dictionary of the interpolators created in deriv_interps
            flight_point is an array of the 6 state/control variables at the flight point.
            
            More specifically, flight_point gives [V, alpha, chs, whs, er]
            In a lot of my examples, we assume 0 flight path angle, so alpha = pitch. But really,
            The angle of attack is what the panel method code cares about, and that's what should be passed
            into the interpolators.

    Outputs: A,B matrices at the selected point'''

    X_v = interpolators['xv'](flight_point)
    X_alpha = interpolators['xalpha'](flight_point)
    X_deltae = interpolators['xdeltae'](flight_point)
    Z_v = interpolators['zv'](flight_point)
    Z_alpha = interpolators['zalpha'](flight_point)
    Z_deltae = interpolators['zdeltae'](flight_point)
    M_v = interpolators['mv'](flight_point)
    M_alpha = interpolators['malpha'](flight_point)
    M_deltae = interpolators['mdeltae'](flight_point)

    Z_q = 0 # cannot take derivs w.r.t. pitch rate
    M_q = 0

    A = np.zeros((4, 4))
    B = np.zeros((4, 2)) # two inputs: thrust and elevator deflection

    g = 9.81  # m/s^2
    gamma_0 = 0  # Flight path angle at trim (radians), we assume level flight
    V_0 = flight_point[0]  # Trim airspeed (m/s)

    A[0, 0] = X_v
    A[0, 1] = X_alpha
    A[0, 2] = 0
    A[0, 3] = -g * math.cos(gamma_0)

    A[1, 0] = Z_v / V_0
    A[1, 1] = Z_alpha/ V_0
    A[1, 2] = 1 + Z_q/ V_0
    A[1,3] = -g * math.sin(gamma_0) / V_0

    A[2, 0] = M_v
    A[2, 1] = M_alpha
    A[2, 2] = M_q
    A[2, 3] = 0

    A[3,:] = np.array([0, 0, 1, 0])

    thrust_force = 96000 # lb
    thrust_force = thrust_force * 4.44822 # convert to N
    weight = 331522 # lb
    weight = weight * 4.44822 # convert to N
    mass = weight / g  # kg

    B[0,0] = thrust_force/mass # for thrust input from 0 to 1, 1 is maximum acceleration forward.
    B[1:,0] = np.zeros((3,))
    B[0, 1] = X_deltae
    B[1, 1] = Z_deltae / V_0
    B[2, 1] = M_deltae
    B[3, 1] = 0

    return A,B


if __name__ == "__main__":
    data_file = 'Aero_data/aero_7dim2.npz'
    interp_dict = aero_interps(data_file)
    print(interp_dict)

    
