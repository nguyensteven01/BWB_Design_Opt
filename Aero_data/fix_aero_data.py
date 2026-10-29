# Fix Aero Data

import numpy as np
import matplotlib.pyplot as plt
from textwrap import wrap

saveit = 1


aero_arrays = np.load('Aero_data/aero_data_merged_forces_fixed.npz')

    

PL = aero_arrays['PL']
WHSL = aero_arrays['WHSL']
CHSL = aero_arrays['CHSL']
ERL = aero_arrays['ERL']
ML = aero_arrays['ML']
X_v = aero_arrays['X_v']
X_alpha = aero_arrays['X_alpha']
X_deltae = aero_arrays['X_deltae']
M_alpha = aero_arrays['M_alpha']
M_v = aero_arrays['M_v']
M_deltae = aero_arrays['M_deltae']
Z_alpha = aero_arrays['Z_alpha']
Z_v = aero_arrays['Z_v']
Z_deltae = aero_arrays['Z_deltae']
Fx = aero_arrays['Fx_mat']
Fz = aero_arrays['Fz_mat']
M = aero_arrays['M_mat']
CL = aero_arrays['CL_mat']
Di = aero_arrays['Di']
L = aero_arrays['L']
Cdi = aero_arrays['Cdi_mat']

WSL = aero_arrays['WSL'] # wing sweep list
THSL = aero_arrays['THSL'] # transition half span list
WDL = aero_arrays['WDL'] # wing dihedral list


weight = 331522 # lb
weight = weight * 4.44822 # convert to N
# weight = 1645907.20892072 # N, estimate on 9/30
mass = weight / 9.81 # kg
Iy = 3759014 # slug ft^2
Iy = Iy * 14.5939 * 0.3048**2 # kg m^2

cg_loc = 14


''' Fix the force rotation '''
def fix_forces(fx,fz,alpha):
    # Take in the incorrectly rotated fx, fz
    # fxo is the reconstruction of the panel method fx, fz
    # fxp is the properly rotated one

    fzo = (fz + fx*np.sin(alpha))/np.cos(alpha)
    fxo = (fx - fzo*np.sin(alpha))/np.cos(alpha)

    fxp = fxo*np.cos(alpha) + fzo*np.sin(alpha)
    fzp = -fxo*np.sin(alpha) + fzo*np.cos(alpha)

    return fxp,fzp


# Fx,Fz = fix_forces(Fx,Fz,PL)
# X_v,Z_v = fix_forces(X_v,Z_v,PL)
# X_alpha,Z_alpha = fix_forces(X_alpha,Z_alpha,PL)
# X_deltae,Z_deltae = fix_forces(X_deltae,Z_deltae,PL)

'''Fix moments and derivatives'''

M= M - cg_loc*Fz * np.cos(PL) - cg_loc*Fx*np.sin(PL)     # This works.

M_v = (Iy)* M_v - cg_loc * np.cos(PL)*Z_v*mass - cg_loc*np.sin(PL)*X_v*mass
M_v = M_v/Iy

M_alpha = (Iy) * M_alpha - cg_loc * np.cos(PL)*Z_alpha * mass - cg_loc*np.sin(PL)*X_alpha * mass + cg_loc * np.sin(PL)*Fz - cg_loc*np.cos(PL)*Fx   # not that close ish, but maybe good enough?
M_alpha = M_alpha / (Iy)

M_deltae = Iy* M_deltae - cg_loc * np.cos(PL)*Z_deltae*mass - cg_loc*np.sin(PL)*X_deltae*mass
M_deltae = M_deltae/Iy 

array_to_save = {
    'X_v': X_v,
    'Z_v': Z_v,
    'M_v': M_v,
    'X_alpha': X_alpha,
    'Z_alpha': Z_alpha,
    'M_alpha': M_alpha,
    'X_deltae': X_deltae,
    'Z_deltae': Z_deltae,
    'M_deltae': M_deltae,
    'CL_mat': CL,
    'Fx_mat': Fx,
    'Fz_mat': Fz,
    'M_mat': M,
    'Cdi_mat': Cdi,
    'ML': ML,
    'PL': PL,
    'CHSL': CHSL,
    'WHSL': WHSL,
    'ERL': ERL,
    'WSL': WSL,
    'THSL': THSL,
    'WDL': WDL,
    'Di': Di,
    'L': L
}
if saveit:
    np.savez_compressed(f'Aero_data/aero_merged_cg_{cg_loc}.npz', **array_to_save)