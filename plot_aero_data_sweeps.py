import numpy as np
import matplotlib.pyplot as plt
from textwrap import wrap

saveit = 0

aero_arrays = np.load('Aero_data/aero_merged_cg_17.npz')

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
# Cdi = aero_arrays['Cdi_mat']
WSL = aero_arrays['WSL'] # wing sweep list
THSL = aero_arrays['THSL'] # transition half span list
WDL = aero_arrays['WDL'] # wing dihedral list

# --- Fix other dimensions (if more than 2D) ---
shape = PL.shape
nparams = len(shape)
ndims = shape[0]

# Index to pick out the slice of data we want to inspect. By default, slice(None) in any axis will slice nothing (take all).
# We want to replace these with constant indices corresponding to the parameters we hold constant.
indices = [0] * nparams

#%%
'''Based on the order of the meshgrid, the indices to vary will follow this order.
Setting sweep_pair to list the names of the variables we want to vary, array_order tells us the indices that need to be
set to slice(None). array_names tells us the corresponding matrix for when we plot the data later'''

array_order = {"speed":0,
         "pitch":1,
         "center halfspan":2,
         "wing halfspan":3,
         "elevator rotation":4,
         "wing sweep":5,
         "trans halfspan":6,
         "wing dihedral":7}
array_names = array_order.copy()
array_names["speed"] = ML
array_names["pitch"] = PL
array_names["center halfspan"] = CHSL
array_names["wing halfspan"] = WHSL
array_names["elevator rotation"] = ERL
array_names["wing sweep"] = WSL
array_names["trans halfspan"] = THSL
array_names["wing dihedral"] = WDL

# if randomize set to False, then all constant parameters will just take their minimum value (essentially, set their index to 0)
randomize = False
sweep_pair = np.array(["pitch","elevator rotation"]) # The two parameters we want to vary in the 2D plot

# indices = np.zeros(ndims)
for i in sweep_pair:
    indices[array_order[i]] = slice(None)

# If we choose to, this will randomize what value each constant parameter is set to.
if randomize:
    for i in range(nparams-1):
        if indices[i] == 0:
            indices[i] = np.random.randint(0,ndims-1)
    indices[-1] = np.random.randint(0,2)

# speed to 200 m/s
indices[0] = 2

indices[5] = 2 # wing sweep 32.5 deg
indices[6] = 2 # ths 5.5 m
indices[7] = 0 # wing dih 10 deg

#chs
indices[2] = 2

# whs
indices[3] = 1

indices = tuple(indices)
print(f'Indices: {indices}')

#%% Plot the data
X = np.rad2deg(array_names[sweep_pair[0]][indices])
Y = np.rad2deg(array_names[sweep_pair[1]][indices])

Z = Fx[indices]/1000
output_label = '$F_x$'

rho = M_alpha * -Z_deltae / (M_deltae * (Di - Z_alpha))

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    'font.size': 15,
})

plt.figure(figsize=(8, 6))
im = plt.pcolormesh(X, Y, Z, shading='auto', cmap='viridis')

if saveit:
    plt.ylabel('Elevator rotation angle, $\delta_e$ [deg]')
    plt.xlabel('Pitch angle, $\\theta$ [deg]')
    plt.colorbar(im, label='Forces in $x$-axis, $F_x$ [kN]')
    plt.savefig('aero_sample.png', dpi=300)
else:
    plt.xlabel(sweep_pair[0].title())
    plt.ylabel(sweep_pair[1].title())
    plt.title("\n".join(wrap(f'{output_label} as function of {sweep_pair[0]} and {sweep_pair[1]}')))
    plt.colorbar(im, label=f'{output_label} [kN]')

plt.tight_layout()
plt.show()