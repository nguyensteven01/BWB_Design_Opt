import numpy as np
import matplotlib.pyplot as plt
from textwrap import wrap

aero_arrays = np.load('Aero_data/aero_7dim2.npz')

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
         "elevator rotation":4}
array_names = array_order.copy()
array_names["speed"] = ML
array_names["pitch"] = PL
array_names["center halfspan"] = CHSL
array_names["wing halfspan"] = WHSL
array_names["elevator rotation"] = ERL

# if randomize set to False, then all constant parameters will just take their minimum value (essentially, set their index to 0)
randomize = False
sweep_pair = np.array(["pitch","elevator rotation"]) # The two parameters we want to vary in the 2D plot

# indices = np.zeros(ndims)
for i in sweep_pair:
    indices[array_order[i]] = slice(None)

# If we choose to, this will randomize what value each constant parameter is set to.
if randomize:
    for i in range(nparams):
        if indices[i] == 0:
            indices[i] = np.random.randint(0,ndims-1)


# These worked for the old, 6 parameter data
# Pitch to 5 deg
# indices[2] = 2

# speed to 200 m/s
# indices[0] = 3

#chs ~5 m
# indices[2] = 3

# whs ~12 m
# indices[3] = 1

indices = tuple(indices)

#%% Plot the data
X = array_names[sweep_pair[0]][indices]
Y = array_names[sweep_pair[1]][indices]


Z = Fx[indices]/1000
output_label = "Forces in $x$-axis [kN]"

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    'font.size': 15,
})

print(X)
print(Y)
print(Z)

# print(f'Altitude: {AL[tuple(indices)]}')
print(f'Speed: {ML[tuple(indices)]}')
print(f'Pitch: {PL[tuple(indices)]}')
print(f'CHS: {CHSL[tuple(indices)]}')
print(f'WHS: {WHSL[tuple(indices)]}')
print(f'ER: {ERL[tuple(indices)]}')

plt.figure(figsize=(8, 6))
im = plt.pcolormesh(X, Y, Z, shading='auto', cmap='viridis')

plt.xlabel(sweep_pair[0].title())
plt.ylabel(sweep_pair[1].title())
plt.title("\n".join(wrap(f'{output_label} as function of {sweep_pair[0]} and {sweep_pair[1]}')))
plt.colorbar(im, label=f'{output_label}')

plt.tight_layout()
plt.show()