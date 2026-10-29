
import numpy as np

N_CHUNKS = 10

merged = {}
for c in range(N_CHUNKS):
    d = np.load(f'Aero_data/Parallel Runs/aero_data{c}.npz')
    for k in d.files:
        if k in ('ML','PL','CHSL','WHSL','ERL','WSL','THSL','WDL'):
            merged[k] = d[k]
        else:
            merged[k] = merged.get(k, np.zeros_like(d[k])) + d[k]
np.savez_compressed('Aero_data/aero_data_merged.npz', **merged)