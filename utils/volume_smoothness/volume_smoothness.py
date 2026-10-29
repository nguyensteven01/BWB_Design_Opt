''' Plot the behavior of the volume of the reachable set in a neighborhood of one of the optimal points'''

from design_optimization import set_hull
import numpy as np
import matplotlib.pyplot as plt
import time
# pick a point

x = np.array([6, 11])
dx = np.array([1, 1])
V0 = 200
num_pts = 15

a = np.linspace(x[0]-dx[0],x[0]+dx[0],num_pts)
b = np.linspace(x[1]-dx[1],x[1]+dx[1],num_pts)

A,B = np.meshgrid(a,b,indexing='ij')
# volumes = np.zeros((num_pts,num_pts))
# start_time = time.perf_counter()
# for i in range(num_pts):
#     for j in range(num_pts):
#         x = np.array([A[i,j],B[i,j]])
#         print(x)
#         hull = set_hull(x,V0)
#         vol = hull.volume
#         volumes[i,j] = vol

# end_time = time.perf_counter()
# data = {'volumes':volumes}
# volumes_data = np.savez_compressed('volume_smoothness/volume_data2',**data)
# print(end_time-start_time)

data = np.load('volume_smoothness/volume_data2.npz')
volumes = data['volumes']

# Creating figure
plt.rcParams.update({'font.size': 15})
plt.rcParams['lines.linewidth'] = 4
plt.rcParams["legend.handlelength"] = 3.5
plt.rcParams['lines.markersize'] = 12

fig = plt.figure(figsize =(14, 9))
ax = plt.axes(projection ='3d')


surf = ax.plot_surface(A,B,volumes, cmap="viridis")

fig.colorbar(surf, ax=ax, shrink=0.5, label="Volume")
ax.set_title('Volume of reachable set for varying design parameters')
ax.set_xlabel("Center half-span", labelpad=15)
ax.set_ylabel("Wing half-span", labelpad=15)
ax.set_zlabel("Volume", labelpad=10)

ax.set_yticks(np.arange(10, 12.2, 0.5))
ax.set_xticks(np.arange(5,7.2, 0.5))

ax.view_init(elev=10, azim=-131, roll=0)
plt.savefig('volume_smoothness/volume_smoothness.png',dpi=300)
plt.show()
print(A)
print(B)
