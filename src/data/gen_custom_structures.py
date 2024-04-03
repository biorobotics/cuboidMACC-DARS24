import numpy as np
import sys

lookup = [ # 8 entries
    {'x': 1, 'y': 1, 'z': 0, 'color': 1, 'size': 3, 'theta': 0},  # voxel 0
    {'x': 1, 'y': 4, 'z': 0, 'color': 1, 'size': 3, 'theta': 0},  # voxel 1
    {'x': 1, 'y': 1, 'z': 1, 'color': 1, 'size': 3, 'theta': 0},  # voxel 2
    {'x': 2, 'y': 1, 'z': 2, 'color': 1, 'size': 3, 'theta': 1},  # voxel 3
    {'x': 0, 'y': 1, 'z': 2, 'color': 1, 'size': 3, 'theta': 1},  # voxel 4
    {'x': 1, 'y': 4, 'z': 1, 'color': 1, 'size': 3, 'theta': 0},  # voxel 5
    {'x': 2, 'y': 4, 'z': 2, 'color': 1, 'size': 3, 'theta': 1},  # voxel 6
    {'x': 0, 'y': 4, 'z': 2, 'color': 1, 'size': 3, 'theta': 1},  # voxel 7
]
structure = []
for l in lookup:
    structure.append([l['color'], l['size'], l['theta'], l['x'], l['y'], l['z']])
print(structure)
# save the state to a file pickle
np.save('/Users/shambhavisingh/Documents/GitHub/macc_general_mamba/src/data/custom/bridge_7x7x4.npy', structure)