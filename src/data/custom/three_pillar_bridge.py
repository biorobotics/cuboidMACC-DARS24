widthGrid = 10
heightGrid = 4
depthGrid = 10

# color corresponds to size: 0 = cube 1x1x1, 1 = cuboid 3x1x1, 2 = cuboid 5x1x1

# orientation = 0 means block varies along x-axis {x-1, x, x+1}
# orientation = 1 means block varies along y-axis {y-1, y, y+1}

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