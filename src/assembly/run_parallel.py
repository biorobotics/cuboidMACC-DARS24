from matplotlib import pyplot as plt
import numpy as np
import networkx as nx
import time
from block_utils import occupancy_grid, find_path_to_boundary

def create_time_windows(assembly_sequence):
    robot_locations = []
    action_locations = []
    time_windows = dict()
    for i in range(len(assembly_sequence)):
        action_locations.append(assembly_sequence[i].state[-1][3:6])
        occu_grid = occupancy_grid(assembly_sequence[i].state[:-1],assembly_sequence[i].x_dim,assembly_sequence[i].y_dim,assembly_sequence[i].z_dim)
        _, path = find_path_to_boundary(assembly_sequence[i].state[-1], occu_grid)
        if len(path)>1:
            robot_locations.append(path[1])
        else:
            robot_locations.append(path[0])
        time_windows[i] = {
            'robot': robot_locations[i],
            'action': action_locations[i],
            'path': path
        }

    constraints = []
    for i in range(len(time_windows)):
        for j in range(0, len(time_windows)):
            if time_windows[i]['robot'] == time_windows[j]['robot'] \
            or time_windows[i]['action'] == time_windows[j]['action'] \
            or time_windows[i]['action'] == time_windows[j]['robot'] \
            or time_windows[i]['robot'] == time_windows[j]['action']:
                if i<j:
                    constraints.append((i, j))
                elif i>j:
                    constraints.append((j, i))
            elif time_windows[i]['robot'] in time_windows[j]['path'] \
            or time_windows[i]['action'] in time_windows[j]['path']:
                if i<j:
                    constraints.append((i, j))
                elif i>j:
                    constraints.append((j, i))
    
    G = nx.DiGraph()
    for i in range(len(time_windows)):
        G.add_node(i)
    for c in constraints:
        G.add_edge(c[0], c[1])

    # if (a, b) and (b, c) and (a, c) in G.edges, remove (a, c)
    for i in range(len(time_windows)):
        for j in range(i+1, len(time_windows)):
            if G.has_edge(i, j):
                for k in range(j+1, len(time_windows)):
                    if G.has_edge(j, k) and G.has_edge(i, k):
                        G.remove_edge(i, k)
    return G
def visualize_dependencies(G):
    pos = nx.shell_layout(G)
    nx.draw(G, pos, with_labels=True)
    plt.show()
