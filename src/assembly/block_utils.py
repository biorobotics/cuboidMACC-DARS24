import numpy as np
import os
import time
import heapq
import pdb  
from numba import njit
from itertools import product,combinations
from collections import defaultdict
from typing import MutableSet,DefaultDict,Tuple,List,Iterable,FrozenSet
import copy
from cpp import reachability

import random
global x_dim, y_dim, z_dim
x_dim = 6
y_dim = 6
z_dim = 4

Cell=Tuple[int,int,int]

def shadow_region(cell,x_dim,y_dim,z_dim,shadow=None):
    x=cell[0]
    y=cell[1]
    z=cell[2]
    if shadow is None:
        shadow=set()
    for dz in range(1,cell[2]+1):
        #produce all dx,dy s.t. |dx|+|dy|<=dz
        for dx in range(-dz,dz+1):
            for dy in range(-(dz-abs(dx)),dz-abs(dx)+1):
                candidate=(x+dx,y+dy,z-dz)
                if in_bounds(*candidate,x_dim,y_dim,z_dim):
                    shadow.add((1,1,1,x+dx,y+dy,z-dz))
                    shadow.add((1,1,0,x+dx,y+dy,z-dz))
    return shadow


def get_scaffolding_options(goal_state,x_dim,y_dim,z_dim):
    scaffold_set=set()
    for action in goal_state:
        scaffold_set=shadow_region(action[3:6],x_dim,y_dim,z_dim,scaffold_set)
    return scaffold_set

def generate_shadow_surface_at_distance(cell,distance,x_dim,y_dim,z_dim):
    x=cell[0]
    y=cell[1]
    z=cell[2]
    zout=z-distance
    if zout<0:
        return
    candidate=(x-distance,y,zout)
    if in_bounds(*candidate,x_dim,y_dim,z_dim):
        yield candidate
    for dx in range(1-distance,distance):
        mag_dy=distance-abs(dx)
        candidate=(x+dx,y+mag_dy,zout)
        if in_bounds(*candidate,x_dim,y_dim,z_dim):
            yield candidate
        candidate=(x+dx,y-mag_dy,zout)
        if in_bounds(*candidate,x_dim,y_dim,z_dim):
            yield candidate
    candidate=(x+distance,y,zout)
    if in_bounds(*candidate,x_dim,y_dim,z_dim):
        yield candidate

def generate_shadow_interior_at_dz(cell,dz,x_dim,y_dim,z_dim):
    x=cell[0]
    y=cell[1]
    z=cell[2]
    zout=z-dz
    if zout<0:
        return
    #produce all dx,dy s.t. |dx|+|dy|<dz
    for dx in range(1-dz,dz):
        for dy in range(1-(dz-abs(dx)),dz-abs(dx)):
            candidate=(x+dx,y+dy,z-dz)
            if in_bounds(*candidate,x_dim,y_dim,z_dim):
                yield candidate

def get_all_actions(goal_state):
    global x_dim, y_dim, z_dim
    children_action = []
    max_x = min_x = max_y = min_y = max_z = 0
    max_x = max([x[3] for x in goal_state])
    min_x = min([x[3] for x in goal_state])
    max_y = max([x[4] for x in goal_state])
    min_y = min([x[4] for x in goal_state])
    max_z = max([x[5] for x in goal_state])

    new_x_max = min(x_dim-1, max_x + max_z)
    new_x_min = max(0, min_x - max_z - 1)
    new_y_max = min(y_dim-1, max_y + max_z)
    new_y_min = max(0, min_y - max_z - 1)
    new_z_max = max_z

    for x in range(new_x_min, new_x_max):
        for y in range(new_y_min, new_y_max):
            for z in range(new_z_max):
                child_action = (1, 1, 1, x, y, z)
                children_action.append(child_action)
    return children_action

class Block:
    def cells_occupied(self,x:int,y:int,z:int,rotation:int)->FrozenSet[Tuple[int,int,int]]:
        raise NotImplementedError
    def cells_swept(self,x1: int, y1: int, z1: int, rotation1: int, x2: int, y2: int, z2: int, rotation2: int)->FrozenSet[Tuple[int,int,int]]:
        raise NotImplementedError
    def get_block_placements(self,x:int,y:int,z:int,rotation:int)->FrozenSet[Tuple[int,int,int]]:
        raise NotImplementedError
    def can_pickup_from(self,x:int,y:int,z:int,rotation:int,block_x:int,block_y:int,block_z:int,block_rotation:int)->bool:
        raise NotImplementedError
    def can_place_at(self,x:int,y:int,z:int,rotation:int,occupancy_grid)->Tuple[bool,Tuple[int,int,int]]:
        raise NotImplementedError
    def collides(self,x:int,y:int,z:int,rotation:int,occupancy_grid)->Tuple[bool,Tuple[int,int,int]]:
        raise NotImplementedError
    def support_options(self,x:int,y:int,z:int,rotation:int)->List[MutableSet[Cell]]:
        raise NotImplementedError
    def action_locations(self,x:int,y:int,z:int,rotation:int)->MutableSet[Cell]:
        raise NotImplementedError
class NoBlock(Block):
    def __init__(self):
        pass
    def cells_occupied(self,x:int,y:int,z:int,rotation:int)->FrozenSet[Tuple[int,int,int]]:
        return frozenset()
    def cells_swept(self, x1: int, y1: int, z1: int, rotation1: int, x2: int, y2: int, z2: int, rotation2: int) -> FrozenSet[Tuple[int,int,int]]:
        return frozenset()
    def __hash__(self):
        return 0
    def __eq__(self, other) -> bool:
        return isinstance(other,NoBlock)
    def __repr__(self):
        return "NoBlock"
    def get_block_placements(self, x: int, y: int, z: int, rotation: int) -> FrozenSet[Tuple[int,int,int]]:
        return frozenset()
    def can_pickup_from(self, x: int, y: int, z: int, rotation: int, block_x: int, block_y: int, block_z: int, block_rotation)->bool:
        return False
    def can_place_at(self, x: int, y: int, z: int, rotation: int, occupancy_grid) -> Tuple[bool,Tuple[int,int,int]]:
        return False,None
    def collides(self, x: int, y: int, z: int, rotation: int, occupancy_grid) -> Tuple[bool,Tuple[int,int,int]]:
        return False,None
    def action_locations(self, x: int, y: int, z: int, rotation: int) -> MutableSet[Tuple[int]]:
        return set()
    
class CuboidalBlock(Block):
    length:int
    def __init__(self,length:int):
        self.length=length
    def cells_occupied(self,x: int, y: int, z: int, rotation: int)->FrozenSet[Tuple[int,int,int]]:
        return frozenset(cells_occupied(self.length,rotation,x,y,z))
    def cells_swept(self, x1: int, y1: int, z1: int, rotation1: int, x2: int, y2: int, z2: int, rotation2: int) -> FrozenSet[Tuple[int,int,int]]:
        if rotation1==rotation2:
            return self.cells_occupied(x1,y1,z1,rotation1)|self.cells_occupied(x2,y2,z2,rotation2)
        xinds=range(x1-int(self.length/2),x1+int(self.length/2)+1)
        yinds=range(y1-int(self.length/2),y1+int(self.length/2)+1)
        return frozenset(product(xinds,yinds,(z1,)))
    def __eq__(self,other):
        return isinstance(other,CuboidalBlock) and other.length==self.length
    def __hash__(self):
        return self.length
    def __repr__(self):
        return f"Cuboid{self.length}"
    def get_block_placements(self, x: int, y: int, z: int, rotation: int) -> FrozenSet[Tuple[int,int,int]]:
        if rotation==1:
            return frozenset(((x-1,y,z),(x+1,y,z)))
        else:
            return frozenset((x,y+1,z),(x,y-1,z))
    def can_pickup_from(self, x: int, y: int, z: int, rotation: int, block_x: int, block_y: int, block_z: int, block_rotation):
        if not rotation==block_rotation:
            return False
        if block_rotation==1:
            return y==block_y and z==block_z and (x==block_x-1 or x==block_x+1)
        else:
            return x==block_x and z==block_z and (y==block_y-1 or y==block_y+1)
    def can_place_at(self, x: int, y: int, z: int, rotation: int, occupancy_grid) -> Tuple[bool,Tuple[int,int,int]]:
        if not has_support_below([1,self.length,rotation,x,y,z],occupancy_grid):
            return False,None
        collides,cell=self.collides(x,y,z,rotation,occupancy_grid)
        return not collides,cell
    def collides(self, x: int, y: int, z: int, rotation: int, occupancy_grid) -> Tuple[bool,Tuple[int,int,int]]:
        for cell in self.cells_occupied(x,y,z,rotation):
            if occupancy_grid[*cell]:
                return True,cell
        return False,None
    def support_options(self, x: int, y: int, z: int, rotation: int) -> List[MutableSet[Tuple[int]]]:
        options=[]
        #supported below CoM
        center={(x,y,z-1)}
        options.append(center)
        #support evenly on either side of COM
        if rotation==0:
            for d in range(1,self.length//2+1):
                cell1=(x-d,y,z-1)
                cell2=(x+d,y,z-1)
                options.append({cell1,cell2})
        else:
            for d in range(1,self.length//2+1):
                cell1=(x,y-d,z-1)
                cell2=(x,y+d,z-1)
                options.append({cell1,cell2})
        return options
    def action_locations(self, x: int, y: int, z: int, rotation: int) -> MutableSet[Tuple[int]]:
        return action_locations(self.length,rotation,(x,y,z))
    
def from_specification(block_type:str,block_data:dict):
    if block_type=="NoBlock":
        return NoBlock()
    elif block_type=="CuboidalBlock":
        return CuboidalBlock(**block_data)
    else:
        raise ValueError(f"Unrecognized block tyoe {block_type}")

@njit
def cells_occupied(length, rotation, x, y, z):
    if rotation!=1:
        x_offset = np.arange(-int(length/2), int(length/2)+1)
        y_offset = np.zeros(length+1,dtype=np.int64)
    else:
        x_offset = np.zeros(length+1,dtype=np.int64)
        y_offset = np.arange(-int(length/2), int(length/2)+1)
    zint=int(z)
    xocc=x+x_offset
    yocc=y+y_offset
    for i in range(length):
        yield (xocc[i],yocc[i],zint)

def has_support_below(new_action, occu_grid):
    if new_action[0] == -1 or new_action[5] == 0:
        return True
    _,length,rotation,x,y,z=new_action
    if occu_grid[x,y,z-1]:
        #supported directly below COM
        return True
    if rotation==0:
        for d in range(1,length//2+1):
            cell1=(x-d,y,z-1)
            cell2=(x+d,y,z-1)
            if in_bounds(*cell1,*occu_grid.shape) and in_bounds(*cell2,*occu_grid.shape) and occu_grid[*cell1] and occu_grid[*cell2]:
                #supported by a pair about the COM
                return True
    else:
        for d in range(1,length//2+1):
            cell1=(x,y-d,z-1)
            cell2=(x,y+d,z-1)
            if in_bounds(*cell1,*occu_grid.shape) and in_bounds(*cell2,*occu_grid.shape) and occu_grid[*cell1] and occu_grid[*cell2]:
                #supported by a pair about the COM
                return True
    return False

def occupancy_grid(state, x_dim,y_dim,z_dim,check_assert=False):
    occu_grid = np.zeros((x_dim, y_dim, z_dim), dtype=bool)
    for action in state:

        for cell in cells_occupied(action[1], action[2], action[3], action[4], action[5]):
            if action[0]==1:
                if check_assert: 
                    if occu_grid[cell[0], cell[1], cell[2]]:
                        raise ValueError(f"Overlap in blocks detected at {cell}, redesign goal structure")
                    if not in_bounds(*cell,x_dim,y_dim,z_dim):
                        raise ValueError(f"Block at {cell} out of bounds")
                occu_grid[cell[0], cell[1], cell[2]] = True
            elif action[0]==-1:
                # assert occu_grid[cell[0], cell[1], cell[2]], "Removing non-existent block"
                occu_grid[cell[0], cell[1], cell[2]] = False
    return occu_grid


def all_neighbours():
    nbrs = [(-1, 0, -1), (1, 0, -1), (0, -1, -1), (0, 1, -1), \
            (-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), \
            (-1, 0, 1), (1, 0, 1), (0, -1, 1), (0, 1, 1)]
    return nbrs    

def horizontal_neighbors(x,y,x_dim,y_dim):
    nbrs=[(-1,0),(1,0),(0,-1),(0,1)]
    for dx,dy in nbrs:
        if 0<=x+dx<x_dim and 0<=y+dy<y_dim:
            yield (x+dx,y+dy)

def action_locations(length,rotation,target_cell:Cell)->MutableSet[Tuple[int,int,int,int]]:
    x,y,z=target_cell
    #can only act standing to the sides of the block
    if rotation==0:
        #block has extent in x
        return {(x,y-1,z,0),(x,y+1,z,0)}
    else:
        #block has extent in y
        return {(x-1,y,z,1),(x+1,y,z,1)}


# def has_round_trip(action, occu_grid):
#     global x_dim, y_dim, z_dim 
#     start = (action[3], action[4], action[5])
#     boundary_cells = []
#     block_len = action[1]
#     block_rot = action[2]
#     for x in range(x_dim):
#         for y in range(y_dim):
#             if x==0 or x==x_dim-1 or y==0 or y==y_dim-1:
#                 boundary_cells.append((x, y, 0))

#     open_list = [start]
#     closed_list = []
#     while len(open_list)>0:
        
#         current_cell = heapq.heappop(open_list)
#         closed_list.append(current_cell)
        
#         if current_cell in boundary_cells:
#             return True
#         all_next_cells = current_cell + np.array(all_neighbours())

#         # if len(open_list)==0:
#         #     all_next_cells = all_next_cells[4:8]
#         #     if block_len>1 and block_rot==0:
#         #         all_next_cells = all_next_cells[2:4]
#         #     elif block_len>1 and block_rot==1:
#         #         all_next_cells = all_next_cells[0:2]

#         for next_cell in all_next_cells:
#             if (next_cell[0]>=0 and next_cell[0]<x_dim and next_cell[1]>=0 and next_cell[1]<y_dim and next_cell[2]>=0 and next_cell[2]<z_dim):
#                 if (occu_grid[next_cell[0], next_cell[1], next_cell[2]-1] or (next_cell[2]==0)) and not occu_grid[next_cell[0], next_cell[1], next_cell[2]]:
#                     if (tuple(next_cell) not in closed_list) and not (tuple(next_cell) == start):
#                         heapq.heappush(open_list, tuple(next_cell))
#     return False

class CachingReachability:
    def __init__(self,occu_grid):
        self.occu_grid=occu_grid
        self.x_dim,self.y_dim,self.z_dim=self.occu_grid.shape
        #initially only mark the boundary as reachable
        self.reachable_cells={(0,y,0) for y in range(0,y_dim)}|{(x,0,0) for x in range(1,x_dim)}|{(x_dim,y,0) for y in range(0,y_dim)}|{(x,y_dim,0) for x in range(0,x_dim)}
        self.unreachable_cells=set()
    def has_path_from_boundary(self,cell):
        x,y,z=cell
        if not can_stand_at(*cell,self.occu_grid):
            return False
        start = cell
        open_list = [start]
        closed_list = []
        while len(open_list)>0:
            
            current_cell = heapq.heappop(open_list)
            closed_list.append(current_cell)
            
            if current_cell in self.reachable_cells:
                #every cell we explored (so open AND closed lists) is reachable
                self.reachable_cells.update(open_list)
                self.reachable_cells.update(closed_list)
                return True
            if current_cell in self.unreachable_cells:
                break
            all_next_cells = current_cell + np.array(all_neighbours())

            for next_cell in all_next_cells:
                if can_stand_at(next_cell[0],next_cell[1],next_cell[2],self.occu_grid):
                    if tuple(next_cell) not in closed_list:
                        heapq.heappush(open_list, tuple(next_cell))
        #every cell we explored (so the closed list) is UNREACHABLE
        self.unreachable_cells.update(closed_list)
        return False
def has_path_to_boundary(action, occu_grid, purpose="before_action"):
    # TODO: Account for orientation of the block, it can only be placed front or back if the length is greater than 1
    x_dim,y_dim,z_dim=occu_grid.shape
    start = (action[3], action[4], action[5])
    boundary_cells = []
    block_len = action[1]
    block_rot = action[2]
    for x in range(x_dim):
        for y in range(y_dim):
            if x==0 or x==x_dim-1 or y==0 or y==y_dim-1:
                boundary_cells.append((x, y, 0))

    open_list = [start]
    closed_list = []
    while len(open_list)>0:
        
        current_cell = heapq.heappop(open_list)
        closed_list.append(current_cell)
        
        if current_cell in boundary_cells:
            return True
        all_next_cells = current_cell + np.array(all_neighbours())
        if len(open_list)==0 and current_cell==start:
            if block_len>1 and block_rot==0 and purpose=="before_action":
                all_next_cells = np.delete(all_next_cells, [0, 1, 2, 3, 4, 5, 8, 9, 10, 11], axis=0)
            elif block_len>1 and block_rot==0 and purpose=="after_action":
                all_next_cells = np.delete(all_next_cells, [0, 1, 4, 5, 8, 9], axis=0)
            elif block_len>1 and block_rot==1 and purpose=="before_action":
                all_next_cells = np.delete(all_next_cells, [0, 1, 2, 3, 6, 7, 8, 9, 10, 11], axis=0)
            elif block_len>1 and block_rot==1 and purpose=="after_action":
                all_next_cells = np.delete(all_next_cells, [2, 3, 6, 7, 10, 11], axis=0)

        for next_cell in all_next_cells:
            if can_stand_at(next_cell[0],next_cell[1],next_cell[2],occu_grid):
                if tuple(next_cell) not in closed_list:
                    heapq.heappush(open_list, tuple(next_cell))
    return False

@njit
def can_stand_at(x,y,z,occupancy_grid):
    x_dim,y_dim,z_dim=occupancy_grid.shape
    #check in bounds
    if x>=0 and x<x_dim and y>=0 and y<y_dim and z>=0 and z<z_dim:
        #check cell is not occupied but the cell below it is
        if not occupancy_grid[x,y,z] and (z==0 or occupancy_grid[x,y,z-1]):
            return True
    return False

def in_bounds(x,y,z,x_dim,y_dim,z_dim):
    return x>=0 and x<x_dim and y>=0 and y<y_dim and z>=0 and z<z_dim
class Reachability:
    reachable_cells:MutableSet[Cell]
    parents:DefaultDict[Cell,MutableSet[Cell]]
    children:DefaultDict[Cell,MutableSet[Cell]]
    def __init__(self,reachable_cells:MutableSet[Cell],parents:DefaultDict[Cell,List[Cell]],children:DefaultDict[Cell,List[Cell]]):
        self.reachable_cells=reachable_cells
        self.parents=parents
        self.children=children
    def copy(self):
        return type(self)(copy.copy(self.reachable_cells),copy.deepcopy(self.parents),copy.deepcopy(self.children))
    @classmethod
    def from_occupancy_grid(cls,occu_grid):
        """
        given an occupancy grid, return a Reachability object recording the cells that can be reached from the boundary
        """
        parents=defaultdict(set)
        children=defaultdict(set)
        x_dim,y_dim,z_dim=occu_grid.shape
        boundary_cells={(-1,y,0) for y in range(-1,y_dim+1)}|{(x,-1,0) for x in range(0,x_dim+1)}
        reachable_cells=boundary_cells.copy()
        frontier=boundary_cells
        while len(frontier)>0:
            current_cell=frontier.pop()
            for delta in all_neighbours():
                next_cell=tuple((current_cell[i]+delta[i]) for i in range(3))
                if can_stand_at(*next_cell,occu_grid):
                    parents[next_cell].add(current_cell)
                    children[current_cell].add(next_cell)
                    if next_cell not in reachable_cells:
                        reachable_cells.add(next_cell)
                        frontier.add(next_cell)
        return cls(reachable_cells,parents,children)
    def mark_reachable(self,occu_grid,newly_reachable_cells:Iterable[Cell]):
        """
        add new reachable states made reachable by the addition of a set of newly reachable cells. Mutates self.reachable_cells!
        """
        frontier=set(newly_reachable_cells)
        for cell in frontier:
            for parent in reachable_neighbors(cell,self.reachable_cells|frontier):
                #every newly_reachable_cell needs to be added to the children of its neighbors
                self.children[parent].add(cell)
                self.parents[cell].add(parent)
        while len(frontier)>0:
            current_cell=frontier.pop()
            if not current_cell in self.reachable_cells:
                #this cell is newly reachable
                self.reachable_cells.add(current_cell)
                for delta in all_neighbours():
                    next_cell=tuple((current_cell[i]+delta[i]) for i in range(3))
                    if can_stand_at(*next_cell,occu_grid):
                        #add current cell to the parents of its reachable children
                        self.children[current_cell].add(next_cell)
                        self.parents[next_cell].add(current_cell)
                        if next_cell not in self.reachable_cells:
                            frontier.add(next_cell)
    def mark_unreachable(self,newly_unreachable_cells:Iterable[Cell]):
        """
        remove reachable states due to removal of newly unreachable cells. Mutates reachable_cells!
        """
        frontier=set(newly_unreachable_cells)
        while len(frontier)>0:
            current_cell=frontier.pop()
            if current_cell in self.reachable_cells:
                self.reachable_cells.remove(current_cell)
                #remove newly unreachable cell from the parents of its children
                cset=self.children[current_cell]
                for child in cset:
                    pset=self.parents[child]
                    pset.remove(current_cell)
                    if len(pset)==0:
                        #this child is no longer reachable
                        frontier.add(child)
                self.children[current_cell]=set()
                #remove newly unreachable cell from the children of its parents
                pset=self.parents[current_cell]
                for parent in pset:
                    cset=self.children[parent]
                    cset.remove(current_cell)
                self.parents[current_cell]=set()
    def __repr__(self):
        return f"ReachableCells{self.reachable_cells}"
    def __eq__(self,other):
        if not self.reachable_cells==other.reachable_cells:
            print(f"reachable sets don't match. Symmetric Difference={self.reachable_cells^other.reachable_cells}")
            return False
        for cell in self.reachable_cells:
            if self.parents[cell]!=other.parents[cell]:
                print(f"Parents of {cell} don't match. {self.parents[cell]}!={other.parents[cell]}")
                return False
            if self.children[cell]!=other.children[cell]:
                print(f"Children of {cell} don't match. {self.children[cell]}!={other.children[cell]}")
                return False
        return True

def update_reachable_states_add_connected_blocks(old_reachable:Reachability,occu_grid,add_block_cells:Iterable[Cell]):
    """
    updates the set of reachable states after adding connected blocks to the world.
    """
    new_reachable=old_reachable.copy()
    #the location the block was added to should have been reachable before, but is now definitely unreachable
    new_reachable.mark_unreachable(add_block_cells)
    reachable=set()
    for block in add_block_cells:
        x,y,z=block
        if z+1<z_dim and not occu_grid[x,y,z+1]:
            above=(x,y,z+1)
            adjacent_locations=action_locations(1,1,block)
            if len(new_reachable.reachable_cells&adjacent_locations)>0:
                reachable.add(above)
    new_reachable.mark_reachable(occu_grid,reachable)
    return new_reachable

def update_reachable_states_remove_connected_blocks(old_reachable:Reachability,occu_grid,remove_block_cells:Iterable[Cell]):
    """
    updates the set of reachable states after removing connected blocks from the world.
    """
    new_reachable=old_reachable.copy()
    #the cells above the location the block was removed from should have been reachable before, but are now definitely unreachable
    reachable=set()
    unreachable=set()
    for cell in remove_block_cells:
        x,y,z=cell
        above=(x,y,z+1)
        if above in old_reachable.reachable_cells:
            reachable.add(cell)
            unreachable.add(above)
    new_reachable.mark_unreachable(unreachable)
    #if cell above was reachable, the cell occupied by the removed block is now reachable
    new_reachable.mark_reachable(occu_grid,reachable)
    return new_reachable

def reachable_neighbors(cell,reachable_cells):
    neighbors=set()
    for delta in all_neighbours():
        next_cell=tuple((cell[i]+delta[i]) for i in range(3))
        if next_cell in reachable_cells:
            neighbors.add(next_cell)
    return neighbors

def check_single_agent_plan_is_feasible(action_sequence,initial_occu_grid):
    """
    given a sequence of actions, verify that they are executable in that order by a single agent

    Parameters: action_sequence : list[list[int]]
                    sequence of actions, where each action is a list [-1|1 means remove or add a block, width of the block, 0|1 means extent in y or x, x loc, y loc, z loc]
                initial_occu_grid : (x_dim,y_dim,z_dim) binary np array
                    which cells have a block in them prior to executing the first action in action_sequence
    Returns:    is_feasible : bool
                    True if entire action sequence was single agent feasible; otherwise False
                reachable_occu_grid : (x_dim,y_dim,z_dim) binary np array
                    which cells have a block in them after executing the last FEASIBLE action in action_sequence
                index : int
                    the length of the feasible action sequence. If not is_feasible, this is the index in action_sequence of the first infeasible action
                message : string
                    string explaining why the action sequence was found infeasible, if it was in fact infeasible.
    """
    occu_grid=initial_occu_grid
    for i,action in enumerate(action_sequence):
        affected_cells=list(cells_occupied(action[1],action[2],action[3],action[4],action[5]))
        affected_occupied=[occu_grid[c] for c in affected_cells]
        child_occupancy=occu_grid.copy()
        if action[0]==-1:
            if not all(affected_occupied):
                #tried to remove a block that doesn't exist
                return False, occu_grid,i,"Tried to remove block that doesn't exist"
            else:
                for c in affected_cells:
                    child_occupancy[c]=False
            outgoing_block_length=0
            incoming_block_length=action[1]
        elif action[0]==1:
            if any(affected_occupied):
                return False, occu_grid,i,"Tried to place block in occupied cell"
            else:
                for c in affected_cells:
                    child_occupancy[c]=True
            outgoing_block_length=action[1]
            incoming_block_length=0
        else:
            raise ValueError(f"Action {i}:{action} has an unrecognized add/remove entry {action[0]}. Must be -1 or 1.")
        locs=action_locations(action[1],action[2],tuple(action[3:6]))
        path_tester=reachability.CachingReachability(outgoing_block_length,*occu_grid.shape,occu_grid)
        child_path_tester=reachability.CachingReachability(incoming_block_length,*child_occupancy.shape,child_occupancy)
        for action_loc in locs:
            if path_tester.has_path_from_boundary(action_loc):
                if child_path_tester.has_path_from_boundary(action_loc):
                    #action is reachable from boundary and has path back to boundary
                    occu_grid=child_occupancy
                    feasible=True
                    break
            feasible=False
        if not feasible:
            return False,occu_grid,i,"No action location is reachable both before and after doing the action"
    return True,occu_grid,i+1,"Sequence is feasible"

def check_plan_achieves_goal(action_sequence,initial_block_set,goal):
    goal_block_set={tuple(action[1:]) for action in goal}
    current_block_set=set(initial_block_set.copy())
    for i,action in enumerate(action_sequence):
        block=tuple(action[1:])
        if action[0]==-1:
            if block not in current_block_set:
                return False, current_block_set,i,"Tried to remove a block that doesn't exist"
            else:
                current_block_set.remove(block)
        elif action[0]==1:
            if block in current_block_set:
                return False,current_block_set,i,"Tried to place a block that already exists"
            else:
                current_block_set.add(block)
        else:
            raise ValueError(f"Action {i}:{action} has an unrecognized add/remove entry {action[0]}. Must be -1 or 1.")
    if current_block_set!=goal_block_set:
        return False, current_block_set,i+1,"Sequence doesn't produce goal"
    else:
        return True, current_block_set,i+1,"Sequence produces goal"
# def has_path_to_boundary(action, occu_grid, return_path = 0):
#     # TODO: Account for orientation of the block, it can only be placed front or back if the length is greater than 1
#     global x_dim, y_dim, z_dim 
#     start = (action[3], action[4], action[5])
#     boundary_cells = []
#     block_len = action[1]
#     block_rot = action[2]
#     for x in range(x_dim):
#         for y in range(y_dim):
#             if x==0 or x==x_dim-1 or y==0 or y==y_dim-1:
#                 boundary_cells.append((x, y, 0))

#     # find a path from start to any boundary cell
#     open_list = [start]
#     closed_list = []
#     while len(open_list)>0:
def find_path_to_boundary(action, occu_grid, purpose="before_action"):
    # TODO: Account for orientation of the block, it can only be placed front or back if the length is greater than 1
    x_dim,y_dim,z_dim=occu_grid.shape
    start = (action[3], action[4], action[5])
    boundary_cells = []
    block_len = action[1]
    block_rot = action[2]
    for x in range(x_dim):
        for y in range(y_dim):
            if (x==0 or x==x_dim-1) or (y==0 or y==y_dim-1):
                boundary_cells.append((x, y, 0))

    # shuffle order of boundary cells
    random.seed()
    random.shuffle(boundary_cells)

    open_list = [start]
    closed_list = []
    parent_map = {}
    while len(open_list)>0:
        
        current_cell = heapq.heappop(open_list)
        closed_list.append(current_cell)
        
        if current_cell in boundary_cells:
            path = []
            while current_cell != start:
                path.append(current_cell)
                current_cell = parent_map[current_cell]
            path.append(start)
            path.reverse()
            return True, path
            # return True
        

        all_next_cells = current_cell + np.array(all_neighbours())
        if len(open_list)==0 and current_cell==start:
            if block_len>1 and block_rot==0 and purpose=="before_action":
                all_next_cells = np.delete(all_next_cells, [0, 1, 2, 3, 4, 5, 8, 9, 10, 11], axis=0)
            elif block_len>1 and block_rot==1 and purpose=="before_action":
                all_next_cells = np.delete(all_next_cells, [0, 1, 2, 3, 6, 7, 8, 9, 10, 11], axis=0)

        for next_cell in all_next_cells:
            if (next_cell[0]>=0 and next_cell[0]<x_dim and next_cell[1]>=0 and next_cell[1]<y_dim and next_cell[2]>=0 and next_cell[2]<z_dim):
                if (occu_grid[next_cell[0], next_cell[1], next_cell[2]-1] or (next_cell[2]==0)) and not occu_grid[next_cell[0], next_cell[1], next_cell[2]]:
                    if tuple(next_cell) not in closed_list:
                        heapq.heappush(open_list, tuple(next_cell))
                        parent_map[tuple(next_cell)] = current_cell
    return False, []