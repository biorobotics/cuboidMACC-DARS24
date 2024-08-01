import numpy as np
import os
import time
import math
from . import block_utils
import pdb
from assembly.cpp import reachability
from typing import Dict,MutableSet,Tuple
from collections import defaultdict

def return_list_with_m1():
    return [-1]

class ReachabilityManager:
    def __init__(self,occupancy_grid=None,state=None,x_dim=None,y_dim=None,z_dim=None):
        self.reachability=dict()
        self.occupancy_grid=occupancy_grid
        self.state=state
        self.x_dim=x_dim
        self.y_dim=y_dim
        self.z_dim=z_dim
    def has_path_from_boundary(self,block_length,action_loc):
        if block_length not in self.reachability:
            if self.occupancy_grid is None:
                self.occupancy_grid=reachability.occupancy_grid(self.state,self.x_dim,self.y_dim,self.z_dim)
            self.reachability[block_length]=reachability.CachingReachability(block_length,*self.occupancy_grid.shape,self.occupancy_grid)
        return self.reachability[block_length].has_path_from_boundary(action_loc)
class Assembly_Node:
    def __init__(self, state=[], parent=None, reachability:ReachabilityManager=None,x_dim=10,y_dim=10,z_dim=4):
        self.parent = parent
        self.state = state
        if len(self.state) != 0:
            self.action = self.state[-1][0]
            self.length = self.state[-1][1]
            self.rotation = self.state[-1][2]
            self.location = self.state[-1][3:6]
        
        if parent is not None:
            curr_node = parent.curr_node.copy()
            if self.action==1: curr_node.add(tuple(self.state[-1]))
            else: curr_node.remove(tuple((1, self.state[-1][1], self.state[-1][2], self.state[-1][3], self.state[-1][4], self.state[-1][5])))
            self.curr_node = curr_node
            self.x_dim=parent.x_dim
            self.y_dim=parent.y_dim
            self.z_dim=parent.z_dim
            self.scaffolding_options=parent.scaffolding_options
        else: 
            self.curr_node = set()
            for action in self.state:
                if action[0]==1:
                    self.curr_node.add(action)
                else:
                    self.curr_node.remove((1,)+action[1:])
            self.x_dim=x_dim
            self.y_dim=y_dim
            self.z_dim=z_dim
            self.scaffolding_options=None
        if reachability is None:
            self.reachability=ReachabilityManager(state=self.state,x_dim=self.x_dim,y_dim=self.y_dim,z_dim=self.z_dim)
        else:
            self.reachability=reachability
        self._hash=frozenset(self.curr_node).__hash__()
        self.child_reachability_cache=dict()
        self.scaff_placed=None
        self.actions_requiring_scaffolding=None
    def __getstate__(self):
        state={"state":self.state,"x_dim":self.x_dim,"y_dim":self.y_dim,"z_dim":self.z_dim}
        return state
    def __setstate__(self,state):
        self.__init__(**state)
    @classmethod
    def from_npy(cls,filepath,world_size=None):
        as_array=np.load(filepath)
        as_tuples=[tuple((int(ai) for ai in a)) for a in as_array]
        if world_size is not None:
            max_x,max_y,max_z=world_size
        else:
            max_x=0
            max_y=0
            max_z=0
            for act in as_tuples:
                action_type,length,rotation,x,y,z=act
                cells=block_utils.cells_occupied(length,rotation,x,y,z)
                for cell in cells:
                    x,y,z=cell
                    max_x=max(x,max_x)
                    max_y=max(y,max_y)
                    max_z=max(z,max_z)
            max_z+=1
        
        return cls(as_tuples,x_dim=max_x,y_dim=max_y,z_dim=max_z)
    @classmethod
    def from_heightmap(cls,heightmap):
        max_z=np.max(heightmap)
        z_dim=max_z+1
        x_dim,y_dim=heightmap.shape
        as_tuples=[]
        for x in range(x_dim):
            for y in range(y_dim):
                for z in range(heightmap[x,y]):
                    as_tuples.append((1,1,0,x,y,z))
        return cls(as_tuples,x_dim=x_dim,y_dim=y_dim,z_dim=z_dim)
    
    def get_scaffolding_options(self,goal_state):
        if self.scaffolding_options is None:
            self.scaffolding_options={action for action in block_utils.get_scaffolding_options(goal_state.curr_node,self.x_dim,self.y_dim,self.z_dim) if not goal_state.reachability.occupancy_grid[*action[3:6]]}
        return self.scaffolding_options
            
    def reachability_from_parent_and_action(self,parent,action):
        action_type=action[0]
        length=action[1]
        rotation=action[2]
        center_cell=tuple(action[3:6])

        blocks_affected=set(block_utils.cells_occupied(length,rotation,center_cell[0],center_cell[1],center_cell[2]))
        if action_type==-1:
            #remove
            inc= block_utils.update_reachable_states_remove_connected_blocks(parent.scratch_reachability,self.reachability.occupancy_grid,blocks_affected)
        elif action_type==1:
            #add
            inc= block_utils.update_reachable_states_add_connected_blocks(parent.scratch_reachability,self.reachability.occupancy_grid,blocks_affected)
        else:
            raise ValueError(f"Unrecognized action type {action_type}! Expected -1 (remove) or 1 (add).")
        if self.scratch_reachability!=inc:
            raise ValueError
        return inc

    def __eq__(self, other):
        return self._hash == other._hash
    def __hash__(self) -> int:
        return self._hash

    def get_parent_node(self):
        return self.parent
    
    def scaffolding_present(self,goal):
        if self.scaff_placed is None:
            self.scaff_placed = set()
            for action in self.curr_node:
                if (action[0] == 1) and (action not in goal.curr_node) and action[1]==1:
                    self.scaff_placed.add(action)
                if (action[0]== -1):
                    self.scaff_placed.remove(action)
        return self.scaff_placed

    def get_child_nodes(self, goal):
        if self.reachability.occupancy_grid is None:
            self.reachability.occupancy_grid = reachability.occupancy_grid(self.state,self.x_dim,self.y_dim,self.z_dim)
        children = []
        unreachable_goals=False
        for action in goal.state:
            if action not in self.curr_node:
                valid,child_reachability=self.is_valid_node(action) # do not remove
                if valid:
                    child_state = self.state.copy()
                    child_state.append(action)
                    children.append(Assembly_Node(child_state, self,child_reachability))
                else:
                    unreachable_goals=True
        scaff_placed = self.scaffolding_present(goal)
        # print("scaff_placed", scaff_placed)
        
        for action in scaff_placed:
            child_action = (-1, 1, action[2], action[3], action[4], action[5])
            valid,child_reachability=self.is_valid_node(child_action)
            if valid:
                child_state = self.state.copy()
                child_state.append(child_action)
                children.append(Assembly_Node(child_state, self,child_reachability))

        if unreachable_goals:
            #only look for scaffolding if at least one goal action is needed and not reachable
            children_action = self.get_scaffolding_options(goal)
            for action in children_action:
                if (-1,)+action[1:] not in self.state:#if we already removed this block don't try replacing it
                    valid,child_reachability=self.is_valid_node(action)
                    if valid:
                        child_state = self.state.copy()
                        child_state.append(action)
                        children.append(Assembly_Node(child_state, self,child_reachability))
        return children
    
    def get_state(self):
        return self.state
    
    def is_valid_node(self, child_action, child_reachability=None):
        if child_action in self.state:
            return False,child_reachability
        if any(self.reachability.occupancy_grid[cell[0],cell[1],cell[2]+1] for cell in block_utils.cells_occupied(child_action[1],child_action[2],*child_action[3:6]) if block_utils.in_bounds(cell[0],cell[1],cell[2]+1,self.x_dim,self.y_dim,self.z_dim)):
            #there is an occupied cell directly above a cell this block action would mutate. Such an action is forbidden.
            return False,child_reachability
        if block_utils.has_support_below(child_action, self.reachability.occupancy_grid): 
            return self.action_is_reachable(child_action,child_reachability)
        return False,child_reachability
    
    def reachable_via_precompute(self,action,child_reachability=None):
        if self.scratch_reachability is None:
            if self.reachability.occupancy_grid is None:
                self.reachability.occupancy_grid = reachability.occupancy_grid(self.state,self.x_dim,self.y_dim,self.z_dim)
            self.scratch_reachability=block_utils.Reachability.from_occupancy_grid(self.reachability.occupancy_grid)
        action_locations=block_utils.action_locations(action[1],action[2],tuple(action[3:6]))
        reachable_action_locations=action_locations^self.scratch_reachability.reachable_cells
        if len(reachable_action_locations)>0:
            if child_reachability is None:
                new_state = self.state.copy()
                new_state.append(action)
                new_occupancy_grid = reachability.occupancy_grid(new_state,self.x_dim,self.y_dim,self.z_dim)
                child_reachability=block_utils.CachingReachability(new_occupancy_grid)
            for reachable in reachable_action_locations:
                if reachable in child_reachability.reachable_cells:
                    return True,child_reachability
        return False,child_reachability
    def action_is_reachable(self,action,child_reachability=None):
        atuple=tuple(action)
        if atuple in self.child_reachability_cache:
            stored_valid,stored_reachability=self.child_reachability_cache[atuple]
            if stored_reachability is None and child_reachability is not None:
                self.child_reachability_cache[atuple]=(stored_valid,child_reachability)
                stored_reachability=child_reachability
        else:
            stored_valid,stored_reachability=self.test_action_is_reachable(action,child_reachability)
            self.child_reachability_cache[atuple]=(stored_valid,child_reachability)
        return stored_valid,stored_reachability
    def test_action_is_reachable(self,action,child_reachability=None):
        """
        determines if there exists an action location (valid for the particular type of action) that is reachable both before and after completing the action
        """
        block_length=action[1]
        if action[0]==-1:
            outgoing_block_length=1
            returning_block_length=block_length
        elif action[0]==1:
            outgoing_block_length=block_length
            returning_block_length=1
        action_locations=block_utils.action_locations(action[1],action[2],tuple(action[3:6]))
        for action_loc in action_locations:
            if self.reachability.has_path_from_boundary(outgoing_block_length,action_loc):
                if child_reachability is None:
                    new_state = self.state.copy()
                    new_state.append(action)
                    new_occupancy_grid = reachability.occupancy_grid(new_state,self.x_dim,self.y_dim,self.z_dim)
                    child_reachability = ReachabilityManager(new_occupancy_grid,new_state,self.x_dim,self.y_dim,self.z_dim)
                if child_reachability.has_path_from_boundary(returning_block_length,action_loc):
                    return True,child_reachability
        return False,child_reachability

    def is_goal_node(self, goal):
        curr_node = set()
        goal_node = set()
        for action in self.state:
            if action[0]==1: curr_node.add(tuple(action))
            elif action[0]==-1: curr_node.remove(tuple((1, action[1], action[2], action[3], action[4], action[5])))
        for action in goal.state:
            if action[0]==1: goal_node.add(tuple(action))
        return curr_node == goal_node


    def scaffolding_needed(self, blocks_to_be_placed):
        scaff_needed = 0
        if self.reachability.occupancy_grid is None:
            self.reachability.occupancy_grid = reachability.occupancy_grid(self.state,self.x_dim,self.y_dim,self.z_dim)
        levels_done = []
        for b_action in blocks_to_be_placed:
            if not b_action[5] in levels_done:
                is_valid=self.action_is_reachable(b_action)[0]
                if not is_valid:
                    levels_done.append(b_action[5])
            
        scaff_needed = len(levels_done)*(len(levels_done)+1)

        return scaff_needed
                # print("here!! h2", h2)
    
    def estimate_scaffolding_required_to_stand_at(self,cell,goal,available_heights_cache):
        if not self.reachability.occupancy_grid[*cell]:
            #location is clear to stand            
            highest_block=-1
            for z in range(cell[2]):
                if self.reachability.occupancy_grid[cell[0],cell[1],z] or goal.reachability.occupancy_grid[cell[0],cell[1],z]:
                    highest_block=z

            highest_block=max(highest_block,max(z for z in available_heights_cache[cell[:2]] if z<cell[2]))
            scaffold_actions_needed=(cell[2]-highest_block-1)*2
            heights_requested=range(highest_block+1,cell[2])
        else:
            scaffold_actions_needed=1#at least one block needs to be removed...not admissible because it might be a goal block not a scaffold block
            heights_requested=[cell[2]-1]
        return scaffold_actions_needed,heights_requested

    def missing_places_to_stand(self, goal):
        needed=0
        actions_needing_scaffolding=goal.actions_that_needed_scaffolding()
        available_heights_cache=defaultdict(return_list_with_m1)
        for action in actions_needing_scaffolding:
            if action[0]==1 and (action not in self.curr_node):
                if self.reachability.occupancy_grid is None:
                    self.reachability.occupancy_grid = reachability.occupancy_grid(self.state,self.x_dim,self.y_dim,self.z_dim)
                action_locs=actions_needing_scaffolding[action]
                scaffolding_cost=np.inf
                best_heights_requested=list()
                best_cell=None
                for cell in action_locs:
                    estimate,heights_requested=self.estimate_scaffolding_required_to_stand_at(cell[:3],goal,available_heights_cache)
                    if estimate<scaffolding_cost:
                        scaffolding_cost=estimate
                        best_heights_requested=heights_requested
                        best_cell=cell
                if best_cell is not None:
                    available_heights_cache[(best_cell[0],best_cell[1])].extend(best_heights_requested)
                needed+=scaffolding_cost
        return needed
    
    def actions_that_needed_scaffolding(self)->Dict[Tuple[int,int,int,int,int,int],MutableSet[Tuple[int,int,int,int]]]:
        if self.actions_requiring_scaffolding is None:
            self.actions_requiring_scaffolding=dict()
            for action in self.state:
                action_locs=block_utils.action_locations(action[1],action[2],action[3:6])
                has_support=any(cell[2]==0 or (block_utils.in_bounds(cell[0],cell[1],cell[2],self.x_dim,self.y_dim,self.z_dim) and self.reachability.occupancy_grid[cell[0],cell[1],cell[2]-1]) for cell in action_locs)
                if not has_support:
                    self.actions_requiring_scaffolding[action]={loc for loc in action_locs if block_utils.in_bounds(loc[0],loc[1],loc[2],self.x_dim,self.y_dim,self.z_dim)}
        return self.actions_requiring_scaffolding
    
    def goal_actions_currently_obstructed(self, goal):
        if self.reachability.occupancy_grid is None:
            self.reachability.occupancy_grid = reachability.occupancy_grid(self.state,self.x_dim,self.y_dim,self.z_dim)
        need_removal=[]
        for action in goal.state:
            if action[0]==1 and (action not in self.curr_node):
                action_locs=block_utils.action_locations(action[1],action[2],action[3:6])
                found_loc=False
                for loc in action_locs:
                    if block_utils.in_bounds(*loc[:3],self.x_dim,self.y_dim,self.z_dim):
                        ok=True
                        cells_in_block=block_utils.cells_occupied(action[1],loc[3],loc[0],loc[1],loc[2]+1)
                        for cell in cells_in_block:
                            if block_utils.in_bounds(*cell,self.x_dim,self.y_dim,self.z_dim) and self.reachability.occupancy_grid[*cell]:
                                ok=False
                                break
                        if self.reachability.occupancy_grid[*loc[:3]]:
                            ok=False
                        if ok:
                            found_loc=True
                if not found_loc:
                    need_removal.append(action)
        return need_removal
    
    # def icra_estimate_required_scaffolding_actions(self,goal):
    #     if self.reachability.occupancy_grid is None:
    #         self.reachability.occupancy_grid = reachability.occupancy_grid(self.state,self.x_dim,self.y_dim,self.z_dim)
    #     return 2*scaffold_estimation.estimate_missing_scaffolding(self.reachability.occupancy_grid,goal.reachability.occupancy_grid)
    def get_heuristic(self, goal):
        h = h2 = 0
        blocks_to_be_placed = []
        for action in self.curr_node: 
            if action[0]==1 and (action not in goal.state):
                h += 1
        for action in goal.state:
            if action[0]==1 and (action not in self.curr_node):
                blocks_to_be_placed.append(action)
                h += 1
        h2=self.scaffolding_needed(blocks_to_be_placed)
        return h+h2+len(self.goal_actions_currently_obstructed(goal))

    def get_cost(self):
        return len(self.get_state())
