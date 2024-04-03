from typing import MutableSet,List,Dict,Tuple,FrozenSet,Literal
import sys
sys.path.insert(0, '../')
import argparse
import yaml
from math import fabs
from itertools import combinations,product,chain
from copy import deepcopy,copy
from a_star import AStar
import time
import math
import random
from lookup_conflict import overlap_edges, overlap_vertices
import timeit
from assembly import block_utils
import numpy as np
from numpy.typing import NDArray
from collections import defaultdict
from functools import wraps

from utility.exceptions import *

TIMEOUT="TIMEOUT"
LOW_LEVEL_FOUND="FOUND"
INFEASIBLE="INFEASIBLE"
class Location(object):
    x:int
    y:int
    turn:int
    def __init__(self, x=-1, y=-1, z=0, turn=random.choice([0, 90])):
        self.x = x
        self.y = y
        self.z = z
        self.turn = turn
        self.in_world=True
        self._vertical=None
        # print("location:", self.x, self.y, self.turn, abs_angle)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.z == other.z and self.vertical() == other.vertical()
    def __repr__(self):
        return str((self.x, self.y, self.z, self.turn))
    def __hash__(self):
        return hash((self.x,self.y,self.z,self.turn))
    def vertical(self):
        if self._vertical is None:
            self._vertical=abs(int(math.sin(math.radians(self.turn))))
        return self._vertical
    def in_bounds(self,x_dim,y_dim,z_dim):
        return block_utils.in_bounds(self.x,self.y,self.z,x_dim,y_dim,z_dim)
    def is_boundary(self,x_dim,y_dim,z_dim):
        return self.z==0 and (self.x==0 or self.y==0 or self.x==x_dim-1 or self.y==y_dim-1)
    def required_support_cells(self):
        if self.z<=0:
            return frozenset()
        else:
            return frozenset({(self.x,self.y,self.z-1)})
    def cells_occupied(self):
        return {(self.x,self.y,self.z)}
    def adjacent_locations(self,x_dim,y_dim,z_dim):
        dz=[-1,0,1]
        if self.vertical():
            dy=[0]
            dx=[-1,0,1]
        else:
            dy=[-1,0,1]
            dx=[0]
        for x,y,z in product(dx,dy,dz):
            if x!=0 or y!=0:
                loc=Location(self.x+x,self.y+y,self.z+z,self.turn)
                if loc.in_bounds(x_dim,y_dim,z_dim):
                    yield loc
        # Left Turn action
        left_turn=Location(self.x,self.y,self.z,(self.turn-90)%360)
        yield left_turn
        # Right Turn action
        right_turn=Location(self.x,self.y,self.z,(self.turn+90)%360)
        yield right_turn
        if self.is_boundary(x_dim,y_dim,z_dim):
            #at boundary
            yield OutsideWorld()

class OutsideWorld(Location):
    def __init__(self):
        super().__init__(-1,-1,-1,0)
        self.in_world=False
    def __repr__(self):
        return "OutsideWorld"
    def in_bounds(self, x_dim, y_dim, z_dim):
        return True
    def is_boundary(self, x_dim, y_dim, z_dim):
        return True
    def required_support_cells(self):
        return frozenset()
    def cells_occupied(self):
        return set()
    def adjacent_locations(self, x_dim, y_dim, z_dim):
        yield OutsideWorld()
        boundary=chain.from_iterable((((0,y,0) for y in range(0,y_dim)),((x,0,0) for x in range(1,x_dim)),((x_dim-1,y,0) for y in range(0,y_dim)),((x,y_dim-1,0) for x in range(0,x_dim))))
        for x,y,z in boundary:
            yield Location(x,y,z,0)
            yield Location(x,y,z,90)

def _return0():
    return 0
class WorldState:
    sequence:List[Tuple[bool,block_utils.Block,Location]]
    blocks:MutableSet[Tuple[block_utils.Block,Location]]
    occupancy:NDArray
    will_be_filled:Dict[Tuple[int,int,int],int]
    will_be_emptied:Dict[Tuple[int,int,int],int]
    last_modifier:NDArray
    def __init__(self,sequence,blocks,occupancy,last_modifier=None,will_be_filled:Dict[Tuple[int,int,int],int]=None,will_be_emptied:Dict[Tuple[int,int,int],int]=None,pending_actions=None):
        self.sequence=sequence
        self.blocks=blocks
        self.occupancy=occupancy
        if pending_actions is None:
            self.pending_actions=set()
        else:
            self.pending_actions=pending_actions
        if will_be_filled is None:
            self.will_be_filled=defaultdict(_return0)
        else:
            self.will_be_filled=will_be_filled
        if will_be_emptied is None:
            self.will_be_emptied=defaultdict(_return0)
        else:
            self.will_be_emptied=will_be_emptied
        if last_modifier is None:
            self.last_modifier=np.full(self.occupancy.shape,-1,np.int64)
        else:
            self.last_modifier=last_modifier
    def __eq__(self,other):
        return np.all(self.occupancy==other.occupancy) and self.blocks==other.blocks and self.sequence==other.sequence and self.pending_actions==other.pending_actions
    @classmethod
    def from_blocks(cls,blocks:List[block_utils.Block],locations:List[Location],dimensions):
        out=WorldState(list(),set(),np.zeros(dimensions,dtype=np.bool_))
        for i in range(len(blocks)):
            success=out.add_block(blocks[i],locations[i])
            if not success:
                raise ValueError(f"{blocks[i]}@{locations[i]} is not legal for assembling a WorldState with {out.blocks}")
        out.will_be_emptied=defaultdict(_return0)
        out.will_be_filled=defaultdict(_return0)
        out.last_modifier=np.full(out.occupancy.shape,-1,np.int64)
        return out
    def add_pending_action(self,addition:bool,block:block_utils.Block,location:Location):
        cells=block.cells_occupied(location.x,location.y,location.z,location.vertical())
        if addition:
            for cell in cells:
                self.will_be_filled[cell]+=1
        else:
            for cell in cells:
                self.will_be_emptied[cell]+=1
        self.pending_actions.add((addition,block,location))
    def remove_pending_action(self,addition:bool,block:block_utils.Block,location:Location):
        action_tuple=(addition,block,location)
        if action_tuple in self.pending_actions:
            cells=block.cells_occupied(location.x,location.y,location.z,location.vertical())
            if addition:
                for cell in cells:
                    self.will_be_filled[cell]-=1
            else:
                for cell in cells:
                    self.will_be_emptied[cell]-=1
            self.pending_actions.discard(action_tuple)
    def cell_will_change(self,cell,want_filled:bool):
        if want_filled:
            return cell in self.will_be_filled and self.will_be_filled[cell]>0
        else:
            return cell in self.will_be_emptied and self.will_be_emptied[cell]>0
    def pickupable_from(self,location:Location):
        for block,loc in self.blocks:
            if block.can_pickup_from(location.x,location.y,location.z,location.vertical(),loc.x,loc.y,loc.z,loc.vertical()):
                yield block,loc
    def maybe_placeable_at(self,block:block_utils.Block,location:Location):
        ok,bad_action,bad_cell,addition=self.placeable_at(block,location)
        if ok:
            return ok
        else:
            #as long as the interfering cell is scheduled to change it's ok 
            return bad_cell!=(-1,-1,-1) and self.cell_will_change(bad_cell,addition)
    def placeable_at(self,block:block_utils.Block,location:Location):
        ok,bad_cell=block.can_place_at(location.x,location.y,location.z,location.vertical(),self.occupancy)
        if ok:
            return True,-1,(-1,-1,-1),False
        else:
            if bad_cell is not None:
                #collides with a cell in the world
                return False,self.last_modifier[*bad_cell],bad_cell,False
            else:
                #missing support
                support_sets=block.support_options(location.x,location.y,location.z,location.vertical())
                #do any support sets contain exclusively blocks that exist or are pending placement?
                #or were any complete support sets REMOVED by a previous action?
                for support in support_sets:
                    will_be_placed=True
                    pending_cell=None
                    was_removed=False
                    removed_cell=None
                    for cell in support:
                        if not self.occupancy[*cell]:
                            if not self.cell_will_change(cell,True):
                                if self.last_modifier[*cell]!=-1:
                                    was_removed=True
                                    removed_cell=cell
                                will_be_placed=False
                                break
                            else:
                                pending_cell=cell
                    if will_be_placed:
                        #report infeasible b/c pending_cell is not yet present
                        return False,-1,pending_cell,True
                    if was_removed:
                        #report infeasible b/c removed_cell was removed
                        return False,self.last_modifier[*removed_cell],removed_cell,True
                #missing support that was never placed and is not scheduled!
                return False,-1,(-1,-1,-1),True
                            
    def add_block(self,block:block_utils.Block,location:Location):
        if self.placeable_at(block,location)[0]:
            self.sequence.append((True,block,location))
            self.blocks.add((block,location))
            self.pending_actions.discard((True,block,location))
            idx=len(self.sequence)-1
            cells=block.cells_occupied(location.x,location.y,location.z,location.vertical())
            for cell in cells:
                self.occupancy[*cell]=True
                self.last_modifier[*cell]=idx
                self.will_be_filled[cell]-=1
            return True
        return False
    def remove_block(self,block:block_utils.Block,location:Location):
        if (block,location) in self.blocks:
            self.sequence.append((False,block,location))
            self.blocks.remove((block,location))
            self.pending_actions.discard((False,block,location))
            idx=len(self.sequence)-1
            for cell in block.cells_occupied(location.x,location.y,location.z,location.vertical()):
                self.occupancy[*cell]=False
                self.last_modifier[*cell]=idx
                self.will_be_emptied[cell]-=1
            return True
        return False
    def copy(self):
        return WorldState(copy(self.sequence),copy(self.blocks),self.occupancy.copy(),self.last_modifier.copy(),copy(self.will_be_filled),copy(self.will_be_emptied),copy(self.pending_actions))
    def guaranteed_world_collision(self,x,y,z):
        return self.occupancy[x,y,z] and not self.cell_will_change((x,y,z),False)
    def possible_world_collision(self,x,y,z):
        return self.occupancy[x,y,z] or self.cell_will_change((x,y,z),True)
    def current_world_collision(self,x,y,z):
        return self.occupancy[x,y,z]
    def maybe_legal(self,state)->bool:
        """
        return if the state is legal or may become so if the cells that can change do so
        """
        cell=(state.location.x,state.location.y,state.location.z)
        if state.location.in_bounds(*self.occupancy.shape):
            if not self.guaranteed_world_collision(*cell):
                if (cell[2]<=0 or self.possible_world_collision(cell[0],cell[1],cell[2]-1)):
                    for filled in state.cells_occupied():
                        if block_utils.in_bounds(*filled,*self.occupancy.shape):
                            if self.guaranteed_world_collision(*filled):
                                return False
                    return True
                else:
                    return False
            else:
                return False
        else:
            return False
    def maybe_can_transition_to(self,state1,state2):
        """
        check the transition to a feasible adjacent State is collision free under current occupancy grid, or if the cells that can change do so
        """
        if state2.location.turn!=state1.location.turn:
            for cell in state1.cells_swept(state2.location):
                if block_utils.in_bounds(*cell,*self.occupancy.shape):
                    if self.guaranteed_world_collision(*cell):
                        return False
            return True
        else:
            return True
    def legal(self,state)->Tuple[bool,int,Tuple[int,int,int],bool]:
        """
        return if the state is legal. If it isn't, return the index of the action in self.sequence responsible for it being illegal. 
        Returns -1 for index if legal, if the state is out of bounds, or if the state is illegal and the relevant cells have not been altered
        """
        cell=(state.location.x,state.location.y,state.location.z)
        if state.location.in_bounds(*self.occupancy.shape):
            if not self.current_world_collision(*cell):
                if (cell[2]<=0 or self.current_world_collision(cell[0],cell[1],cell[2]-1)):
                    for filled in state.cells_occupied():
                        if block_utils.in_bounds(*filled,*self.occupancy.shape):
                            if self.current_world_collision(*filled):
                                if self.cell_will_change(filled,False):
                                    #not legal, not because of a previous action, offending cell is filled, needs to be emptied
                                    return False,-1,filled,False
                                #not legal, because of a previous action, offending cell is filled, needs to be emptied
                                return False,self.last_modifier[*filled],filled,False
                    #is legal, so no previous action issue, no offending cell, no change to make
                    return True,-1,(-1,-1,-1),False
                else:
                    #missing the support block to stand upon
                    support=(cell[0],cell[1],cell[2]-1)
                    if self.cell_will_change(support,True):
                        #not legal, not because of a previous action, offending cell is support, needs to be filled
                        return False,-1,support,True
                    #not legal, because of a previous action, offending cell is support, needs to be filled
                    return False,self.last_modifier[cell[0],cell[1],cell[2]-1],support,True
            else:
                #cell is occupied
                if self.cell_will_change(cell,False):
                    #not legal, not because of a previous action, offending cell is cell, needs to be emptied
                    return False,-1,cell,False
                #not legal, because of a previous action, offending cell is cell, needs to be emptied
                return False,self.last_modifier[*cell],cell,False
        else:
            #out of bounds, this is just a bad action that shouldn't be tried!
            return False,-1,(-1,-1,-1),False
    def can_transition_to(self,state1,state2)->Tuple[bool,int,Tuple[int,int,int],bool]:
        """
        check the transition to a feasible adjacent State is collision free under current occupancy grid
        """
        if state2.location.turn!=state1.location.turn:
            for cell in state1.cells_swept(state2.location):
                if block_utils.in_bounds(*cell,*self.occupancy.shape):
                    if self.current_world_collision(*cell):
                        if self.cell_will_change(cell,False):
                            #not legal, not because of a previous action, offending cell is cell, needs to be emptied
                            return False,-1,cell,False
                        #not legal, because of a previous action, offending cell is cell, needs to be emptied
                        return False,self.last_modifier[*cell],cell,False
            #is legal, so no previous action issue, no offending cell, no change to make
            return True,-1,(-1,-1,-1),False
        else:
            return True,-1,(-1,-1,-1),False
class State(object):
    time:int
    location:Location
    block:block_utils.Block
    agent_mode:int
    def __init__(self, time, location, block=block_utils.CuboidalBlock(3),agent_mode=None):
        self.time = time
        self.location = location
        self.block = block
        if agent_mode is None:
            self.agent_mode = Agent.MODE_GOAL
        else:
            self.agent_mode = agent_mode
    def __eq__(self, other):
        return self.time == other.time and self.location == other.location and self.block==other.block
    def __hash__(self):
        return hash(tuple((self.time,self.location,self.block)))
    def cells_occupied(self):
        return self.location.cells_occupied()|self.block.cells_occupied(self.location.x,self.location.y,self.location.z+1,self.location.vertical())
    def cells_swept(self,destination:Location):
        if destination.in_world:
            return self.location.cells_occupied()|self.block.cells_swept(self.location.x,self.location.y,self.location.z+1,self.location.vertical(),destination.x,destination.y,destination.z+1,destination.vertical())|destination.cells_occupied()
        else:
            return self.cells_occupied()
    def is_equal_except_time(self, state):
        my_cells=self.cells_occupied()
        their_cells=state.cells_occupied()
        return any(c in their_cells for c in my_cells)
    def is_equal_except_time_edge(self, self_t2, other_t1, other_t2):
        if not self.time==other_t1.time:
            return False
        my_swept=self.cells_swept(self_t2.location)
        their_swept=other_t1.cells_swept(other_t2.location)
        return any(cell in their_swept for cell in my_swept)
        self_t1 = self
        edge1 = EdgeConstraint(self_t1.time, self_t1.location, self_t2.location)
        edge2 = EdgeConstraint(other_t1.time, other_t1.location, other_t2.location)
        return edge1.edge_collision(edge2)
    
    def is_equal_except_time_goal(self, state):
        return self.location == state.location and self.block==state.block
    def __repr__(self):
        return f"{self.location}:{self.block}@t={self.time}"
    def get_neighbors(self,world_size:Tuple[int,int,int]):
        x_dim,y_dim,z_dim=world_size
        for location in self.location.adjacent_locations(x_dim,y_dim,z_dim):
            yield Edge(self,MoveAction(self.block,location))
class RobotOutsideWorld(State):
    def __init__(self, time, block=block_utils.CuboidalBlock(3)):
        location=OutsideWorld()
        super().__init__(time, location, block)
    def cells_occupied(self):
        return frozenset()
    def cells_swept(self,destination:Location):
        if destination.in_world:
            return destination.cells_occupied()|self.block.cells_occupied(destination.x,destination.y,destination.z+1,destination.vertical())
        return set()
    def __repr__(self):
        return f"OutsideWorld:{self.block}@t={self.time}"
class Action:
    REMOVAL=-1
    MOVE=0
    PLACEMENT=1
    action_type_names=["MOVE","PLACE","REMOVE"]
    BLOCK_ACTIONS={PLACEMENT,REMOVAL}
    action_type:int
    location2:Location
    block:block_utils.Block
    required_block:block_utils.Block
    def __init__(self):
        raise NotImplementedError
    def __eq__(self,other):
        return self.action_type==other.action_type and self.location2==other.location2 and self.block==other.block
    def __hash__(self):
        return hash((self.action_type,self.location2,self.block))
    def __repr__(self):
        action_type=Action.action_type_names[self.action_type]
        return f"{action_type}To{self.location2}With{self.block}"
    def exit_state(self,state1:State)->State:
        raise NotImplementedError
    def inverse(self,state:State):
        raise NotImplementedError
    def involved_cells(self,state:State)->MutableSet[Tuple[int,int,int]]:
        raise NotImplementedError
    def required_occupied(self,state:State)->MutableSet[Tuple[int,int,int]]:
        raise NotImplementedError
    def required_unoccupied(self,state:State)->MutableSet[Tuple[int,int,int]]:
        raise NotImplementedError
    def feasible(self,world:WorldState)->Tuple[bool,int,Tuple[int,int,int],bool]:
        """
        determine if it is possible to execute the action in the specified WorldState
        If one exists, returns the index in world.sequence that changed a cell's state such that the action is infeasible
        """
        raise NotImplementedError
    def maybe_feasible(self,world:WorldState)->bool:
        raise NotImplementedError
    def executable_from(self,state:State,world:WorldState):
        """
        for an action that is feasible in world and a state that is legal in world, determine if executing action from state is possible
        Does NOT test if the initial or exit_state is legal, nor if the action is feasible at all!
        """
        raise NotImplementedError
    def maybe_executable_from(self,state:State,world:WorldState):
        raise NotImplementedError
    def cells_changed(self):
        return self.block.cells_occupied(self.location2.x,self.location2.y,self.location2.z,self.location2.vertical())
    def performable_from(self,state:State,world_size:Tuple[int,int,int])->bool:
        raise NotImplementedError
    def apply_to_world(self,world:WorldState)->bool:
        raise NotImplementedError
    def interferes_with_robot_at(self,state:State)->bool:
        raise NotImplementedError
    def possible_action_locations(self,world_size:Tuple[int,int,int])->MutableSet[Location]:
        raise NotImplementedError
    def cost(self,state:State)->int:
        raise NotImplementedError

class PlacementAction(Action):
    def __init__(self,block:block_utils.Block,destination:Location):
        self.action_type=Action.PLACEMENT
        self.location2=destination
        self.block=block
        self.required_block=block
    def exit_state(self,state1:State):
        return State(state1.time+1,state1.location,block_utils.NoBlock())
    def inverse(self, state: State):
        return PickupAction(self.block,self.location2)
    def involved_cells(self, state: State) -> MutableSet[Tuple[int]]:
        return state.cells_occupied()|self.cells_changed()
    def required_occupied(self, state: State) -> MutableSet[Tuple[int]]:
        place_to_stand=(state.location.x,state.location.y,max(0,state.location.z-1))
    def feasible(self,world:WorldState)->Tuple[bool,int,Tuple[int,int,int],bool]:
        return world.placeable_at(self.block,self.location2)
    def maybe_feasible(self,world:WorldState):
        return world.maybe_placeable_at(self.block,self.location2)
    def executable_from(self,state:State,world:WorldState):
        """
        for an action that is feasible in world and a compatible state that is feasible in world, determine if executing action from state is possible
        Does NOT test if the initial or exit_state is legal, nor if the action is feasible at all!
        """
        return True,-1,(-1,-1,-1),False
    def maybe_executable_from(self,state:State,world:WorldState):
        return True
    def performable_from(self, state: State,world_size:Tuple[int,int,int]) -> bool:
        return self.block.can_pickup_from(state.location.x,state.location.y,state.location.z,state.location.vertical(),self.location2.x,self.location2.y,self.location2.z,self.location2.vertical())
    def apply_to_world(self,world:WorldState)->bool:
        return world.add_block(self.block,self.location2)
    def interferes_with_robot_at(self, state: State) -> bool:
        affected=self.cells_changed()
        return len(state.cells_occupied()&affected)>0 or len(state.location.required_support_cells() & affected)>0
    def possible_action_locations(self,world_size:Tuple[int,int,int]) -> MutableSet[Location]:
        locs=set()
        for x,y,z,rotation in self.block.action_locations(self.location2.x,self.location2.y,self.location2.z,self.location2.vertical()):
            loc=Location(x,y,z,rotation*90)
            if loc.in_bounds(*world_size):
                locs.add(loc)
        return locs
    def cost(self, state: State) -> int:
        return 1
class PickupAction(Action):
    def __init__(self,block:block_utils.Block,source:Location):
        self.action_type=Action.REMOVAL
        self.location2=source
        self.block=block
        self.required_block=block_utils.NoBlock()
    def exit_state(self,state1:State):
        return State(state1.time+1,state1.location,self.block)
    def inverse(self, state: State):
        return PlacementAction(self.block,self.location2)
    def involved_cells(self, state: State) -> MutableSet[Tuple[int]]:
        return state.cells_occupied()|self.cells_changed()
    def feasible(self,world:WorldState)->Tuple[bool,int,Tuple[int,int,int],bool]:
        if (self.block,self.location2) in world.blocks:
            return True,-1,(-1,-1,-1),False
        sequence_length=len(world.sequence)
        for i in reversed(range(sequence_length)):
            addition,block,location=world.sequence[i]
            if not addition:
                if self.block==block and self.location2==location:
                    return False,i,(self.location2.x,self.location2.y,self.location2.z),True
        return False,-1,(self.location2.x,self.location2.y,self.location2.z),True#block was never placed in the first place!
    def maybe_feasible(self, world: WorldState) -> bool:
        return ((self.block,self.location2) in world.blocks or (True,self.block,self.location2) in world.pending_actions)
    def executable_from(self,state:State,world:WorldState):
        """
        for an action that is feasible in world and a compatible state that is feasible in world, determine if executing action from state is possible
        Does NOT test if the initial or exit_state is legal, nor if the action is feasible at all!
        """
        return True,-1,(-1,-1,-1),False
    def maybe_executable_from(self,state:State,world:WorldState):
        return True
    def performable_from(self, state: State,world_size:Tuple[int,int,int]) -> bool:
        return self.block.can_pickup_from(state.location.x,state.location.y,state.location.z,state.location.vertical(),self.location2.x,self.location2.y,self.location2.z,self.location2.vertical())
    def apply_to_world(self, world: WorldState) -> bool:
        return world.remove_block(self.block,self.location2)
    def interferes_with_robot_at(self, state: State) -> bool:
        affected=self.cells_changed()
        return len(state.cells_occupied()&affected)>0 or len(state.location.required_support_cells() & affected)>0
    def possible_action_locations(self,world_size:Tuple[int,int,int]) -> MutableSet[Location]:
        locs=set()
        for x,y,z,rotation in self.block.action_locations(self.location2.x,self.location2.y,self.location2.z,self.location2.vertical()):
            loc=Location(x,y,z,rotation*90)
            if loc.in_bounds(*world_size):
                locs.add(loc)
        return locs
    def cost(self, state: State) -> int:
        return 1
class MoveAction(Action):
    def __init__(self,block:block_utils.Block,destination:Location):
        self.action_type=Action.MOVE
        self.location2=destination
        self.block=block
        self.required_block=block
    def exit_state(self, state1: State):
        if self.location2.in_world:
            return State(state1.time+1,self.location2,state1.block)
        else:
            return RobotOutsideWorld(state1.time+1,state1.block)
    def involved_cells(self, state: State) -> MutableSet[Tuple[int]]:
        return state.cells_swept(self.location2)
    def inverse(self, state: State):
        return MoveAction(self.block,state.location)
    def feasible(self,world:WorldState)->Tuple[bool,int,Tuple[int,int,int],bool]:
        return True,-1,(-1,-1,-1),False
    def maybe_feasible(self, world: WorldState) -> bool:
        return True
    def executable_from(self,state:State,world:WorldState):
        """
        for an action that is feasible in world and a compatible state that is feasible in world, determine if executing action from state is possible
        Does NOT test if the initial or exit_state is legal, nor if the action is feasible at all!
        """
        return world.can_transition_to(state,self.exit_state(state))
    def maybe_executable_from(self,state:State,world:WorldState):
        return world.maybe_can_transition_to(state,self.exit_state(state))
    def performable_from(self, state: State,world_size:Tuple[int,int,int]) -> bool:
        if not self.location2.in_world:
            if state.location.is_boundary(*world_size):
                return True
        elif not state.location.in_world:
            if self.location2.is_boundary(*world_size):
                return True
        return (abs(self.location2.x-state.location.x)+abs(self.location2.y-state.location.y))<=1 and abs(self.location2.z-state.location.z)<=1
    def apply_to_world(self, world: WorldState) -> bool:
        return True
    def interferes_with_robot_at(self, state: State) -> bool:
        exit_state=self.exit_state()
        return state.is_equal_except_time(exit_state)
    def possible_action_locations(self,world_size:Tuple[int,int,int]) -> MutableSet[Location]:
        return set(self.location2.adjacent_locations(*world_size))
    def cost(self, state: State) -> int:
        if state.location.in_world or self.location2.in_world:
            return 1
        return 0
class Edge:
    state1:State
    state2:State
    edge_type:int
    involved_cells:MutableSet[Tuple[int,int,int]]
    goal_done:bool
    def __init__(self,state1:State,action:Action):
        self.state1=state1
        self.action=action
        self._involved_cells=None
        self.cost=self.action.cost(self.state1)
        self.goal_done=state1.agent_mode==Agent.MODE_LEAVE_WORLD
    @property
    def involved_cells(self):
        if self._involved_cells is None:
            self._involved_cells=self.action.involved_cells(self.state1)
        return self._involved_cells
    def __eq__(self,other):
        return self.action==other.action and self.state1==other.state1
    def equal_except_time(self,other):
        return self.action==other.action and self.state1.location==other.state1.location and self.state1.block==other.state1.block
    def __hash__(self):
        return hash((self.state1,self.action))
    def __repr__(self):
        return f"({self.state1}-{self.action})"
    def reversed(self):
        exit_state=self.exit_state()
        inverse_action=self.action.inverse(self.state1)
        state1=State(self.state1.time,exit_state.location,exit_state.block)
        return Edge(state1,inverse_action)
    def exit_state(self):
        state= self.action.exit_state(self.state1)
        if self.goal_done:
            state.agent_mode=Agent.MODE_LEAVE_WORLD
        return state
    def feasible(self,world:WorldState):
        """
        determine if it is possible to execute the action in the specified WorldState
        If one exists, returns the index in world.sequence that changed a cell's state such that the action is infeasible
        """
        return self.action.feasible(world)    
    def executable(self,world:WorldState):
        """
        for an action that is feasible in world and a compatible state that is feasible in world, determine if executing action from state is possible
        Does NOT test if the initial or exit_state is legal, nor if the action is feasible at all!
        """
        return self.action.executable_from(self.state1,world)
    def legal(self,world:WorldState):
        """
        test if the exit state of the action is legal in the world
        """
        return world.legal(self.action.exit_state(self.state1))
    def maybe_valid(self,world:WorldState):
        return world.maybe_legal(self.action.exit_state(self.state1)) and self.action.maybe_executable_from(self.state1,world) and self.action.maybe_feasible(world)
class EdgeWindowConstraint:
    def __init__(self,start,end,edge:Edge):
        self.start=start
        self.end=end
        self.edge=edge
    def __eq__(self,other):
        return self.start==other.start and self.end==other.end and self.edge.equal_except_time(other.edge)
    def __hash__(self):
        return hash((self.start,self.end,self.edge))
    def __repr__(self):
        return f"EWConstraint({self.edge} in [{self.start},{self.end}])"
    def violated(self,edge:Edge):
        return edge.state1.time>=self.start and edge.state1.time<=self.end and self.edge.equal_except_time(edge)
    def satisfied(self,path:List[Edge]):
        for i,edge in enumerate(path):
            if self.violated(edge):
                return False
        return True
class EdgeConstraint(EdgeWindowConstraint):
    def __init__(self, time, edge:Edge):
        self.start = time
        self.end = time
        self.edge=edge
    def violated(self,edge:Edge):
        return edge.state1.time==self.start and self.edge.equal_except_time(edge)
    def __repr__(self):
        return f"EConstraint({self.edge}@{self.start})"

class BlockActionWindowConstraint(EdgeWindowConstraint):
    def __init__(self,start:int,end:int,action:Action):
        self.start=start
        self.end=end
        self.edge=Edge(State(start,action.location2,action.block),action)#TODO this might compare equal with an edge involving placing or picking to an adjacent cell!
    def violated(self,edge:Edge):
        return edge.state1.time>=self.start and edge.state1.time<=self.end and edge.action==self.edge.action
    def __repr__(self):
        return f"BConstraint({self.edge.action} in [{self.start},{self.end}])"
        
class VertexWindowConstraint:
    """
    Only permit the robot to visit location outside the time window [start,end] inclusive
    """
    def __init__(self,start,end,location):
        self.start=start
        self.end=end
        self.location=location
    def __eq__(self, other):
        return self.start==other.start and self.end==other.end and self.location==other.location
    def collision(self,other:State):
        return (self.start<=other.time and self.end>=other.time) and self.location==other.location
    def __hash__(self):
        return hash((self.start,self.end,self.location))
    def __repr__(self):
        return f"VWConstraint({self.location} in [{self.start},{self.end}])"
    def satisfied(self,path:List[Edge]):
        return not any(self.collision(e.state1) for e in path) and not self.collision(path[-1].exit_state())
    
class VertexConstraint(VertexWindowConstraint):
    start:int
    end:int
    location:Location
    def __init__(self, time, location):
        self.start = time
        self.end = time
        self.location = location
    def collision(self,other:State):
        return self.start==other.time and self.location==other.location
    def __repr__(self):
        return f"VConstraint({self.location}@{self.start})"

class CellWindowConstraint:
    """
    Ban a robot or the block it is carrying from occupying a cell during a time window
    """
    start:int
    end:int
    cell:Tuple[int,int,int]
    def __init__(self,start,end,cell) -> None:
        self.start=start
        self.end=end
        self.cell=cell
    def __eq__(self,other):
        return self.start==other.start and self.end==other.end and self.cell==other.cell
    def __hash__(self):
        return hash((self.start,self.end,self.cell))
    def __repr__(self) -> str:
        return f"CellConstraint({self.cell}) in [{self.start},{self.end}]"
    def violated(self,edge:Edge):
        return (edge.state1.time>=self.start and edge.state1.time<=self.end) and self.cell in edge.involved_cells
    def satisfied(self,path:List[Edge]):
        for i,edge in enumerate(path):
            if self.violated(edge):
                return False
        return True

class CellSetConstraint:
    """
    Ban a robot or the block it is carrying from occupying an element of a set of cells
    """
    t:int
    cell_set:MutableSet[Tuple[int,int,int]]
    def __init__(self,t,cell_set:MutableSet[Tuple[int,int,int]]) -> None:
        self.t=t
        self.cell_set=cell_set
    def __eq__(self,other):
        return self.t==other.t and self.cell_set==other.cell_set
    def __repr__(self):
        return f"CellSet({self.cell_set}@{self.t})"
    def violated(self,edge:Edge):
        return edge.state1.time==self.t and any(c in self.cell_set for c in edge.involved_cells)
    def collision(self,state:State):
        return state.time==self.t and any(c in self.cell_set for c in state.cells_occupied())
    def satisfied(self,path:List[Edge]):
        for i,edge in enumerate(path):
            if self.violated(edge):
                return False
        return True
class Constraints(object):
    vertex_constraints:Dict[int,MutableSet[VertexWindowConstraint]]
    unbounded_vertex_constraints:Dict[int,MutableSet[VertexWindowConstraint]]
    edge_constraints:Dict[int,MutableSet[EdgeWindowConstraint]]
    unbounded_edge_constraints:Dict[int,MutableSet[EdgeWindowConstraint]]
    cell_constraints:Dict[int,CellSetConstraint]
    def __init__(self):
        self.vertex_constraints = defaultdict(set)
        self.edge_constraints = defaultdict(set)
        self.cell_constraints = dict()
        self.unbounded_vertex_constraints = defaultdict(set)
        self.unbounded_edge_constraints = defaultdict(set)
    def add_constraint(self, other):
        for t in other.vertex_constraints:
            self.vertex_constraints[t] |= other.vertex_constraints[t]
        for t in other.edge_constraints:
            self.edge_constraints[t] |= other.edge_constraints[t]
        for t in other.cell_constraints:
            self.add_c_constraint(other.cell_constraints[t])
        for t in other.unbounded_vertex_constraints:
            self.unbounded_vertex_constraints[t] |= other.unbounded_vertex_constraints[t]
        for t in other.unbounded_edge_constraints:
            self.unbounded_edge_constraints[t] |= other.unbounded_edge_constraints[t]
    def add_v_constraint(self,con:VertexWindowConstraint):
        if np.isfinite(con.start) and con.start>=0:
            if np.isfinite(con.end):
                for t in range(con.start,con.end+1):
                    self.vertex_constraints[t].add(con)
            else:
                #finite start, infinite end
                self.unbounded_vertex_constraints[con.start].add(con)
        else:
            raise ValueError(f"{con}'s start should be finite and non-negative")
        
    def add_e_constraint(self,con:EdgeWindowConstraint):
        if np.isfinite(con.start) and con.start>=0:
            if np.isfinite(con.end):
                for t in range(con.start,con.end+1):
                    self.edge_constraints[t].add(con)
            else:
                #finite start, infinite end
                self.unbounded_edge_constraints[con.start].add(con)
        else:
            raise ValueError(f"{con}'s start should be finite and non-negative")
    def add_c_constraint(self,con:CellSetConstraint):
        if con.t in self.cell_constraints:
            self.cell_constraints[con.t].cell_set |=con.cell_set
        else:
            self.cell_constraints[con.t]=CellSetConstraint(con.t,copy(con.cell_set))

    def get_unbounded_v_constraints(self,time:int)->MutableSet[VertexWindowConstraint]:
        cons=set()
        for t in range(time+1):
            cons.update(self.unbounded_vertex_constraints[t])
        return cons
    def get_v_constraints(self,time:int)->MutableSet[VertexWindowConstraint]:
        cons=self.get_unbounded_v_constraints(time)
        cons.update(self.vertex_constraints[time])
        return cons
    def get_unbounded_e_constraints(self,time:int)->MutableSet[EdgeWindowConstraint]:
        cons=set()
        for t in range(time+1):
            cons.update(self.unbounded_edge_constraints[t])
        return cons
    def get_e_constraints(self,time:int)->MutableSet[EdgeWindowConstraint]:
        cons=self.get_unbounded_e_constraints(time)
        cons.update(self.edge_constraints[time])
        return cons
    def __str__(self):
        return "VC: " + str([str(vc) for vc in self.vertex_constraints])  + \
            "EC: " + str([str(ec) for ec in self.edge_constraints]) + \
            "CC: " +str(self.cell_constraints)
    def __repr__(self):
        return f"{self.vertex_constraints},{self.edge_constraints},{self.cell_constraints}"
    def edge_valid(self,edge:Edge):
        t1=edge.state1.time
        if t1 in self.cell_constraints:
            if self.cell_constraints[t1].violated(edge):
                return False
        exit_state=edge.exit_state()
        t2=exit_state.time
        if t2 in self.cell_constraints:
            if self.cell_constraints[t2].collision(exit_state):
                return False
        econ=self.edge_constraints[t1]
        if any(e.violated(edge) for e in econ):
            return False
        unbounded_e_cons=self.get_unbounded_e_constraints(t1)
        if any(e.violated(edge) for e in unbounded_e_cons):
            return False
        
        vcon=self.vertex_constraints[exit_state.time]
        if any(v.collision(exit_state) for v in vcon):
            return False
        unbounded_v_cons=self.get_unbounded_v_constraints(exit_state.time)
        if any(v.collision(exit_state) for v in unbounded_v_cons):
            return False
        return True
    def satisfied(self,path:List[Edge]):
        t0=path[0].state1.time
        if t0>0:
            unbounded_v_cons=self.get_unbounded_v_constraints(t0-1)
            unbounded_e_cons=self.get_unbounded_e_constraints(t0-1)
        else:
            unbounded_v_cons=set()
            unbounded_e_cons=set()
        for edge in path:
            t=edge.state1.time
            if t in self.cell_constraints:
                if self.cell_constraints[t].violated(edge):
                    return False
            econ=self.edge_constraints[t]
            if any(e.violated(edge) for e in econ):
                return False
            unbounded_e_cons|=self.unbounded_edge_constraints[t]
            if any(e.violated(edge) for e in unbounded_e_cons):
                return False
            vcon=self.vertex_constraints[t]
            if any(v.collision(edge.state1) for v in vcon):
                return False
            unbounded_v_cons|=self.unbounded_vertex_constraints[t]
            if any(v.collision(edge.state1) for v in unbounded_v_cons):
                return False
        exit_state=path[-1].exit_state()
        t2=exit_state.time
        if t2 in self.cell_constraints:
            if self.cell_constraints[t2].collision(exit_state):
                return False
        vcon=self.vertex_constraints[exit_state.time]
        if any(v.collision(exit_state) for v in vcon):
            return False
        unbounded_v_cons|=self.unbounded_vertex_constraints[exit_state.time]
        if any(v.collision(exit_state) for v in unbounded_v_cons):
            return False
        return True
    def is_goal_possible(self,state:State,goal_action:Action,world:WorldState):
        action_locs=goal_action.possible_action_locations(world.occupancy.shape)
        possible_goal_edges=(Edge(State(state.time+1,action_loc,goal_action.required_block),goal_action) for action_loc in action_locs)
        for edge in possible_goal_edges:
            banned=False
            if not edge.maybe_valid(world):
                banned=True
                continue
            time=state.time+1
            for constraint in self.get_unbounded_v_constraints(time):
                #all children would have to respect this constraint
                if constraint.collision(edge.state1): 
                    banned=True
                    break
            if not banned:
                for constraint in self.get_unbounded_e_constraints(time):
                    #all children would have to respect this constraint
                        if constraint.violated(edge):
                            banned=True
                            break
                if not banned:
                    return True
        return False

class Conflict:
    def __repr__(self):
        raise NotImplementedError
    def get_constraints(self)->List[Dict[str,Constraints]]:
        raise NotImplementedError
class VertexConflict(Conflict):
    def __init__(self,time:int,agent1:str,location1:Location,agent2:str,location2:Location) -> None:
        self.time=time
        self.agent_1=agent1
        self.agent_2=agent2
        self.location1=location1
        self.location2=location2
    def __repr__(self):
        return f"VConflict({self.agent_1},{self.location1};{self.agent_2},{self.location2}@t={self.time})"
    def get_constraints(self) -> Dict[str, Constraints]:
        v1_constraint = VertexConstraint(self.time, self.location1)
        v2_constraint = VertexConstraint(self.time,self.location2)
        constraint1 = Constraints()
        constraint2 = Constraints()
        constraint1.add_v_constraint(v1_constraint)
        constraint2.add_v_constraint(v2_constraint)
        return [{self.agent_1:constraint1},{self.agent_2:constraint2}]
class EdgeConflict(Conflict):
    def __init__(self,time:int,agent1:str,edge1:Edge,agent2:str,edge2:Edge) -> None:
        self.time=time
        self.agent_1=agent1
        self.agent_2=agent2
        self.edge1=edge1
        self.edge2=edge2
    def __repr__(self):
        return f"EConflict({self.agent_1},{self.edge1};{self.agent_2},{self.edge2}@t={self.time})"
    def get_constraints(self) -> Dict[str, Constraints]:
        constraint1 = Constraints()
        constraint2 = Constraints()

        e_constraint1 = EdgeConstraint(self.time, self.edge1)
        e_constraint2 = EdgeConstraint(self.time, self.edge2)

        constraint1.add_e_constraint(e_constraint1)
        constraint2.add_e_constraint(e_constraint2)

        return [{self.agent_1:constraint1},{self.agent_2:constraint2}]

class WorldConflict(Conflict):
    """
    Conflict with the world state
    """
    time_1:int
    agent_1:str
    time_2:int
    agent_2:str
    action_2:Action
    pass
class RobotWorldConflict(WorldConflict):
    """
    Conflict where agent1 attempts to occupy location1 at time time1 but agent2 places or removes a block at time time2,
    and time1 and time2 are such that location1 cannot be occupied at time1 as a result

    i.e. four cases are contained here. 
    In the first, agent2 places a block in location1 at time2, time2<=time1
    In the second, agent2 removes a block from location1 at time2, time2>=time1
    In the third, agent 2 places the block below location1 at time2, time2>=time1, and the block was not present before time1
    In the fourth, agent 2 removes the block below location1 at time2, time2<=time1, and the block is not replaced before time1
    """
    def __init__(self,time1:int,agent1:str,location1:Location,time2:int,agent2:str,action2:Action):
        self.time_1=time1
        self.agent_1=agent1
        self.location_1=location1
        self.time_2=time2
        self.agent_2=agent2
        self.action_2=action2
    def __repr__(self):
        return f"RWConflict({self.agent_1},{self.location_1}@t1={self.time_1},{self.agent_2},{self.action_2}@{self.time_2})"
    def get_constraints(self) -> Dict[str, Constraints]:
        constraint1=Constraints()
        constraint2=Constraints()

        if self.time_1<=self.time_2:
            #aobstacle that should be removed or support that should be placed by time_1 wasn't
            #in one branch, force agent_1 to reach location_1 after time_2
            constraint1.add_v_constraint(VertexWindowConstraint(0,self.time_2,self.location_1))
            #in the other branch, we would force agent_2 to do its action before time_1
            #but if agent_2 could do this, it already would be. So we have an empty constraint2 that will lead to a skipped node
        else:
            #obstacle should have been placed or support removed after we use this cell
            #in one branch, force agent_1 to reach location_1 before time_2
            constraint1.add_v_constraint(VertexWindowConstraint(self.time_2,float("inf"),self.location_1))
            #in the other branch, force agent_2 to act after time_1
            constraint2.add_e_constraint(BlockActionWindowConstraint(0,self.time_1,self.action_2))
        options=list()
        if self.agent_1 is not None:
            options.append({self.agent_1:constraint1})
        if self.agent_2 is not None:
            options.append({self.agent_2:constraint2})
        return options

class ActionWorldConflict(WorldConflict):
    """
    Conflict where an Action by agent 1 is not feasible in the world at t1 because of an action by agent 2 at t2
    """
    def __init__(self,time1:int,agent1:str,action1:Action,time2:int,agent2:str,action2:Action) -> None:
        self.time_1=time1
        self.agent_1=agent1
        self.action_1=action1
        self.time_2=time2
        self.agent_2=agent2
        self.action_2=action2
    def __repr__(self):
        return f"AWConflict({self.agent_1},{self.action_1}@t1={self.time_1},{self.agent_2},{self.action_2}@{self.time_2})"
    def get_constraints(self) -> Dict[str, Constraints]:
        constraint1=Constraints()
        constraint2=Constraints()

        if self.time_1<self.time_2:
            #in one branch, force agent_1 to act after time_2
            constraint1.add_e_constraint(BlockActionWindowConstraint(0,self.time_2,self.action_1))
            #in the other branch, we force agent_2 to do its action before time_1
            constraint2.add_e_constraint(BlockActionWindowConstraint(self.time_1,float("inf"),self.action_2))
        else:
            #in one branch, force agent_1 to act location_1 before time_2
            constraint1.add_e_constraint(BlockActionWindowConstraint(self.time_2,float("inf"),self.action_1))
            #in the other branch, force agent_2 to act after time_1
            constraint2.add_e_constraint(BlockActionWindowConstraint(0,self.time_1,self.action_2))
        options=list()
        if self.agent_1 is not None:
            options.append({self.agent_1:constraint1})
        if self.agent_2 is not None:
            options.append({self.agent_2:constraint2})
        return options
    
class EdgeWorldConflict(WorldConflict):
    """
    Conflict where agent 1 following an edge is not feasible in the world at t1 because of an action by agent 2 at t2
    """
    def __init__(self,time1:int,agent1:str,edge1:Edge,time2:int,agent2:str,action2:Action) -> None:
        self.time_1=time1
        self.agent_1=agent1
        self.edge_1=edge1
        self.time_2=time2
        self.agent_2=agent2
        self.action_2=action2
    def __repr__(self):
        return f"EWConflict({self.agent_1},{self.edge_1},{self.agent_2},{self.action_2}@{self.time_2})"
    def get_constraints(self) -> Dict[str, Constraints]:
        constraint1=Constraints()
        constraint2=Constraints()

        if self.time_1<self.time_2:
            #in one branch, force agent_1 to act after time_2
            constraint1.add_e_constraint(EdgeWindowConstraint(0,self.time_2,self.edge_1))
            #in the other branch, we force agent_2 to do its action before time_1
            constraint2.add_e_constraint(BlockActionWindowConstraint(self.time_1,float("inf"),self.action_2))
        else:
            #in one branch, force agent_1 to act location_1 before time_2
            constraint1.add_e_constraint(EdgeWindowConstraint(self.time_2,float("inf"),self.edge_1))
            #in the other branch, force agent_2 to act after time_1
            constraint2.add_e_constraint(BlockActionWindowConstraint(0,self.time_1,self.action_2))
        options=list()
        if self.agent_1 is not None:
            options.append({self.agent_1:constraint1})
        if self.agent_2 is not None:
            options.append({self.agent_2:constraint2})
        return options
class Agent:
    start:State
    goal:Action
    name:str
    committed_action_sequence:List[Edge]
    agent_constraints:Constraints
    earliest_goal_time:int
    MODE_GOAL=1
    MODE_LEAVE_WORLD=0
    def __init__(self,name,start,goal_action,committed_action_sequence=None,agent_constraints=None,earliest_goal_time=0) -> None:
        self.name=name
        self.start=start
        self.goal=goal_action
        if committed_action_sequence is None:
            self.committed_action_sequence=list()
        else:
            self.committed_action_sequence=committed_action_sequence
        if agent_constraints is None:
            self.agent_constraints=Constraints()
        else:
            self.agent_constraints=agent_constraints
        self.earliest_goal_time=earliest_goal_time
    def __repr__(self):
        return f"Agent {self.name}: {self.start}->{self.goal}"
    
def leave_world_heuristic(location:Location,world_size:Tuple[int,int,int]):
    #estimate cost of shortest path to a boundary cell
    facing_horizontal=int(location.vertical())
    x_dist = min(location.x,world_size[0]-location.x-1)
    y_dist = min(location.y,world_size[1]-location.y-1)
    if x_dist>0 and not facing_horizontal:
        x_cost=x_dist+1
    else:
        x_cost=x_dist
    if y_dist>0 and facing_horizontal:
        y_cost=y_dist+1
    else:
        y_cost=y_dist
    return min(x_cost,y_cost)+1#add 1 for actually leaving the world once boundary is reached
    
def travel_distance_heuristic(location1:Location,location2:Location,world_size:Tuple[int,int,int]):
    if not location1.in_world:
        return leave_world_heuristic(location2,world_size)
    elif not location2.in_world:
        return leave_world_heuristic(location1,world_size)
    else:
        x_dist = abs(location1.x - location2.x)
        y_dist = abs(location1.y - location2.y)
        facing_horizontal=location1.vertical()
        will_face_vertical=not facing_horizontal
        will_face_horizontal=facing_horizontal
        turn_cost=0
        if x_dist!=0 and not facing_horizontal:
            turn_cost+=1
            will_face_horizontal=True
        if y_dist!=0 and facing_horizontal:
            turn_cost+=1
            will_face_vertical=True
        end_facing_horizontal=location2.vertical()
        if (end_facing_horizontal and not will_face_horizontal) or (not end_facing_horizontal and not will_face_vertical):
            turn_cost+=1
        return x_dist + y_dist + turn_cost

def act_and_leave_world_heuristic(state:State,earliest_goal_time:int,goal:Action,world_size:Tuple[int,int,int],makespan_lb=0):
    if state.agent_mode==Agent.MODE_GOAL:
        action_locs=goal.possible_action_locations(world_size)
        goal_costs=[travel_distance_heuristic(state.location,aloc,world_size)+1 for aloc in action_locs]
        leave_world_costs=[leave_world_heuristic(aloc,world_size) for aloc in action_locs]
        cost_estimate=min(g+l for g,l in zip(goal_costs,leave_world_costs))
        min_leave_world=min(leave_world_costs)
        return max(earliest_goal_time-state.time+min_leave_world,makespan_lb-state.time,cost_estimate),cost_estimate
    else:
        leave_world_cost=leave_world_heuristic(state.location,world_size)
        return max(leave_world_cost,makespan_lb-state.time),leave_world_cost

class Environment(object):
    dimension:Tuple[int,int,int]
    agent_dict:Dict[str,Agent]
    constraints:Constraints
    constraint_dict:Dict[str,Constraints]
    a_star:AStar
    _world_state:WorldState
    initial_world:NDArray
    scheduled_block_actions:Dict[int,List[Action]]
    committed_block_actions:Dict[int,List[Action]]
    scheduled_goal_action:Tuple[int,Action]
    def __init__(self, dimension, agent_dict:Dict[str,Agent], initial_world_state:WorldState=None,universal_constraints:Constraints=None,scheduled_edges:List[Edge]=None):
        self.dimension = dimension
        self.agent_dict = agent_dict

        if initial_world_state is None:
            copied_world_state=WorldState([],set(),np.zeros(self.dimension,dtype=np.bool_))
        else:
            copied_world_state=initial_world_state.copy()
        self._world_state_with_pendings=copied_world_state.copy()
        self.committed_block_actions=defaultdict(list)
        for agent in self.agent_dict.values():
            goal=agent.goal
            if goal.action_type in Action.BLOCK_ACTIONS:
                copied_world_state.add_pending_action(goal.action_type==Action.PLACEMENT,goal.block,goal.location2)
            for edge in agent.committed_action_sequence:
                if edge.action.action_type in Action.BLOCK_ACTIONS:
                    self._world_state_with_pendings.add_pending_action(edge.action.action_type==Action.PLACEMENT,edge.action.block,edge.action.location2)
                    self.committed_block_actions[edge.state1.time].append(edge.action)
        self.scheduled_edges=defaultdict(list)
        self.scheduled_block_actions=defaultdict(list)
        self.max_scheduled_time=0
        if scheduled_edges is not None:
            for edge in scheduled_edges:
                self.scheduled_edges[edge.state1.time].append(edge)
                self.max_scheduled_time=edge.state1.time if edge.state1.time>self.max_scheduled_time else self.max_scheduled_time
                if edge.action.action_type in Action.BLOCK_ACTIONS:
                    self.scheduled_block_actions[edge.state1.time].append(edge.action)
                    self._world_state_with_pendings.add_pending_action(edge.action.action_type==Action.PLACEMENT,edge.action.block,edge.action.location2)
        self._world_state=copied_world_state
        self._worlds=defaultdict(self.new_world_state_list)
        self._worlds_with_pendings=defaultdict(self.new_pending_world_state_list)
        self.scheduled_goal_action=(None,None)

        self.constraints = Constraints()
        self.constraint_dict = {}
        if universal_constraints is None:
            self.universal_constraints = Constraints()
        else:
            self.universal_constraints = universal_constraints
        #add edge constraints relating to scheduled actions to the universal_constraints:
        for t in self.scheduled_edges:
            for edge in self.scheduled_edges[t]:
                if edge.action.action_type==Action.MOVE and (edge.state1.location.in_world or edge.action.location2.in_world):
                    self.universal_constraints.add_e_constraint(EdgeConstraint(t,edge))
        self.makespan_lb=0
        
        self.subroutine_calls=dict()
        self.subroutine_times=dict()

        self.get_edges=self.register_subroutine(self.get_edges,"get_edges")
        self.edge_valid=self.register_subroutine(self.edge_valid,"edge_valid")
        self.get_world_at_t=self.register_subroutine(self.get_world_at_t,"get_world_at_t")
        self.compute_solution=self.register_subroutine(self.compute_solution,"compute_solution")
        self.compute_solution_cost=self.register_subroutine(self.compute_solution_cost,"compute_solution_cost")
        self.get_all_conflicts=self.register_subroutine(self.get_all_conflicts,"get_all_conflicts")
        self.choose_conflict_by_type=self.register_subroutine(self.choose_conflict_by_type,"choose_conflict_by_type")
        self.is_goal_possible=self.register_subroutine(self.is_goal_possible,"is_goal_possible")
        self.get_neighbors=self.register_subroutine(self.get_neighbors,"get_neighbors")
        self.performable_from=self.register_subroutine(self.performable_from,"performable_from")
        self.a_star = AStar(self)
        self.a_star_search=self.register_subroutine(self.a_star.search,"a_star_search")
    def register_subroutine(self,function,name):
        self.subroutine_calls[name]=0
        self.subroutine_times[name]=0.0
        @wraps(function)
        def subroutine(*args,**kwargs):
            self.subroutine_calls[name]+=1
            s=timeit.default_timer()
            output=function(*args,**kwargs)
            e=timeit.default_timer()
            self.subroutine_times[name]+=e-s
            return output
        return subroutine
    def reset_subroutine_logs(self):
        for key in self.subroutine_calls:
            self.subroutine_calls[key]=0
            self.subroutine_times[key]=0.0
    def is_goal_possible(self,state:State,goal_action:Action,world:WorldState):
        return self.constraints.is_goal_possible(state,goal_action,world)
    def performable_from(self,goal_action:Action,state:State):
        return goal_action.performable_from(state,self.dimension)
    def get_neighbors(self,state:State):
        return state.get_neighbors(self.dimension)
    def new_world_state_list(self):
        return [self._world_state]
    def new_pending_world_state_list(self):
        return [self._world_state_with_pendings]
    def get_world_at_t(self,time:int,include_pending=False):
        if not include_pending:
            storage=self._worlds[self.scheduled_goal_action]
        else:
            storage=self._worlds_with_pendings[self.scheduled_goal_action]
        tstart=len(storage)-1
        if time<=tstart:
            return storage[time]
        else:
            world=storage[-1]

        for t in range(tstart,time):
            world=world.copy()
            if t==self.scheduled_goal_action[0]:
                self.scheduled_goal_action[1].apply_to_world(world)
            for action in self.scheduled_block_actions[t]:
                action.apply_to_world(world)
            for action in self.committed_block_actions[t]:
                action.apply_to_world(world)
            storage.append(world)
        return world
    def apply_scheduled_actions(self,time:int,world:WorldState,actions:Dict[Action,Tuple[int,str]]):
        for action in self.scheduled_block_actions[time]:
            action.apply_to_world(world)
            actions[action]=(time,None)
        return world,actions
    def achieved_world(self):
        world=self._world_state.copy()
        for agent in self.agent_dict:
            goal=self.agent_dict[agent].goal
            if goal.action_type== Action.PLACEMENT:
                world.add_block(goal.block,goal.location2)
            elif goal.action_type==Action.REMOVAL:
                world.remove_block(goal.block,goal.location2)
        return world
    def get_block_actions(self,solution:Dict[str,List[Edge]]):
        return {agent:[edge for edge in solution[agent] if edge.action.action_type in {Action.PLACEMENT,Action.REMOVAL}] for agent in solution}
    
    def get_edges(self, state:State, agent_name:str)->List[Edge]:
        world=self.get_world_at_t(state.time).copy()
        goal_action=self.agent_dict[agent_name].goal
        world.remove_pending_action(goal_action.action_type==Action.PLACEMENT,goal_action.block,goal_action.location2)
        if state.agent_mode==Agent.MODE_GOAL:
            if self.performable_from(goal_action,state):
                goal_edge=Edge(state,goal_action)
                goal_edge.goal_done=True
                if self.edge_valid(goal_edge,world):
                    return [goal_edge]
            #test if goal is infeasible under constraints
            world_with_future_blocks_marked=self.get_world_at_t(state.time,True).copy()
            world_with_future_blocks_marked.remove_pending_action(goal_action.action_type==Action.PLACEMENT,goal_action.block,goal_action.location2)
            if not self.is_goal_possible(state,goal_action,world_with_future_blocks_marked):
                #dead end!
                return []
        neighbor_candidates = self.get_neighbors(state)
        neighbors=[]
        for e in neighbor_candidates:
            if self.edge_valid(e,world):
                neighbors.append(e)
        return neighbors

    def verify_solution(self,solution:Dict[str,List[Edge]])->Tuple[bool,dict]:
        """
        check that solution accomplishes goals and is all feasible and collision free
        """
        if len(solution)!=len(self.agent_dict):
            return False,{"reason":"Empty solution"}
        max_t = max(max([plan[-1].exit_state().time for plan in solution.values()]),self.max_scheduled_time+1)
        #first confirm goals are met
        for agent in solution:
            goal=self.agent_dict[agent].goal
            if not any(edge.action==goal for edge in solution[agent]):
                return False,{"reason":"Goal not met","agent":agent,"edge":solution[agent][-1],"world":self._world_state,"goal":goal}
        #next check for validity of actions in sequence
        actions=dict()
        world=self._world_state
        for t in range(max_t):
            for agent in solution:
                edge=self.get_edge(agent,solution,t)
                if not edge.action.performable_from(edge.state1,self.dimension):
                    return False,{"reason":"action not performable","agent":agent,"edge":edge,"world":world,"goal":goal}
                if not edge.feasible(world):
                    return False,{"reason":"edge not feasible","agent":agent,"edge":edge,"world":world,"goal":goal}
                if not edge.executable(world):
                    return False,{"reason":"edge not executable","agent":agent,"edge":edge,"world":world,"goal":goal}
                if not edge.legal(world):
                    return False,{"reason":"edge exit state not legal before world changes","agent":agent,"edge":edge,"world":world,"goal":goal}
            #update the world
            world,actions=self.update_world(t,world,solution,actions)
            #check exit states are feasible
            for agent in solution:
                edge=self.get_edge(agent,solution,t)
                exit_state=edge.exit_state()
                if not world.legal(exit_state):
                    return False,{"reason":"exit state not legal after world changes","agent":agent,"edge":edge,"world":world,"goal":goal}
        return True,{"reason":"Ok"}
    def get_first_conflict(self, solution:Dict[str,List[Edge]])->Conflict|Literal[False]:
        max_t = max(max([plan[-1].exit_state().time for plan in solution.values()]),self.max_scheduled_time+1)
        world=self._world_state
        world_history=[world]
        actions=dict()
        for t in range(max_t):
            result=self.get_vertex_conflicts_at_t(t,solution,True)
            if len(result)>0:
                return result[0]
            result=self.get_edge_conflicts_at_t(t,solution,True)
            if len(result)>0:
                return result[0]
            #world changes due to one agent such that other agent's action is not feasible at the time it was taken
            #to avoid depending on the order of execution at a given timestep, we return a conflict if a MOVE action is not feasible
            #in the world state BEFORE OR AFTER the block actions at time t are taken
            result=self.find_conflicts_in_moves_at_t(t,world,solution,actions)
            if len(result)>0:
                return result[0]
            #apply world modifications
            world,actions=self.update_world(t,world,solution,actions)
            #check move actions again!
            result=self.find_conflicts_in_moves_at_t(t,world,solution,actions)
            if len(result)>0:
                return result[0]
                    
            world_history.append(world)

        return False
    
    def get_all_conflicts(self,solution:Dict[str,List[Edge]]):
        max_t = max(max([plan[-1].exit_state().time for plan in solution.values()]),self.max_scheduled_time+1)

        world=self._world_state
        world_history=[world]
        conflicts=[]
        actions=dict()
        for t in range(max_t):
            result=self.get_vertex_conflicts_at_t(t,solution,False)
            conflicts.extend(result)
            result=self.get_edge_conflicts_at_t(t,solution,False)
            conflicts.extend(result)
            #world changes due to one agent such that other agent's action is not feasible at the time it was taken
            #to avoid depending on the order of execution at a given timestep, we return a conflict if a MOVE action is not feasible
            #in the world state BEFORE OR AFTER the block actions at time t are taken
            result=self.find_conflicts_in_moves_at_t(t,world,solution,actions,False)
            conflicts.extend(result)
            #apply world modifications
            try:
                world,actions=self.update_world(t,world,solution,actions)
            except UnhandledConflictError as e:
                if len(conflicts)>0:
                    return conflicts
                else:
                    raise e
            #check move actions again!
            result=self.find_conflicts_in_moves_at_t(t,world,solution,actions,False)
            conflicts.extend(result)
                    
            world_history.append(world)

        return conflicts
    
    def choose_conflict_by_type(self,conflicts:List[Conflict]):
        """
        Return a conflict, prioritizing ActionWorldConflict>RobotWorldConflict>EdgeWorldConflict>VertexConflict>EdgeConflict
        """
        first_AW=None
        first_RW=None
        first_EW=None
        first_E=None
        first_V=None
        for con in conflicts:
            if first_AW is None and isinstance(con,ActionWorldConflict):
                first_AW=con
                return first_AW
            if first_RW is None and isinstance(con,RobotWorldConflict):
                first_RW=con
            if first_EW is None and isinstance(con,EdgeWorldConflict):
                first_EW=con
            if first_E is None and isinstance(con,EdgeConflict):
                first_E=con
            if first_V is None and isinstance(con,VertexConflict):
                first_V=con
        if first_AW is not None:
            return first_AW
        if first_RW is not None:
            return first_RW
        if first_EW is not None:
            return first_EW
        if first_V is not None:
            return first_V
        if first_E is not None:
            return first_E
        return None
    
    def get_vertex_conflicts_at_t(self,t,solution,return_first=True):
        #two agents try to occupy same cell at same time
        conflicts=[]
        for agent_1, agent_2 in combinations(solution.keys(), 2):
            state_1 = self.get_state(agent_1, solution, t)
            state_2 = self.get_state(agent_2, solution, t)
            if state_1.is_equal_except_time(state_2):
                result=VertexConflict(t,agent_1,state_1.location,agent_2,state_2.location)
                if return_first:
                    return [result]
                else:
                    conflicts.append(result)
        return conflicts

    def get_edge_conflicts_at_t(self,t,solution,return_first=True):
        #two agents collide while moving
        conflicts=[]
        for agent_1, agent_2 in combinations(solution.keys(), 2):
            state_1a = self.get_state(agent_1, solution, t)
            state_1b = self.get_state(agent_1, solution, t+1)

            state_2a = self.get_state(agent_2, solution, t)
            state_2b = self.get_state(agent_2, solution, t+1)
            #tried to swap locations
            clear_edge = (state_1a.is_equal_except_time(state_2b) and state_1b.is_equal_except_time(state_2a))
            #rotated in place and intersected as a result
            turn_edge = state_1a.is_equal_except_time_edge(state_1b, state_2a, state_2b)
            if clear_edge or turn_edge:
                # print("clear edge") if clear_edge else print("turn edge")
                edge1=self.get_edge(agent_1,solution,t)
                edge2=self.get_edge(agent_2,solution,t)
                result=EdgeConflict(t,agent_1,edge1,agent_2,edge2)
                if return_first:
                    return [result]
                else:
                    conflicts.append(result)
        return conflicts
    def update_world(self,t:int,world:WorldState,solution:Dict[str,List[Edge]],actions:Dict[Action,Tuple[int,str]]):
        world=world.copy()
        for agent in solution:
            edge=self.get_edge(agent,solution,t)
            if edge.action.action_type == Action.PLACEMENT:
                success=world.add_block(edge.action.block,edge.action.location2)
            elif edge.action.action_type == Action.REMOVAL:
                success=world.remove_block(edge.action.block,edge.action.location2)
            else:
                continue
            if not success:
                #conflict between current block action and a preceding block action!
                #if this occurs, a WConflict should have been found
                raise UnhandledConflictError(f"Cannot perform {edge.action} in world at t={t}")
            else:
                actions[edge.action]=(t,agent)
        world,actions=self.apply_scheduled_actions(t,world,actions)
        return world,actions
    def find_conflicts_in_moves_at_t(self,t:int,world:WorldState,solution:Dict[str,List[Edge]],actions:Dict[Action,Tuple[int,str]],return_first=True):
        conflicts=[]
        for agent in solution:
            edge=self.get_edge(agent,solution,t)
            conflict=self.get_edge_world_conflict(agent,t,edge,world,solution,actions)
            if conflict is not None:
                conflicts.append(conflict)
                if return_first:
                    return conflicts
        for edge in self.scheduled_edges[t]:
            conflict=self.get_edge_world_conflict(None,t,edge,world,solution,actions)
            if conflict is not None:
                conflicts.append(conflict)
                if return_first:
                    return conflicts
        return conflicts
    def get_edge_world_conflict(self,agent,t,edge,world,solution,actions):
        feasible,interference_index,cell,addition=edge.feasible(world)
        if not feasible:
            #action-block conflict!
            interfering_time,interfering_agent,interfering_action=self.get_interfering_action_for_cell(t,cell,interference_index,addition,world,solution,actions)
            if interfering_agent==agent and t==interfering_time:
                #this gets called before and after we execute the actions at t, so avoid returning conflicts with your own action!
                return None
            conflict=ActionWorldConflict(t+1,agent,edge.action,interfering_time,interfering_agent,interfering_action)
            return conflict
        else:
            executable,interference_index,cell,addition=edge.executable(world)
            if not executable:
                #edge-block conflict
                interfering_time,interfering_agent,interfering_action=self.get_interfering_action_for_cell(t,cell,interference_index,addition,world,solution,actions)
                if interfering_agent==agent and t==interfering_time:
                    #this gets called before and after we execute the actions at t, so avoid returning conflicts with your own action!
                    return None
                conflict=EdgeWorldConflict(t+1,agent,edge,interfering_time,interfering_agent,interfering_action)
                return conflict
            legal,interference_index,cell,addition=edge.legal(world)
            if not legal:
                #exit state-block conflict
                interfering_time,interfering_agent,interfering_action=self.get_interfering_action_for_cell(t,cell,interference_index,addition,world,solution,actions)
                if interfering_agent==agent and t==interfering_time:
                    #this gets called before and after we execute the actions at t, so avoid returning conflicts with your own action!
                    return None
                conflict=RobotWorldConflict(t+1,agent,edge.action.location2,interfering_time,interfering_agent,interfering_action)
                return conflict
        return None
    
    def get_interfering_action_for_cell(self,t:int,cell:Tuple[int,int,int],interference_index:int,addition:bool,world:WorldState,solution:Dict[str,List[Edge]],actions:Dict[Action,Tuple[int,str]]):
        if interference_index==-1:
            #action that has NOT happened needs to happen
            if cell!=(-1,-1,-1):
                #find pending block action that would place/remove the cell
                if addition:
                    action_type=Action.PLACEMENT
                else:
                    action_type=Action.REMOVAL
                interfering_action,interfering_time,interfering_agent=self.find_pending_action_affecting_cell(t,cell,action_type,solution)
                #action is in the future, so we cannot attempt to use the cell until interfering_time+1
                interfering_time+=1
            else:
                raise UnhandledConflictError("Action that has not happened needs to happen")
        else:
            addition,bad_block,bad_location=world.sequence[interference_index]
            if addition:
                interfering_action=PlacementAction(bad_block,bad_location)
            else:
                interfering_action=PickupAction(bad_block,bad_location)
            interfering_time,interfering_agent=actions[interfering_action]
        return interfering_time,interfering_agent,interfering_action
    
    def find_pending_action_affecting_cell(self,t:int,cell:Tuple[int,int,int],action_type:int,solution:Dict[str,List[Edge]]):
        for agent in solution:
            goal=self.agent_dict[agent].goal
            if goal.action_type==action_type:
                #right kind of block action
                action_time=self.goal_completion_time(agent,solution)
                if action_time>=t:
                    #it's at the current time or afterwards
                    if cell in goal.cells_changed():
                        #it affects the cell
                        return goal,action_time,agent
        for time in self.committed_block_actions:
            if time>=t:
                for action in self.committed_block_actions[time]:
                    if action.action_type==action_type and cell in action.cells_changed():
                        return action,time,None
        for time in self.scheduled_block_actions:
            if time>=t:
                for action in self.scheduled_block_actions[time]:
                    if action.action_type==action_type and cell in action.cells_changed():
                        return action,time,None
        raise UnhandledConflictError(f"No {Action.action_type_names[action_type]} action at t>={t} affecting {cell}")
            

    def get_state(self, agent_name:str, solution:Dict[str,List[Edge]], t:int):
        tprime=solution[agent_name][0].state1.time
        delta_t=t-tprime
        if delta_t>=0 and delta_t<len(solution[agent_name]):
            edge=solution[agent_name][delta_t]
            return edge.state1
        elif delta_t>=len(solution[agent_name]):
            state1=solution[agent_name][-1].exit_state()
            return state1
        elif delta_t<0 and -delta_t<=len(self.agent_dict[agent_name].committed_action_sequence):
            edge=self.agent_dict[agent_name].committed_action_sequence[delta_t]
            return edge.state1
        else:
            raise ValueError(f"Time {t} is not present in either the committed action sequence or the plan of agent {agent_name}")
        
    def get_edge(self,agent_name:str,solution:Dict[str,List[Edge]],t:int):
        tprime=solution[agent_name][0].state1.time
        delta_t=t-tprime
        if delta_t>=0 and delta_t<len(solution[agent_name]):
            edge=solution[agent_name][delta_t]
        elif delta_t>=len(solution[agent_name]):
            state1=solution[agent_name][-1].exit_state()
            edge=Edge(state1,MoveAction(state1.block,state1.location))
        elif delta_t<0 and -delta_t<=len(self.agent_dict[agent_name].committed_action_sequence):
            edge=self.agent_dict[agent_name].committed_action_sequence[delta_t]
        else:
            raise ValueError(f"Time {t} is not present in either the committed action sequence or the plan of agent {agent_name}")
        return edge
    
    def edge_valid(self,edge:Edge,world:WorldState):
        valid = edge.maybe_valid(world)
        return valid and self.constraints.edge_valid(edge)
    
    def is_solution(self, agent_name):
        pass

    def admissible_heuristic(self, state:State, agent_name:str):
        return act_and_leave_world_heuristic(state,self.agent_dict[agent_name].earliest_goal_time,self.agent_dict[agent_name].goal,self.dimension,self.makespan_lb)

    def is_at_goal(self, edge:Edge, agent_name:str):
        return edge.state1.agent_mode==Agent.MODE_LEAVE_WORLD and not edge.exit_state().location.in_world
        if self.agent_modes[agent_name]==Agent.MODE_GOAL:
            goal_action = self.agent_dict[agent_name].goal
            if edge.action==goal_action:
                return True
            else:
                return False
        elif self.agent_modes[agent_name]==Agent.MODE_LEAVE_WORLD:
            if not edge.exit_state().location.in_world:
                return True
            else:
                return False
        else:
            raise ValueError(f"Unrecognized Agent mode {self.agent_modes[agent_name]}")
        
    def goal_completion_time(self,agent_name:str,solution:Dict[str,List[Edge]]):
        goal=self.agent_dict[agent_name].goal
        for edge in self.agent_dict[agent_name].committed_action_sequence:
            if edge.action==goal:
                return edge.state1.time
        for edge in solution[agent_name]:
            if edge.action==goal:
                return edge.state1.time
        return None

    def compute_solution(self,start_time,time_limit,existing_solution=None):
        if existing_solution is None:
            existing_solution=dict()
        solution = {}
        for agent in self.agent_dict.keys():
            self.constraints = self.constraint_dict.setdefault(agent, Constraints())
            self.constraints.add_constraint(self.universal_constraints)
            self.constraints.add_constraint(self.agent_dict[agent].agent_constraints)
            if agent in existing_solution and self.constraints.satisfied(existing_solution[agent]):
                solution[agent]=existing_solution[agent]
                continue
            start_state=self.agent_dict[agent].start
            if any(v.collision(start_state) for v in self.constraints.get_v_constraints(start_state.time)):
                return False
            start_state.agent_mode=Agent.MODE_GOAL
            path,final_state = self.a_star_search(self.agent_dict[agent].start,agent,start_time,time_limit)
            if not path:
                return False
            # self.agent_modes[agent]=Agent.MODE_LEAVE_WORLD
            # self.scheduled_goal_action=(do_action_path[-1].state1.time,do_action_path[-1].action)
            # leave_world_path,final_state= self.a_star_search(intermediate_state,agent,start_time,time_limit)
            # self.scheduled_goal_action=(None,None)
            # if not leave_world_path:
            #     return False
            solution.update({agent:path})
        return solution

    def compute_solution_cost(self, solution:Dict[str,List[Edge]]):
        return max(path[-1].exit_state().time for path in solution.values()),sum(e.cost for path in solution.values() for e in path)

class HighLevelNode(object):
    solution:Dict[str,List[State]]
    constraint_dict:Dict[str,Constraints]
    conflicts:List[Conflict]
    cost:int
    def __init__(self):
        self.solution = {}
        self.constraint_dict = {}
        self.conflicts = list()
        self.cost = 0

    def __eq__(self, other):
        if not isinstance(other, type(self)): return NotImplemented
        return self.solution == other.solution and self.cost == other.cost

    def __lt__(self, other):
        return self.cost < other.cost or (self.cost==other.cost and len(self.conflicts)<len(other.conflicts))

class CBS(object):
    env:Environment
    open_list:List[HighLevelNode]
    def __init__(self, environment):
        self.env = environment
        self.open_list = list()
        self.closed_list = list()
    def init_statistics(self):
        self.time_start = timeit.default_timer()
        self.trivial_nodes_skipped=0
        self.infeasible_nodes=0
        self.bypasses=0
        self.search_calls=0
        self.env.reset_subroutine_logs()
    def end_statistics(self,timeout,no_solution):
        closed_list_len=len(self.closed_list)
        open_list_len=len(self.open_list)
        elapsed=timeit.default_timer()-self.time_start
        stats= {"elapsed":elapsed,
                "high_level_expansions":closed_list_len,"search_calls":self.search_calls,
                "trivial_nodes_skipped":deepcopy(self.trivial_nodes_skipped),"infeasible_nodes":deepcopy(self.infeasible_nodes),"timeout":timeout,"proved infeasible":no_solution,
                "bypasses":deepcopy(self.bypasses),"subroutine_calls":deepcopy(self.env.subroutine_calls),"subroutine_times":deepcopy(self.env.subroutine_times)}
        return stats
    def search(self,time_limit=np.inf):
        try:
            return self.search_work(time_limit)
        except TimeoutError:
            stats=self.end_statistics(True,False)
            return {},stats
    def search_work(self,time_limit=np.inf):
        self.init_statistics()
        start = HighLevelNode()
        # TODO: Initialize it in a better way
        start.constraint_dict = {}
        for agent in self.env.agent_dict.keys():
            start.constraint_dict[agent] = Constraints()
        self.env.constraint_dict=start.constraint_dict
        self.search_calls+=1
        self.env.makespan_lb=0#reset environment's makespan lb just in case
        start.solution = self.env.compute_solution(self.time_start,time_limit)
        self.open_list= [start]
        self.closed_list = list()
        if not start.solution:
            self.infeasible_nodes+=1
            stats=self.end_statistics(False,True)
            return {},stats
        start.cost = self.env.compute_solution_cost(start.solution)

        #rerun search with the known lb on makespan to get properly (makespan,sum of costs) optimal single agent paths
        self.env.makespan_lb=start.cost[0]
        self.search_calls+=1
        start.solution = self.env.compute_solution(self.time_start,time_limit)

        start.conflicts = self.env.get_all_conflicts(start.solution)
        plan={}
        elapsed=timeit.default_timer()-self.time_start
        timeout=elapsed>=time_limit
        no_solution=len(self.open_list)==0
        while not no_solution and not timeout:
            P = min(self.open_list)
            self.open_list.remove(P)
            self.closed_list.append(P)
            self.env.constraint_dict = P.constraint_dict
            conflict=self.env.choose_conflict_by_type(P.conflicts)
            print(f"{P.cost} with {len(P.conflicts)} conflicts: {conflict}")
            if conflict is None:
                print("solution found")
                plan=P.solution
                break 
            constraint_options=conflict.get_constraints()
            child_nodes=[]
            bypass=False
            self.env.makespan_lb=P.cost[0]#update the makespan lb
            for option in constraint_options:
                if all(option[agent].satisfied(P.solution[agent]) for agent in option):
                    #if solution doesn't violate this constraint, no point in adding a node for it
                    self.trivial_nodes_skipped+=1
                    continue
                new_node = deepcopy(P)
                for agent in option:
                    new_node.constraint_dict[agent].add_constraint(option[agent])

                self.env.constraint_dict = new_node.constraint_dict
                self.search_calls+=1
                new_node.solution = self.env.compute_solution(self.time_start,time_limit,P.solution)
                if not new_node.solution:
                    self.infeasible_nodes+=1
                    continue
                new_node.cost = self.env.compute_solution_cost(new_node.solution)
                new_node.conflicts = self.env.get_all_conflicts(new_node.solution)
                print(f"child cost {new_node.cost} with {len(new_node.conflicts)} conflicts")
                if new_node.cost<=P.cost and len(new_node.conflicts)<len(P.conflicts):
                    #found a bypass
                    self.open_list.append(new_node)
                    bypass=True
                    self.bypasses+=1
                    break
                child_nodes.append(new_node)
            if not bypass:
                self.open_list.extend(child_nodes)
            elapsed=timeit.default_timer()-self.time_start
            timeout=elapsed>=time_limit
            no_solution=len(self.open_list)==0
        stats=self.end_statistics(timeout,no_solution)
        return plan,stats

    def generate_plan(self, solution:Dict[str,List[Edge]]):
        plan = {}
        for agent, path in solution.items():
            path_dict_list = [{'t':edge.state1.time, 'x':edge.state1.location.x, 'y':edge.state1.location.y,"z":edge.state1.location.z, 'turn':edge.state1.location.turn} for edge in path]
            final_state=path[-1].exit_state()
            path_dict_list.append({"t":final_state.time,"x":final_state.location.x,"y":final_state.location.y,"z":final_state.location.z,"turn":final_state.location.turn})
            plan[agent] = path_dict_list
        return plan

def agent_data_to_agent_dict(agent_data:List[Dict]):
    agent_dict=dict()
    for agent_specification in agent_data:
        name=agent_specification['name']
        start_coord=agent_specification['start']
        start_time=agent_specification['start_time']
        goal_coord=agent_specification['goal']
        start_block_type=agent_specification["start_block_type"]
        start_block_data=agent_specification["start_block_data"]
        start_block=block_utils.from_specification(start_block_type,start_block_data)
        start=State(start_time,Location(*start_coord),start_block)

        goal_type=agent_specification["goal_type"]
        goal_location=Location(*goal_coord)
        if goal_type=="MOVE":
            goal_action=MoveAction(start_block,goal_location)
        elif goal_type=="PLACE":
            goal_action=PlacementAction(start_block,goal_location)
        elif goal_type=="PICK":
            goal_block_type=agent_specification["goal_block_type"]
            goal_block_data=agent_specification["goal_block_data"]
            goal_block=block_utils.from_specification(goal_block_type,goal_block_data)
            goal_action=PickupAction(goal_block,goal_location)
    
        agent_dict[name]=Agent(name,start,goal_action)
    return agent_dict



def main():
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("param", help="input file containing map and obstacles")
    parser.add_argument("output", help="output file with the schedule")
    args = parser.parse_args()

    # Read from input file
    with open(args.param, 'r') as param_file:
        try:
            param = yaml.load(param_file, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            print(exc)

    dimension = param["map"]["dimensions"]
    obstacles = None
    agent_data = param['agents']

    env = Environment(dimension, agent_data_to_agent_dict(agent_data), obstacles)

    # Searching
    cbs = CBS(env)
    solution,stats = cbs.search()
    if not solution:
        print(" Solution not found" )
        # # Write to output file that solution not found
        # output = dict()
        # output["schedule"] = "Solution not found"
        # output["computation_time"] = time.time() - start_time
        # with open(args.output, 'w') as output_yaml:
        #     yaml.safe_dump(output, output_yaml)
        return
    print("--- %s seconds ---" % (time.time() - start_time))
    # Write to output file
    output = dict()
    output["schedule"] = solution
    output["cost"] = env.compute_solution_cost(solution)
    output["computation_time"] = time.time() - start_time 
    output["makespan"] = max([len(path) for path in solution.values()])
    output.update(stats)

    with open(args.output, 'w') as output_yaml:
        yaml.safe_dump(output, output_yaml)

if __name__ == "__main__":
    main() 