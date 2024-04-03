"""
Interfaces between the representations used by assembly and by path
"""
import copy
import timeit
import traceback

import numpy as np
import networkx as nx
from scipy import optimize
from typing import List,Dict,MutableSet,Tuple
from collections import defaultdict

from assembly import  block_utils,problem_node,search_algo
from path import cbs
from utility import timing,exceptions

from networkx.algorithms import bipartite

class InfeasibleMAPFProblem(Exception):
    pass

def action_from_assembly_node(node:problem_node.Assembly_Node):
    block=block_utils.CuboidalBlock(node.length)
    location=cbs.Location(*node.location,90*node.rotation)
    if node.action==-1:
        #removal
        action=cbs.PickupAction(block,location)
    elif node.action==1:
        #placement
        action=cbs.PlacementAction(block,location)
    return action

def action_sequence_from_assembly_node(node:problem_node.Assembly_Node):
    sequence=[]
    for action in node.state:
        action_kind,length,rotated,x,y,z=action
        block=block_utils.CuboidalBlock(length)
        location=cbs.Location(x,y,z,90*rotated)
        if action_kind==-1:
            sequence.append(cbs.PickupAction(block,location))
        elif action_kind==1:
            sequence.append(cbs.PlacementAction(block,location))
    return sequence

def make_agent_from_assembly_node(name:str,node:problem_node.Assembly_Node):
    action=action_from_assembly_node(node)
    if action.action_type==cbs.Action.PLACEMENT:
        block=action.block
    elif action.action_type==cbs.Action.REMOVAL:
        block=block_utils.NoBlock()
    start=cbs.RobotOutsideWorld(0,block)
    return cbs.Agent(name,start,action)

def world_state_from_assembly_node(node:problem_node.Assembly_Node):
    blocks=[]
    locations=[]
    for placement in node.state:
        if placement[0]==1:
            block=block_utils.CuboidalBlock(placement[1])
            location=cbs.Location(*placement[3:6],90*placement[2])
            blocks.append(block)
            locations.append(location)
    return cbs.WorldState.from_blocks(blocks,locations,(node.x_dim,node.y_dim,node.z_dim))

def create_time_windows(starting_world_state:cbs.WorldState,assembly_sequence:List[problem_node.Assembly_Node]):
    world_state=starting_world_state.copy()
    robot_locations = []
    action_locations = []
    time_windows = dict()
    for i in range(len(assembly_sequence)):
        agent=make_agent_from_assembly_node(str(i),assembly_sequence[i])
        action=agent.goal
        action_locations.append(assembly_sequence[i].state[-1][3:6])
        environment=cbs.Environment(world_state.occupancy.shape,{i:agent},world_state)
        solution=environment.compute_solution(0,np.inf)
        states=[edge.state1 for edge in solution[i]]
        action_state=None
        for edge in solution[i]:
            if edge.action==action:
                action_state=edge.state1
                break
        robot_locations.append(action_state)
        time_windows[i] = {
            'robot': robot_locations[i],
            'action': action,
            'path': states
        }
        agent.goal.apply_to_world(world_state)

    constraints = []
    for i in range(len(time_windows)):
        for j in range(0, len(time_windows)):
            if time_windows[i]['robot'] == time_windows[j]['robot'] \
            or len(time_windows[i]['action'].cells_changed() & time_windows[j]['action'].cells_changed())>0:
                #robots stand in same place or act on same location
                if i<j:
                    constraints.append((i, j))
                elif i>j:
                    constraints.append((j, i))
            elif time_windows[i]["action"].interferes_with_robot_at(time_windows[j]['robot']):
                #action affects where another robot wants to stand
                if i<j:
                    constraints.append((i, j))
                elif i>j:
                    constraints.append((j, i))
            elif time_windows[i]['robot'] in time_windows[j]['path'] \
            or any(time_windows[i]['action'].interferes_with_robot_at(state) for state in time_windows[j]['path']):
                #path of j interferes with action location of i or i's action interferes with path of j
                if i<j:
                    constraints.append((i, j))
                elif i>j:
                    constraints.append((j, i))
            elif time_windows[j]["action"].action_type==cbs.Action.PLACEMENT and time_windows[j]["action"].location2.z>0:
                #any blocks directly below a placement action need to get placed first
                cells_filled=time_windows[j]["action"].cells_changed()
                other_cells_changed=time_windows[i]["action"].cells_changed()
                for cell in cells_filled:
                    if (cell[0],cell[1],cell[2]-1) in other_cells_changed:
                        if i<j:
                            constraints.append((i,j))
            elif time_windows[j]["action"].action_type==cbs.Action.REMOVAL and time_windows[i]["action"].action_type==cbs.Action.REMOVAL:
                #any removals directly above a removal action need to happen first
                cells_emptied=time_windows[j]["action"].cells_changed()
                other_cells_emptied=time_windows[i]["action"].cells_changed()
                for cell in cells_emptied:
                    if (cell[0],cell[1],cell[2]+1) in other_cells_emptied:
                        if i<j:
                            constraints.append((i,j))
    
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

def make_environments_from_assembly_plan(start:problem_node.Assembly_Node,goal:problem_node.Assembly_Node,plan:List[problem_node.Assembly_Node]):
    world=world_state_from_assembly_node(start)
    dimension=world.occupancy.shape

    G=create_time_windows(world,plan)
    roots=[n for n,i in G.in_degree if i==0]
    envs=[]
    while len(roots)>0:
        agent_dict=dict()
        for node_index in roots:
            agent=make_agent_from_assembly_node(str(node_index),plan[node_index])
            agent_dict[agent.name]=agent
            G.remove_node(node_index)
        env=cbs.Environment(dimension,agent_dict,world)
        envs.append(env)
        world=env.achieved_world()
        roots=[n for n,i in G.in_degree if i==0]
    return envs

def compute_low_level_plan_from_assembly_plan(start:problem_node.Assembly_Node,goal:problem_node.Assembly_Node,plan:List[problem_node.Assembly_Node],time_limit=np.inf):
    world=world_state_from_assembly_node(start)
    dimension=world.occupancy.shape

    G=create_time_windows(world,plan)
    roots=[n for n,i in G.in_degree if i==0]
    output=[]
    all_agents=dict()
    full_solution=dict()
    round_start=0
    while len(roots)>0:
        round={"tasks":{i:plan[i] for i in roots}}
        agent_dict=dict()
        for node_index in roots:
            agent=make_agent_from_assembly_node(str(node_index),plan[node_index])
            agent.committed_action_sequence=make_waits_at(agent.start,0,round_start)
            agent.start.time=round_start
            agent_dict[agent.name]=agent
            G.remove_node(node_index)
        env=cbs.Environment(dimension,agent_dict,world)
        round["env"]=env

        solver=cbs.CBS(env)
        solution,stats=solver.search(time_limit)
        round["solution"]=solution
        round["stats"]=stats
        all_agents.update(agent_dict)
        full_solution.update(solution)
        round_start+=max(len(p) for p in solution.values())
        output.append(round)

        world=env.achieved_world()
        roots=[n for n,i in G.in_degree if i==0]
    full_env=cbs.Environment(dimension,all_agents,world_state_from_assembly_node(start))
    return output,full_env,full_solution

def make_waits_at(state:cbs.State,t0:int,delay:int):
    stay=cbs.MoveAction(state.block,state.location)
    wait=[]
    for i in range(delay):
        e=cbs.Edge(cbs.State(t0+i,state.location,state.block),stay)
        wait.append(e)
    return wait

def delay_solution(delay:int,solution:Dict[str,List[cbs.Edge]])->Dict[str,List[cbs.Edge]]:
    delayed=dict()
    for agent in solution:
        traj=[cbs.Edge(cbs.State(e.state1.time+delay,e.state1.location,e.state1.block),e.action) for e in solution[agent]]
        start=solution[agent][0].state1
        wait=make_waits_at(start,0,delay)
        full_traj=wait+traj
        delayed[agent]=full_traj
    return delayed

def make_single_environment_from_plan(start:problem_node.Assembly_Node,plan:List[problem_node.Assembly_Node]):
    world=world_state_from_assembly_node(start)
    dimension=world.occupancy.shape
    agents={str(i):make_agent_from_assembly_node(str(i),node) for i,node in enumerate(plan)}
    environment=cbs.Environment(dimension,agents,world)
    return environment

def reserve_path(path:List[cbs.Edge],world_size:Tuple[int,int,int])->List[cbs.CellSetConstraint]:
    """
    Create CellWindowConstraints that will keep the robot path feasible
    """
    constraints=list()
    for e in path:
        t=e.state1.time
        cells=copy.copy(e.involved_cells)
        if e.state1.location.z>0:
            cell_below=(e.state1.location.x,e.state1.location.y,e.state1.location.z-1)
            cells.add(cell_below)
        exit_state=e.exit_state()
        if exit_state.location.z>0:
            cell_below=(exit_state.location.x,exit_state.location.y,exit_state.location.z-1)
            cells.add(cell_below)
        in_bound_cells=[cell for cell in cells if block_utils.in_bounds(*cell,*world_size)]
        if len(in_bound_cells)>0:
            constraints.append(cbs.CellSetConstraint(t,frozenset(in_bound_cells)))
    return constraints

def find_allocation_with_specified_makespan(agents,tasks,cost_matrix,makespan_guess):
    G=nx.Graph()
    for agent in agents:
        G.add_node(agent,bipartite=0)
    for task in tasks:
        G.add_node(task,bipartite=1)
    for i,agent in enumerate(agents):
        for j,task in enumerate(tasks):
            if cost_matrix[i,j]<=makespan_guess:
                G.add_edge(agent,task)
    matching=bipartite.maximum_matching(G,tasks)
    if any(t not in matching for t in tasks):
        return False,matching
    else:
        return True, matching
    
def min_makespan_allocation(agents,tasks,cost_matrix):
    max_cost=np.max(cost_matrix[np.isfinite(cost_matrix)])

    
    feasible,solution=find_allocation_with_specified_makespan(agents,tasks,cost_matrix,max_cost)
    if not feasible:
        raise ValueError(f"Cannot assign {len(tasks)} tasks to {len(agents)} agents")
    smallest_success=max_cost
    largest_fail=-1
    makespan_guess=0
    while smallest_success-largest_fail>1:
        feasible,matching=find_allocation_with_specified_makespan(agents,tasks,cost_matrix,makespan_guess)
        if not feasible:
            largest_fail=max(makespan_guess,largest_fail)
        else:
            if makespan_guess<smallest_success:
                solution=matching
                smallest_success=makespan_guess
        new_guess=int((smallest_success+largest_fail)/2)
        if smallest_success-largest_fail>1 and new_guess==makespan_guess:
            raise ValueError("Makespan guess didn't change")
        else:
            makespan_guess=new_guess
    return {t:solution[t] for t in tasks},smallest_success

def compute_low_level_plan_with_assignment(start:problem_node.Assembly_Node,goal:problem_node.Assembly_Node,plan:List[problem_node.Assembly_Node],time_limit=np.inf,initial_walltime=0.0,phase_durations=None):
    if phase_durations is None:
        phase_durations=dict()
    tstart=timeit.default_timer()
    n_agents=len(plan)#TODO don't assume exactly 1 agent per high level action
    world=world_state_from_assembly_node(start)
    dimension=world.occupancy.shape

    G=create_time_windows(world,plan)
    walltime=timing.record_duration("time_windows",tstart,phase_durations)
    dependency_graph=G.copy()
    roots=[n for n,i in G.in_degree if i==0]
    output=[]
    all_agents=dict()
    task_paths=defaultdict(list)#maps agent name to List of Edges that the agent follows from its initial state to do every task it has been assigned
    home_paths=dict()#maps agent name to List of Edges that the agent can follow from its final task to outside the world
    available_state={str(i):make_agent_from_assembly_node(str(i),plan[i]).start for i in range(n_agents)}#State from which agent can be assigned to a new task
    agents_to_task=dict()#map an agent name to the task it has been assigned
    agent_task_sequence=defaultdict(list)
    task_to_agent=dict()#map a task to the agent it is assigned to
    retasked_agents_by_task=dict()#map a task index to the Agent object
    task_completion_times=dict()#map a task index to the time it was completed at
    task_allocation_walltime=0
    while len(roots)>0:
        #assemble a MAPF problem
        round={"tasks":{i:action_from_assembly_node(plan[i]) for i in roots}}
        agent_dict=dict()
        for node_index in roots:
            if node_index in retasked_agents_by_task:
                #an old agent from a previous round will handle this action from where it ended up after completing the previous job
                agent=retasked_agents_by_task[node_index]
            else:
                #a new agent will enter from OutsideWorld to handle this action
                agent=make_agent_from_assembly_node(str(node_index),plan[node_index])
                agents_to_task[agent.name]=node_index
            tlimit=-1
            for dependency in dependency_graph.predecessors(node_index):
                if dependency not in task_completion_times:
                    #incomplete dependency!
                    print("Trying to plan for a task with an incomplete dependency; expect MAPF to fail")
                else:
                    tlimit=max(tlimit,task_completion_times[dependency])
            if tlimit>=0:
                act_after_dependencies=cbs.BlockActionWindowConstraint(0,tlimit,agent.goal)
                agent.agent_constraints.add_e_constraint(act_after_dependencies)
                agent.earliest_goal_time=tlimit
            agent_dict[agent.name]=agent
            G.remove_node(node_index)
            task_to_agent[node_index]=agent
            agent_task_sequence[agent.name].append(node_index)
        reserved_path_constraints=cbs.Constraints()
        for agent in task_paths:
            if agent not in agent_dict:
                for con in reserve_path(task_paths[agent],world.occupancy.shape):
                    reserved_path_constraints.add_c_constraint(con)
                for con in reserve_path(home_paths[agent],world.occupancy.shape):
                    reserved_path_constraints.add_c_constraint(con)
        #handle scheduled edges that will be done by agents not participating in this MAPF instance
        scheduled_edges=[]
        for agent in task_paths:
            if agent not in agent_dict:
                scheduled_edges.extend(task_paths[agent])
                scheduled_edges.extend(home_paths[agent])
        env=cbs.Environment(dimension,agent_dict,world,reserved_path_constraints,scheduled_edges)
        round["env"]=env

        #solve the MAPF problem
        solver=cbs.CBS(env)
        elapsed=timeit.default_timer()-initial_walltime
        solution,stats=solver.search(time_limit-elapsed)
        round["solution"]=solution
        round["stats"]=stats
        if stats["proved infeasible"]:
            raise InfeasibleMAPFProblem("CBS reported an infeasible MAPF problem")
        elif stats["timeout"]:
            raise TimeoutError("CBS timed out")
        all_agents.update(agent_dict)

        #figure out when and where agents become available for new work
        for agent in agent_dict:
            task=agents_to_task[agent]
            goal_action_time=env.goal_completion_time(agent,solution)
            t0=solution[agent][0].state1.time
            break_point=goal_action_time-t0

            available_state[agent]=solution[agent][break_point].exit_state()
            available_state[agent].agent_mode=cbs.Agent.MODE_GOAL#reset to GOAL seeking mode
            task_completion_times[task]=available_state[agent].time
            #record the paths taken by agents in this round
            task_paths[agent].extend(solution[agent][:break_point+1])
            home_paths[agent]=solution[agent][break_point+1:]
        output.append(round)

        #assign new work to old agents where appropriate
        roots=[n for n,i in G.in_degree if i==0]
        n_tasks_this_round=len(roots)
        if n_tasks_this_round>0:
            start_task_allocation=timeit.default_timer()
            cost_matrix=np.full((n_agents,n_tasks_this_round),np.inf)
            for i,node_index in enumerate(roots):
                goal=action_from_assembly_node(plan[node_index])
                for agent_idx in range(n_agents):
                    start=available_state[str(agent_idx)]
                    if goal.required_block==start.block:
                        cost_matrix[agent_idx,i]=available_state[str(agent_idx)].time+cbs.act_and_leave_world_heuristic(available_state[str(agent_idx)],0,goal,dimension)[1]
            try:
                task_to_agent,predicted_makespan=min_makespan_allocation([str(idx) for idx in (range(n_agents))],list(range(n_tasks_this_round)),cost_matrix)
            except ValueError as e:
                raise exceptions.InfeasibleAllocationError() from e
            for task_idx,agent_name in task_to_agent.items():
                if agent_name in all_agents:
                    #this is a retasked agent
                    agent=all_agents[agent_name]
                    node_index=roots[task_idx]
                    agent.start=available_state[agent_name]
                    agent.goal=action_from_assembly_node(plan[node_index])
                    agent.committed_action_sequence=task_paths[agent.name]
                    agents_to_task[agent_name]=node_index
                    retasked_agents_by_task[roots[task_idx]]=agent
                else:
                    agents_to_task[agent_name]=node_index
            end_task_allocation=timeit.default_timer()
            task_allocation_walltime+=end_task_allocation-start_task_allocation
    full_solution=dict()
    full_agent_dict=dict()
    for agent in task_paths:
        full_solution[agent]=task_paths[agent]+home_paths[agent]
        full_agent_dict[agent]=make_agent_from_assembly_node(agent,plan[agent_task_sequence[agent][0]])
    full_env=cbs.Environment(dimension,full_agent_dict,world)
    walltime=timing.record_duration("MAPF",walltime,phase_durations)
    phase_durations["task_allocation"]=task_allocation_walltime
    return output,full_env,full_solution,phase_durations

def solve_assembly_problem_using_rounds(start:problem_node.Assembly_Node,goal:problem_node.Assembly_Node,time_limit=np.inf):
    t0=timeit.default_timer()
    result=dict()
    t=t0
    phase_durations=dict()
    result["phase_durations"]=phase_durations
    try:
        status,high_level_plan=search_algo.a_star_search(start,goal,time_limit)
    except search_algo.TimeoutError:
        status=search_algo.TIMEOUT
        high_level_plan=None
    t=timing.record_duration("high_level",t,phase_durations)
    result["high_level_plan_status"]=status
    result["high_level_plan"]=high_level_plan[-1].state
    if status==search_algo.HIGH_LEVEL_FOUND:
        try:
            output,full_env,full_solution,_=compute_low_level_plan_with_assignment(start,goal,high_level_plan,time_limit,t0,phase_durations)
            low_level_plan_status=cbs.LOW_LEVEL_FOUND
            msg="success"
        except TimeoutError as e:
            output=[]
            full_env=None
            full_solution=dict()
            low_level_plan_status=cbs.TIMEOUT
            msg=traceback.format_exc()
        except InfeasibleMAPFProblem as e:
            output=[]
            full_env=None
            full_solution=dict()
            low_level_plan_status=cbs.INFEASIBLE
            msg=traceback.format_exc()
        except exceptions.InfeasibleAllocationError as e:
            output=[]
            full_env=None
            full_solution=dict()
            low_level_plan_status="AllocationFailed"
            msg=traceback.format_exc()
        except Exception as e:
            output=[]
            full_env=None
            full_solution=dict()
            low_level_plan_status="OtherException"
            msg=traceback.format_exc()
        result["low_level_output"]=output
        result["full_environment"]=full_env
        result["full_solution"]=full_solution
        result["low_level_plan_status"]=low_level_plan_status
        result["low_level_msg"]=msg
    return result



    