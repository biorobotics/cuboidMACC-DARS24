import numpy as np
from assembly import block_utils
from path import cbs
from assembly.end_to_end_tests.test_a_star import generate_random_structure

def place_blocks_on_ground(lengths,x_dim,y_dim,num_blocks,rng:np.random.Generator):
    blocks_to_place=generate_random_structure(lengths,x_dim,y_dim,1,num_blocks,rng)
    occupancy_grid=block_utils.occupancy_grid(blocks_to_place,x_dim,y_dim,1)
    free_space=list()
    for i in range(x_dim):
        for j in range(y_dim):
            if not occupancy_grid[i,j]:
                free_space.append((i,j))
    agents=dict()
    for i,block_tuple in enumerate(blocks_to_place):
        _,length,rotation,x,y,z=block_tuple
        turn=[0,90][rotation]
        block=block_utils.CuboidalBlock(length)
        action=cbs.PlacementAction(block,cbs.Location(x,y,z,turn))

        options=cbs.copy(free_space)
        location=None
        while len(options)>0:
            option_id=rng.choice(len(options))
            x,y=options[option_id]
            
            if not any(occupancy_grid[*cell] for cell in block.cells_occupied(x,y,0,0) if block_utils.in_bounds(*cell,*occupancy_grid.shape)):
                location=cbs.Location(x,y,0,0)
                break
            elif not any(occupancy_grid[*cell] for cell in block.cells_occupied(x,y,0,1)  if block_utils.in_bounds(*cell,*occupancy_grid.shape)):
                location=cbs.Location(x,y,0,90)
                break
            else:
                options.remove((x,y))
        
        free_space.remove((x,y))
        start=cbs.State(0,location,block)
        
        agent=cbs.Agent(f"Agent {i}",start,action)
        agents[agent.name]=agent
        for cell in block.cells_occupied(x,y,0,location.vertical()):
            if block_utils.in_bounds(*cell,*occupancy_grid.shape):
                occupancy_grid[*cell]=True
    env=cbs.Environment((x_dim,y_dim,1),agents,None)
    return env

def place_blocks_on_ground_from_outside_world(lengths,x_dim,y_dim,num_blocks,rng:np.random.Generator):
    blocks_to_place=generate_random_structure(lengths,x_dim,y_dim,1,num_blocks,rng)
    occupancy_grid=block_utils.occupancy_grid(blocks_to_place,x_dim,y_dim,1)
    agents=dict()
    for i,block_tuple in enumerate(blocks_to_place):
        _,length,rotation,x,y,z=block_tuple
        turn=[0,90][rotation]
        block=block_utils.CuboidalBlock(length)
        action=cbs.PlacementAction(block,cbs.Location(x,y,z,turn))

        start=cbs.RobotOutsideWorld(0,block)
        
        agent=cbs.Agent(f"Agent {i}",start,action)
        agents[agent.name]=agent
    env=cbs.Environment((x_dim,y_dim,1),agents,None)
    return env

def run_case(environment,time_limit):
    solver=cbs.CBS(environment)
    solution,stats=solver.search(time_limit)
    passed,info=environment.verify_solution(solution)
    return passed,solution,stats,info

def test_random_from_empty(lengths,x_dim,y_dim,num_blocks,seed,num_cases,time_limit):
    rng=np.random.default_rng(seed)
    passes={}
    fails={}
    for case_num in range(num_cases):
        env=place_blocks_on_ground(lengths,x_dim,y_dim,num_blocks,rng)
        ok,solution,stats,info=run_case(env,time_limit)
        if not ok:
            print(f"{case_num}: Failed")
            fails[case_num]=(env,solution,stats,info)
        else:
            print(f"{case_num}: Passed")
            passes[case_num]=(env,solution,stats)
    return len(fails)==0,passes,fails

def test_random_from_empty_case_i(lengths,x_dim,y_dim,num_blocks,seed,case_i,time_limit):
    rng=np.random.default_rng(seed)
    passes={}
    fails={}
    for case_num in range(case_i+1):
        env=place_blocks_on_ground(lengths,x_dim,y_dim,num_blocks,rng)
        if case_num==case_i:
            ok,solution,stats,info=run_case(env,time_limit)
            if not ok:
               print(f"{case_num}: Failed")
               fails[case_num]=(env,solution,stats,info)
            else:
                print(f"{case_num}: Passed")
                passes[case_num]=(env,solution,stats)
    return len(fails)==0,passes,fails

def test_random_starting_outside(lengths,x_dim,y_dim,num_blocks,seed,num_cases,time_limit):
    rng=np.random.default_rng(seed)
    passes={}
    fails={}
    for case_num in range(num_cases):
        env=place_blocks_on_ground_from_outside_world(lengths,x_dim,y_dim,num_blocks,rng)
        ok,solution,stats,info=run_case(env,time_limit)
        if not ok:
            print(f"{case_num}: Failed")
            fails[case_num]=(env,solution,stats,info)
        else:
            print(f"{case_num}: Passed")
            passes[case_num]=(env,solution,stats)
    return len(fails)==0,passes,fails