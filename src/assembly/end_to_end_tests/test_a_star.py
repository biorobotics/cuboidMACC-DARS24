import numpy as np
import block_utils
import problem_node
import search_algo

def generate_random_structure(lengths,x_dim,y_dim,z_dim,num_blocks,rng:np.random.Generator):
    #when sampling heights, all non-zero heights are equally likely
    #and are r times more likely than height 0
    #thus, the probability for one of the n above 0 heights should be r/(r*n+1)
    ratio_0_to_nonzero=3
    p_above_0=ratio_0_to_nonzero/(ratio_0_to_nonzero*(z_dim-1)+1)
    height_preference=np.zeros(z_dim)
    height_preference[1:]=p_above_0
    height_preference[0]=1-np.sum(height_preference[1:])
    state_ = []
    for i in range(num_blocks):
        grid=block_utils.occupancy_grid(state_,x_dim,y_dim,z_dim)
        placed = False
        j = 0
        new_state = state_.copy()
        while(not placed):
            j += 1
            length = rng.choice(lengths)
            rotation = rng.choice([0, 1])
            z = rng.choice(range(z_dim),p=height_preference)
            location = [rng.choice(range(x_dim)), rng.choice(range(y_dim)), z]
            block = (1, int(length), int(rotation), int(location[0]), int(location[1]), int(location[2]))
            if z==0 or block_utils.has_support_below(block,grid):
                affected_cells=list(block_utils.cells_occupied(length,rotation,location[0],location[1],location[2]))
                if all(block_utils.in_bounds(*c,x_dim,y_dim,z_dim) for c in affected_cells) and not any(grid[c] for c in affected_cells):
                    #block is supported and doesn't overlap with any existing blocks
                    new_state.append(block)
                    placed=True
        state_ = new_state.copy()
    return state_

def run_case(start_state,goal_state,x_dim,y_dim,z_dim):
    start_node=problem_node.Assembly_Node(start_state,x_dim=x_dim,y_dim=y_dim,z_dim=z_dim)
    goal_node=problem_node.Assembly_Node(goal_state,x_dim=x_dim,y_dim=y_dim,z_dim=z_dim)
    solution=search_algo.a_star_search(start_node,goal_node)
    action_sequence=solution[-1].state
    is_feasible,reachable_occu_grid,index,message=block_utils.check_single_agent_plan_is_feasible(action_sequence,block_utils.occupancy_grid(start_state,x_dim,y_dim,z_dim))
    if not is_feasible:
        return is_feasible,action_sequence,reachable_occu_grid,index,message
    achieves_goal,achieved_blocks,index,message= block_utils.check_plan_achieves_goal(action_sequence,{tuple(action[1:]) for action in start_state},goal_state)
    return achieves_goal,action_sequence,achieved_blocks,index,message

def test_random_from_empty(lengths,x_dim,y_dim,z_dim,num_blocks,seed,num_cases):
    start_state=[]
    rng=np.random.default_rng(seed)
    passes={}
    fails={}
    for case_num in range(num_cases):
        goal_state=generate_random_structure(lengths,x_dim,y_dim,z_dim,num_blocks,rng)
        ok,solution,status,index,message=run_case(start_state,goal_state,x_dim,y_dim,z_dim+1)#z_dim+1 to ensure robots can carry blocks at the top of the world!
        if not ok:
            print(f"{case_num}: Failed")
            fails[case_num]=(start_state,goal_state,solution,status,index,message)
        else:
            print(f"{case_num}: Passed")
            passes[case_num]=(start_state,goal_state,solution)
    return len(fails)==0,passes,fails