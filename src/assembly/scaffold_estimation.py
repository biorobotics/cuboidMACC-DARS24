import numpy as np

from typing import Tuple,List
from numpy.typing import NDArray

from assembly import block_utils

def find_groups(world_size:Tuple[int,int,int],goal_occupancy_grid:NDArray,current_occupancy_grid:NDArray):
    """
    find the groups of unplaced goal blocks that are adjacent and at the same height as each other

    Parameters: world_size : Tuple[int,int,int]
                    x,y,z dimensions of the world
                goal_occupancy_grid : np boolean array, shape=world_size
                    which cells are supposed to be occupied in the goal state
                current_occupancy_grid : np boolean array, shape=world_size
                    which cells are occupied in the current state
    """
    groups=[]
    for level in range(world_size[2]):
        unfinished_goals=np.logical_and(goal_occupancy_grid[:,:,level],np.logical_not(current_occupancy_grid[:,:,level]))
        group_map=np.zeros((world_size[0],world_size[1]),dtype=np.int8)
        gid=1
        for x in range(world_size[0]):
            for y in range(world_size[1]):
                if unfinished_goals[x,y]:
                    this_group_map=np.zeros((world_size[0],world_size[1]),dtype=np.bool_)
                    connect_goals(world_size,unfinished_goals,x,y,this_group_map)
                    group_map+=gid*this_group_map#puts the integer group id in the cells belonging to that group
                    gid+=1
        groups.append(group_map)
    groups=np.stack(groups,axis=0)
    return groups

def connect_goals(world_size:Tuple[int,int,int],unfinished_goals:NDArray,query_x:int,query_y:int,this_group_map:NDArray):
    """
    connect neighboring goals into a group by marking this_group_map True and unfinished_goals to False for every cell in the group
    """
    if not unfinished_goals[query_x,query_y]:
        return
    unfinished_goals[query_x,query_y]=False
    this_group_map[query_x,query_y]=True
    for x,y in block_utils.horizontal_neighbors(query_x,query_y,world_size[0],world_size[1]):
        connect_goals(world_size,unfinished_goals,x,y,this_group_map)

def find_group_support(world_size:Tuple[int,int,int],groups:NDArray):
    group2support=[]
    sizes=[]
    for level in range(world_size[2]):
        group_map=groups[level]
        gids=np.unique(group_map)[1:]
        number_of_groups=len(gids)
        group_sizes=np.zeros(number_of_groups,dtype=np.int8)
        support_map=np.zeros((number_of_groups,world_size[2]-1,2,world_size[0],world_size[1]),dtype=np.bool_)#different groups, height, (dist <d, dist==d),x dim, y dim
        for i,g in enumerate(gids):
            g_map = group_map==g
            relevant_columns=np.transpose(np.nonzero(g_map))
            group_sizes[i]=len(relevant_columns)
            s_map=support_map[i]
            for support_height in range(level):
                for x,y in relevant_columns:
                    for support_cell in block_utils.generate_shadow_interior_at_dz((x,y,level),level-support_height,*world_size):
                        if not g_map[support_cell[0],support_cell[1]]:#skip cells directly below the group
                            s_map[support_height,0,support_cell[0],support_cell[1]]=True#interior of the shadow region; sometimes referred to as the "red" part of the support set
                for x,y in relevant_columns:
                    for support_cell in block_utils.generate_shadow_surface_at_distance((x,y,level),level-support_height,*world_size):
                        if not g_map[support_cell[0],support_cell[1]]:#skip cells directly below the group
                            if not s_map[support_height,0,support_cell[0],support_cell[1]]:#if not in the interior of some other block in the group's shadow region
                                s_map[support_height,1,support_cell[0],support_cell[1]]=True#surface of the shadow region; sometimes referred to as the "blue" part of the support set
        group2support.append(support_map)
        sizes.append(group_sizes)
    return group2support,sizes

def cast_scaffold_value(world_size:Tuple[int,int,int],goal_occupancy_grid:NDArray,group2support:List[NDArray],group_sizes:List[NDArray],current_occupancy_grid:NDArray):
    """
    For goal groups at each level, cast scaffold value to levels below, and record useful d-support

    Parameters: world_size : Tuple[int,int,int]
                    x,y,z dimensions of the world
                goal_occupancy_grid : np boolean array, shape=world_size
                    which cells are supposed to be occupied in the goal state
                group2support : List[(num groups at height, z_dim-1,2,x_dim,y_dim) boolean arrays]
                    entry i contains, for each group at height i, an indicator array showing [scaffolding options in interior of shadow, scaffolding options on surface of shadow]
                group_sizes : List[NDArray]
                    entry i contains, for each group at height i, the integer number of elements in that group
                current_occupancy_grid : np boolean array, shape=world_size
                    which cells are occupied in the current state
    Returns:    scaffold_v : a numpy array of shape (H, H - 1, w, w)
                useful_support: a list of useful entry, each of shape (num_groups, H - 1)
    """
    scaffold_val=np.zeros((world_size[2],world_size[2]-1,world_size[0],world_size[1]),dtype=np.int8)
    useful_support=[]
    for height in range(world_size[2]):
        val=scaffold_val[height]
        supports=group2support[height]
        sizes=group_sizes[height]
        number_of_groups=supports.shape[0]
        useful=np.zeros((number_of_groups,world_size[2]-1),dtype=np.int8)
        for i in range(number_of_groups):
            for support_height in range(height):
                support=supports[i,support_height]
                _,u,v=cal_support_val(support_height,support,current_occupancy_grid,goal_occupancy_grid,height,sizes[i])
                val[support_height]+=v
                useful[i,support_height]=u
        useful_support.append(useful)
    return scaffold_val,useful_support

def cal_support_val(support_height:int,support:NDArray,current_occupancy_grid:NDArray,goal_occupancy_grid:NDArray,goal_height:int,number_of_group_elements:int):
    """
    Calculate the scaffold value

    Parameters: support_height : int
                    height of the supports we are investigating
                support : (2,x_dim,y_dim) boolean array
                    [0,x,y] is True if xy in interior of shadow (red region), [1,x,y] is True if xy is on the surface of the shadow (blue region)
                current_occupancy_grid : np boolean array, shape=world_size
                    which cells are occupied in the current state
                goal_occupancy_grid : np boolean array, shape=world_size
                    which cells are supposed to be occupied in the goal state
                goal_height : int
                    the height of the goal block group being analyzed
                number_of_group_elements : int
                    the integer number of elements in the group being analyzed
    Returns:    valid : bool
                    at least one support block exists that does not have a goal block already placed directly above it
                u : bool
                    is a scaffold at this height useful
                v : (x_dim,y_dim) boolean array OR boolean scalar
                    value of each cell. 0 if not scaffold, level not useful, or a goal block placed directly abobe it; otherwise 1

    Scaffold usefulness rules:
        Red region: 0
        Blue region: 1 if not enough useful supports exist; 0 otherwise
    A support cell within the support set is useful if:
        1. It's an added scaffold block, or
        2. It's an un-added goal block, or
        3. It's an added goal block with no goal block added above
    Threshold for useful supports:
        1) 1 in blue region, or
        2) n in red region, n = goal lv - support lv - 1 * (# goal = 1)
    """
    no_goal_above=np.logical_not(goal_occupancy_grid[:,:,support_height+1]&current_occupancy_grid[:,:,support_height+1])#is there a goal placed above this cell?
    current_blocks=current_occupancy_grid[:,:,support_height]
    goal_blocks=goal_occupancy_grid[:,:,support_height]
    useful_blue=support[1] & (current_blocks | goal_blocks) & no_goal_above
    if np.any(useful_blue):
        return True,False,False
    useful_red=support[0] & (current_blocks | goal_blocks) & no_goal_above
    if np.sum(useful_red)>=max(1,goal_height-support_height-(number_of_group_elements==1)):
        return True,False,False
    valid = np.any((support[0]|support[1]) & no_goal_above)
    return valid,True,(support[1] & no_goal_above)

def cal_goal_val(world_size:Tuple[int,int,int],group2support:List[NDArray],useful_support:List[NDArray],scaffold_val:NDArray)->float:
    """
    compute an underestimate of the number of scaffolding blocks needed to be placed

    Parameters: world_size : Tuple[int,int,int]
                    x,y,z dimensions of the world
                group2support : List[(num groups at height, z_dim-1,2,x_dim,y_dim) boolean arrays]
                    entry i contains, for each group at height i, an indicator array showing [scaffolding options in interior of shadow, scaffolding options on surface of shadow]
                scaffold_v : a numpy array of shape (H, H - 1, w, w)
                useful_support: a list of useful entry, each of shape (num_groups, H - 1)
    Return:     h : int
                    underestimate of the number of scaffolding blocks needed to be placed
    """
    scaffold_v=np.sum(scaffold_val,axis=0)#scaffold_v[z,x,y] is the number of goal block groups at any height that benefit from scaffolding in the cell x,y,z
    goal_v=0
    for height in range(world_size[2]-1):
        supports_at_height=group2support[height].sum(axis=2)#supports_at_height[group_id,z,x,y] is True if a support block at x,y,z would be helpful to group_id
        num_useful_at_height=useful_support[height]#for each group at height, the number of useful blocks at a particular z
        num_groups=num_useful_at_height.shape[0]
        for i in range(num_groups):
            val=(scaffold_v*supports_at_height[i]).max(axis=(1,2))
            val=np.divide(num_useful_at_height[i],val,out=np.zeros_like(val,dtype=np.float32),where=val>0)
            goal_v+=val.sum()
    return goal_v

def estimate_missing_scaffolding(current_occupancy_grid:NDArray,goal_occupancy_grid:NDArray):
    groups=find_groups(current_occupancy_grid.shape,goal_occupancy_grid,current_occupancy_grid)
    group2support,sizes=find_group_support(current_occupancy_grid.shape,groups)
    scaffold_val,useful_support=cast_scaffold_value(current_occupancy_grid.shape,goal_occupancy_grid,group2support,sizes,current_occupancy_grid)
    return cal_goal_val(current_occupancy_grid.shape,group2support,useful_support,scaffold_val)