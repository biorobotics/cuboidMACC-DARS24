import assembly.block_utils

def test_incremental(start_grid,action_sequence):
    """
    check that incremental and from grid constructions of Reachability produce the same thing

    Parameters: start_grid : (x_dim, y_dim, z_dim) binary np array
                    occupancy grid to initialize the world at
                action_sequence : List[(bool,(int,int,int))]
                    first element True means add, False means remove. Second element is block coordinate
                    If the action is not feasible at the location, it is skipped.
                action_type : List[bool]
                    action_type[i]==True means add a block, action_type[i]==False means remove.
                    If the action is not feasible at block_sequence[i], it is skipped.
    Returns:    passed : bool
                    True if the incremental and from grid Reachability were exactly the same after every action taken. If False, we terminated as soon as they deviated.
                actual_action_sequence : List[(bool,(int,int,int))]
                    the actions we didn't skip, in the order they were executed
                occu_grid : (x_dim, y_dim, z_dim) binary np array
                    occupancy grid after actual_action_sequence
                correct : block_utils.Reachability
                    Reachability instance made from the occupancy grid at termination
                for_incremental : block_utils.Reachability
                    If passed, the Reachability instance made incrementally. If not passed, the Reachability instance made from the occupancy grid before divergence (1 action short of correct).
    """
    current_grid=start_grid.copy()
    from_grid=block_utils.Reachability.from_occupancy_grid(current_grid)
    actual_action_sequence=[]
    for add,block in action_sequence:
        if add and current_grid[*block]:
            print(f"Block already placed at {block}; skipping")
            continue
        if not add and not current_grid[*block]:
            print(f"No block placed at {block}; skipping")
            continue
        if add and not block_utils.can_stand_at(*block,current_grid):
            #input block sequence tries to place a block somewhere impossible
            print(f"No support for block at {block}; skipping")
            continue
        if add and block not in from_grid.reachable_cells:
            print(f"No path to place {block}; skipping")
            continue
        if not add and len(from_grid.reachable_cells&block_utils.action_locations(1,1,block))==0:
            print(f"No path to remove {block}; skipping")
            continue
        actual_action_sequence.append((add,block))
        current_grid[*block]=add
        if add:
            from_incremental=block_utils.update_reachable_states_add_connected_blocks(from_grid,current_grid,{block})
        else:
            from_incremental=block_utils.update_reachable_states_remove_connected_blocks(from_grid,current_grid,{block})
        new_from_grid=block_utils.Reachability.from_occupancy_grid(current_grid)
        if new_from_grid!=from_incremental:
            msg=""
            action_names=[" Remove "," Add "]
            for action,loc in actual_action_sequence:
                msg+=action_names[action]+str(loc)
            print(f"Incremental and direct don't match after sequence:{msg}")
            return False, actual_action_sequence,current_grid,new_from_grid,from_grid
        from_grid=new_from_grid
    return True, actual_action_sequence,current_grid,from_grid,from_incremental

case1=(block_utils.occupancy_grid([]),
       [(1,(4,4,0)),(1,(4,5,0)),(1,(4,5,1)),(0,(4,4,0)),(1,(5,5,0)),(1,(6,5,0)),(1,(5,5,1)),(1,(4,5,2))])

case2=(block_utils.occupancy_grid([]),
       [(1,(4,4,0)),(1,(4,5,0)),(1,(4,5,1)),(0,(4,4,0)),(1,(5,5,0)),(1,(6,5,0)),(1,(5,5,1)),(1,(4,5,2)),
        (0,(4,5,2)),(0,(5,5,1)),(0,(6,5,0)),(0,(5,5,0)),(1,(4,4,0)),(0,(4,5,1)),(0,(4,5,0)),(0,(4,4,0))])

defaultcases=[case1,case2]

def main(cases):
    passes=dict()
    fails=dict()
    for i,case in enumerate(cases):
        out=test_incremental(*case)
        if out[0]:
            print(f"Passed Case {i}")
            passes[i]=out
        else:
            print(f"Failed Case {i}")
            fails[i]=out
    print(f"Passed {len(passes)} of {len(cases)}")
    print(f"Failed {len(fails)} of {len(cases)}")
    return passes,fails
if __name__=="__main__":
    main(defaultcases)