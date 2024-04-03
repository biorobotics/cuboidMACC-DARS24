import heapq
from itertools import count
import timeit
import block_utils
from utility.exceptions import *

TIMEOUT="TIMEOUT"
HIGH_LEVEL_FOUND="FOUND"
NO_SOLUTION="INFEASIBLE"

def a_star_search(start, goal, time_limit=float('inf')):
    start_time = timeit.default_timer()
    tiebreaker = count()
    if goal.reachability.occupancy_grid is None:
        goal.reachability.occupancy_grid=block_utils.reachability.occupancy_grid(goal.state,goal.x_dim,goal.y_dim,goal.z_dim)
    start.get_scaffolding_options(goal)
    root = [start.get_heuristic(goal)+start.get_cost(),  # f
            -start.get_cost(),  # -g
            -len(start.scaffolding_present(goal)),#amount of scaffolding present
            next(tiebreaker), # tiebreaker
            None,  # parent
            start, # node
            False] # obsolete entry to skip
    open_list = [root]
    open_map = {start:root}
    closed_list = []
    branching=[]
    refound=0
    reopened=0
    generated=0

    while len(open_list) > 0:
        if timeit.default_timer()-start_time>=time_limit:
            print(f"Refound: {refound}")
            print(f"Reopened: {reopened}")
            print(f"Loops: {len(closed_list)}")
            print(f"Generated: {generated}")
            raise TimeoutError("High Level Search timed out")
        current_node = heapq.heappop(open_list)
        parent=current_node[4]
        node=current_node[5]
        ignore=current_node[6]
        if ignore:
            continue
        current_node[6]=True
        # print("Open list size: " + str(len(open_list)))
        # print("Closed list size: " + str(len(closed_list)))
        # print("Current node: " + str(current_node[-1].state) + "\n")
        # print("Current node h: " + str(current_node[0]-current_node[1]))

        if node.is_goal_node(goal):
            print("Found goal node")
            print("Time taken: " + str(timeit.default_timer() - start_time))
            plan = []
            while current_node[5] != start:
                plan.append(current_node[5])
                current_node = current_node[4]
            
            plan.reverse()
            return HIGH_LEVEL_FOUND,plan
        closed_list.append(current_node)
        children=node.get_child_nodes(goal)
        branching.append(len(children))
        for child in children:
            generated+=1
            child_node = [child.get_heuristic(goal)+child.get_cost(), 
                            -child.get_cost(), 
                            -len(child.scaffolding_present(goal)),
                            next(tiebreaker),
                            current_node, 
                            child,
                            False]
            if child in open_map:
                refound+=1
                existing=open_map[child]
                if child_node[0]<existing[0]:
                    reopened+=1
                    if existing[6]:
                        #replace it
                        existing[6]=True
                    open_map[child]=child_node
                    heapq.heappush(open_list, child_node)
            else:
                open_map[child]=child_node
                heapq.heappush(open_list, child_node)
    print(f"Refound:{refound}")
    print(f"Reopened:{reopened}")
    print(f"Loops: {len(closed_list)}")
    print(f"Generated: {generated}")
    return NO_SOLUTION,list()