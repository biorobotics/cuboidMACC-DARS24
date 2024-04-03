import cbs
import block_utils
from assembly import problem_node,search_algo
from parallelization import interface

def three_block_stair():
    cube=block_utils.CuboidalBlock(1)
    locs=[cbs.Location(2,2,0,0),cbs.Location(2,1,0,0),cbs.Location(2,2,1,0)]
    starts=[cbs.RobotOutsideWorld(0,cube) for _ in range(len(locs))]
    agents={i:cbs.Agent(i,starts[i],cbs.PlacementAction(cube,locs[i])) for i in range(len(locs))}
    environment=cbs.Environment((6,6,2),agents)  
    return environment

def bridge_of_three():
    start=problem_node.Assembly_Node([])
    goal=problem_node.Assembly_Node.from_npy("src/data/custom/bridge_7x7x4.npy")
    plan=search_algo.a_star_search(start,goal)
    environment=interface.make_single_environment_from_plan(start,plan)
    return environment