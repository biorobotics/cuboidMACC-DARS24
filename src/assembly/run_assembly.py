import sys
import pdb
from problem_node import Assembly_Node
from search_algo import a_star_search

def main(structure_array):
    start = Assembly_Node([])
    goal = Assembly_Node(structure_array)
    solution = a_star_search(start, goal)
    return solution[-1].state

if __name__ == "__main__":
    main([[1, 3, 0, 3, 2, 0], [1, 3, 0, 1, 4, 0]])
    # stats = cProfile.run('main(None)', 'restats')
    # p = pstats.Stats('restats')
    # p.sort_stats('tottime').print_stats(10)
