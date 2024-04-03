import sys
import numpy as np
sys.path.append('/Users/shambhavisingh/Documents/GitHub/macc_general_mamba/src/assembly')
from problem_node import Assembly_Node
from search_algo import a_star_search
from run_parallel import create_time_windows
# from assembly.run_assembly import main as run_assembly
from config import get_parser

def main():
    args = get_parser().parse_args()
    input_filename = args.input
    output_filename = args.output

    # load from pickle file input and output
    # input_data = np.load(input_filename, allow_pickle=True)
    output_data = np.load(output_filename, allow_pickle=True).tolist()
    # convert to list

    start = Assembly_Node([])
    goal = Assembly_Node(output_data)
    plan = a_star_search(start, goal)
    # print(solution[-1].state)
    parallel_solution = create_time_windows(plan)

if __name__ == '__main__':
    main()