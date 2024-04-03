import numpy as np
import random
import sys
# sys.path.append('/home/shambhavi/macc_general_mamba/src/')
from src.assembly.block_utils import occupancy_grid, has_support_below
def generate_random_structure(x_dim=7,y_dim=7,z_dim=4,num_blocks=15):
    state_ = []
    grid = occupancy_grid(state_,x_dim,y_dim,z_dim ,check_assert=True)
    for i in range(num_blocks):
        placed = False
        print("Placed", i, "blocks")
        j = 0
        while(not placed):
            j += 1
            print("Trying to place block", i+1, "for the", j, "time")
            new_state = state_.copy()
            length = random.choice([1, 3, 5])
            rotation = random.choice([0, 1])
            z = random.choice([0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
            location = [random.choice(range(x_dim)), random.choice(range(y_dim)), z]
            block = [1, length, rotation, location[0], location[1], location[2]]            
            try:
                assert (z==0 or has_support_below(block, grid))
                new_state.append(block)
                grid = occupancy_grid(new_state,x_dim,y_dim,z_dim, check_assert=True)
                placed = True
            except:
                continue
        state_ = new_state.copy()
    return state_

def main():
    for index in range(10):
        state = generate_random_structure()
        # save the state to a file pickle
        np.save('/home/shambhavi/macc_general_mamba/src/data/struct_10x10x4_n'+str(index)+'.npy', state)


if __name__ == "__main__":
    main()
