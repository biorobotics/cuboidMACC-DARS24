Code and test set for Singh et al., "Hierarchical Planner for Heterogeneous
Multi-Robot Construction," submitted to DARS2024.

# About
Recent work in planning for cooperative mobile robots to assemble a structure much larger than themselves out of only cubic ($1 \times 1 \times 1$) blocks has been termed "the multi-agent collective construction (MACC) problem." In this work, we create a new planner that solves the MACC problem with \emph{cuboid} ($n \times 1 \times 1$)  blocks, which we call the cuboid-MACC problem. In doing so, we can now build structures with hollow features. The cuboid blocks introduce an additional challenge: they impose restrictions on the maneuverability of the robots that carry the blocks. To address cuboid-MACC, we present a novel hierarchical planning approach that handles the maneuverability constraints imposed by cuboid blocks. First, we use $A^*$ to determine a sequence of abstract actions (block placements and removals) to build the structure without planning specific paths for the robots. We nevertheless ensure that such paths will exist through the concept of a reachable abstract action. Next, we identify dependencies between the chosen abstract actions by checking for conflicts between single-agent paths to execute the actions and store the dependencies as an acyclic Action Dependency Graph. Finally, we iterate on the specific paths taken by robots using a low-level multi-agent pathfinding algorithm by suitable modifications to conflict-based search (CBS). We demonstrate our planner on a set of randomly generated structures built with three types of cuboidal blocks. 

# Installation
## Dependencies
A conda environment.yml is provided. Requires numpy, scipy, numba, networkx, pyyaml. Saving and loading data files require dill. Visualizations require matplotlib, meshcat, trimesh, and plotly.

The C++ extension modules use pybind11, which you will need to install: https://pybind11.readthedocs.io/en/stable/
## Compilation
The high level planner relies on a compiled extension module to compute reachable abstract actions. The source code is in src/assembly/cpp. A Makefile is provided. 

You will need to update MACC_HOME with the correct path to the folder of this repository.
You will need to update PYBIND_INCLUDE with the correct path to the pybind11 include folder.
You will need to update PYTHON_INCLUDE with the correct path to the python include folder for your choice of python executable.
You MAY need to update PYTHON_EXT_SUFFIX depending on your python version and platform.

# Contents
## Code
The code is organized into 5 submodules: assembly, parallelization, path, utility, and visualize. data contains sample test structures.

### assembly
assembly contains code related to the High-Level (Abstract Action) planner. It uses A* to find the shortest sequence of block placements and removals to build the structure. 
Note the cpp subfolder, containing the c++ implementation of the incremental reachability calculations.

### parallelization
parallelization contains code related to the interface between the high and low-level planners (that is, the computation of the action dependency graph, by interface.py's create_time_windows function). 
The repeated calls to task allocation and multi agent path finding are also handled inside interface.py, by the compute_low_level_plan_with_assignment function

It also contains code to run the full pipeline in run_structures.py.

### path
path contains code for the Low-Level (Multi-Agent Path Finding) planning.

### utility
contains support code for the rest of the package

### visualization
contains code to use Meshcat to make 3D visualizations and animations of construction. See visualize3D's visualize_meshcat, visualize_world_meshcat, animate_high_level_plan, and meshcat_frames_to_video.

### data
contains structure designs stored as npy files. They can be loaded by assembly.problem_node.Assembly_Node.from_npy (be sure to specify the correct world size).

## Test Data
The 185 of the test structures from the paper are specified as .npy files in src/data/random_10x10x4_15blocks. The last 15 are the struct_10x10x4_n{i}.npy files in src/data/random.

The structures named struct_10x10x4_n{i}.npy in a folder at path/to/folder (e.g. "cuboidMACC-DARS24/src/data/random") can be processed programmatically by:

```python
  from parallelization import  run_structures
  x_dim=10
  y_dim=10
  z_dim=5
  structure_prefix="struct_10x10x4_n"
  num_structs=15#processes structures named f"struct_10x10x4_n{i}.npy" for i in range(num_structs)
  time_limit=1000#seconds for a single structure after which we report timeout and move to the next one
  run_structures.run_and_save("/path/to/folder","path/to/output_folder", "struct_10x10x4_n",x_dim,y_dim,z_dim,num_structs,time_limit)
```
