from parallelization import run_structures
x_dim=10
y_dim=10
z_dim=5
structure_prefix="struct_10x10x4_n"
num_structs=15
time_limit=1000
run_structures.run_and_save("cuboidMACC-DARS24/src/data/random","cuboidMACC-DARS24/src/data/output", structure_prefix,x_dim,y_dim,z_dim,num_structs,time_limit)

