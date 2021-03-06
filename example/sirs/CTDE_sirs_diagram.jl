# Call this script from the command line 
# with an HDF5 config file and a group id as arguments
# julia -p 2 CTDE_sirs_diagram.jl config.hdf5 g1 output.hdf5

if length(ARGS) != 3
    println("Need 3 arguments: configuration file name, group id, output file name")
    exit(1)
end

# read in the configuration file name
cfilename = string(ARGS[1]);
group_id = string(ARGS[2]);
output_filename = string(ARGS[3]);

using HDF5 #, JLD

f = h5open(cfilename, "r");

if !in(group_id, names(f))
    println("Incorrect group id, try one of these")
    println(names(f))
    exit(1)
end

group = f[group_id];

if length(names(group)) < 8
    println("Insufficient information to run simulation")
    println("Need: Edgedata, otimes, R0s, alphas, mean k, nruns, filename, and description")
    exit(1)
end

tic()

edata = read(group["edgedata"])';
otimes = read(group["otimes"]);
r0s = read(group["r0s"]);
alphas = read(group["alphas"]);
k = read(group["kmean"]);
nruns = read(group["nruns"]);
# Ran into problems reading strings from HDF5 in Julia
#filename = read(group["filename"]);
filename = output_filename
desc = read(group["description"]);

# set the simulation random seed
if "seed" in names(group)
    seed = read(group["seed"]);
else
    seed = 0;
end

# For debugging
println("Shape of edge data array: ", size(edata))
#println("Observation times: ", otimes)
#println("r_0 array: ", r0s)
#println("alpha array: ", alphas)
#println("Mean degree: ", k)
#println("Number of runs: ", nruns)
println("Output filename: ", filename)
#println("Graph description: ", desc)
println("Simulation seed: ", seed)

# Now what I need to do is make sure that I can actually
# correctly call routines from sirs_graph
@everywhere require("sirs.jl")
@everywhere include("sirs_graph.jl")
#println("included sirs_graph.jl")

g = contact_graph(edata);

# need to make sure that the observation times has right data type
otimes = Float64[t for t in otimes];

toc()

# time everything

@time sirs_diagram(nruns, g, alphas, r0s, otimes, k, seed, filename)
