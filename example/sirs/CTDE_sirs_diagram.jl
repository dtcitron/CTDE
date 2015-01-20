# Call this script from the command line 
# with an HDF5 config file and a group id as arguments
# julia -p 2 CTDE_sirs_diagram.jl config.hdf5 g1

if length(ARGS) != 2
    println("Need 2 arguments: configuration file name, group id")
    exit(1)
end

# read in the configuration file name
cfilename = string(ARGS[1]);
group_id = string(ARGS[2]);

using HDF5 #, JLD

f = h5open(cfilename, "r");

if !in(group_id, names(f))
    println("Incorrect group id, try one of these")
    println(names(f))
    exit(1)
end

group = f[group_id];

if length(names(group)) != 9
    println("Insufficient information to run simulation")
    println("Need: Edgedata, graphsize, otimes, R0s, alphas, mean k, nruns, filename, and description")
    exit(1)
end

tic()

edata = read(group["edgedata"])';
otimes = read(group["otimes"]);
r0s = read(group["R0s"]);
alphas = read(group["alphas"]);
k = read(group["kmean"]);
nruns = read(group["nruns"]);
filename = read(group["filename"]);
desc = read(group["description"]);

# For debugging
println("Shape of edge data array: ", size(edata))
println("Observation times: ", otimes)
println("r_0 array: ", r0s)
println("alpha array: ", alphas)
println("Mean degree: ", k)
println("Number of runs: ", nruns)
println("Output filename: ", filename)
println("Graph description: ", desc)

# Now what I need to do is make sure that I can actually
# correctly call routines from sirs_graph
@everywhere require("sirs.jl")
@everywhere include("sirs_graph.jl")
#println("included sirs_graph.jl")

seed = 0;

g = contact_graph(edata);

# need to make sure that the observation times has right data type
otimes = Float64[t for t in otimes];

toc()

# time everything
tic()

sirs_diagram(nruns, g, alphas, r0s, otimes, k, seed, filename)

toc()