#!/Applications/Julia-0.3.0.app/Contents/Resources/julia/bin/julia

module SIRSGRAPH

# This is a script that allows us to call the code from sirs_parallel.jl
# specifically, we will run fully mixed SIRS simulations
# this may be run in parallel, 
# calling julia -p n, where n is the number of threads

using HDF5
include("sirs.jl")
push!(LOAD_PATH, "../../src") # where SemiMarkov and other required modules sit

# variables glossary
# n : number of individuals
# nruns : number of times to run the simulation
# beta : infection rate (will divide by number of contacts in the code below)
# gamma : recovery rate
# rho : waning rate
# seed : seed for RNG, defaults to 34
# otimes : observation times, defaults to a particular sequence but can be added


function sirs_graph(nruns, g, beta, gamma, rho, seed = 34, ii = 1,
                    otimes = None, outname = None)
    # ii is number of "initial infected" nodes

    # g is a graph, n counts the number of nodes
    n = length(g.node)
    
    # define dictionary of SIRS model parameters
    disease_exponential={
        # rate of infection of neighbor for fully mixed case
        # BE CAREFUL: this should be beta, divide out mean degree from R0!
        'i'=>beta, 
        'r'=>gamma, # infectious to removed
        'w'=>rho, # removed to susceptible 
    }
    
    # define observation times vector
    if otimes == None
        otimes = [1.0, 3.0, 10.0, 30.0]
    end
    
    # define the output file name
    if outname == None
        outname = "z.hdf5"
    end
    
    # initialize RNG
    for init_idx in 2:nprocs()
        remotecall(init_idx, set_rng, seed)
    end
    
    # create array for parallel simulations
    work=Array(Any,nruns)
    for i in 1:nruns
        # pass parameters dictionary, graph, and observation times
        # to the algorithm
        work[i]=(disease_exponential, g, ii, otimes)
    end
    
    # apply mapping
    # execute simulation in parallel
    r=pmap(work) do package
        apply(herd_graph, package)
    end

    # construct 3-d array for the results
    # 1st dimension is the run number
    # 2nd dimension is the number of states
    # 3rd dimension is observation time
    results=zeros(Float64, nruns, 4, length(otimes))
        
    # fill in the results array
    for (run_idx, entry) in enumerate(r)
        for (obs_idx, obs) in enumerate(entry)
            if obs[2]>0.0001 # used to be obs[4]
                results[run_idx, 1, obs_idx]=obs[1]
                results[run_idx, 2, obs_idx]=obs[2]
                results[run_idx, 3, obs_idx]=obs[3]
                results[run_idx, 4, obs_idx]=obs_idx
                #println(obs)
            else
                results[run_idx, 1, obs_idx]=-1
                results[run_idx, 2, obs_idx]=-1
                results[run_idx, 3, obs_idx]=-1
                results[run_idx, 4, obs_idx]=obs_idx
            end
        end
    end
    
    # Output results array
    #writedlm(outname, results, ',');
    # Use hdf5 to transfer files from Julia environment
    f = h5open(outname, "w");
    d_write(f, "sirs", results);    
    d_write(f, "otimes", otimes);
    # write out parameters as attributes 
    # (can't seem to transfer dicts with hdf5)
    attrs(f)["nruns"] = nruns
    attrs(f)["n"] = n
    attrs(f)["beta"] = beta
    attrs(f)["gamma"] = gamma
    attrs(f)["rho"] = rho    
    close(f);
end

# ends the module
end