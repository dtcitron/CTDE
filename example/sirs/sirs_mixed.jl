#!/Applications/Julia-0.3.0.app/Contents/Resources/julia/bin/julia

module SIRSMIXED

# This is a script that allows us to call the code from sirs_parallel.jl
# specifically, we will run fully mixed SIRS simulations
#   depending on how the julia environment was called, this may be run in parallel

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


function sirs_mixed(nruns, n, beta, gamma, rho, seed = 34, otimes = None, outname = None)
    
    # define dictionary of SIRS model parameters
    disease_exponential={
        'i'=>beta/n, # rate of infection of neighbor for fully mixed case
        'r'=>gamma, # infectious to removed
        'w'=>rho, # removed to susceptible 
    }
    
    # define observation times vector
    if otimes == None
        otimes = [1.0, 3.0, 10.0, 30.0]
    end
    
    # define the output file name
    if outname == None
        outname = "z.txt"
    end
    
    # initialize RNG
    for init_idx in 2:nprocs()
        remotecall(init_idx, set_rng, seed)
    end
    
    # create array for parallel simulations
    work=Array(Any,nruns)
    for i in 1:nruns
        work[i]=(disease_exponential, n, otimes)
    end
    
    # apply mapping
    # execute simulation in parallel
    r=pmap(work) do package
        apply(herd_single, package)
    end

    # construct 3-d array for the results
    # 1st dimension is the run number
    # 2nd dimension is the number of states
    # 3rd dimension is observation time
    results=zeros(Float64, nruns, 4, length(otimes))
        
    # fill in the results array
    for (run_idx, entry) in enumerate(r)
        for (obs_idx, obs) in enumerate(entry)
            if obs[4]>0.0001
                results[run_idx, 1, obs_idx]=obs[1]
                results[run_idx, 2, obs_idx]=obs[2]
                results[run_idx, 3, obs_idx]=obs[3]
                results[run_idx, 4, obs_idx]=obs_idx
            else
                results[run_idx, 1, obs_idx]=-1
                results[run_idx, 2, obs_idx]=-1
                results[run_idx, 3, obs_idx]=-1
                results[run_idx, 4, obs_idx]=obs_idx
            end
        end
    end
    
    # output results array
    writedlm(outname, results, ',')
    # h5write("z.h5", "sir", results)
    # Use "h5dump z.h5" to see what's in there.
end

# can't seem to find an obvious way to reconstitute text file output from julia in julia
#function readtxt(filename):
#    f = open(filename)
#    s = readdlm(f, ',')
#    close(f)
#    return z
#end
        

# ends the module
end