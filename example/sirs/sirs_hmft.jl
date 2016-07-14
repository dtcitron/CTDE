#!/Applications/Julia-0.3.0.app/Contents/Resources/julia/bin/julia

require("sirs.jl")
using HDF5
using Compat

# variables glossary
# n : number of individuals
# nruns : number of times to run the simulation
# beta : infection rate (will divide by number of contacts in the code below)
# gamma : recovery rate
# rho : waning rate
# seed : seed for RNG, defaults to 34
# otimes : observation times, defaults to a particular sequence but can be added

function sirs_hmft(nruns, g, beta, gam, rho, 
                  init_s, init_i, otimes = None,
                  outname = None, seed = 34)
    # nruns = number of runs of the SIRS dynamics with this parameter set
    # g = substrate graph for the simulation
    # beta = r0*gamma/<k>, rate of transmission parameter
    # gamma = rate of recovery parameter
    # rho = alpha*gamma, rate of waning immunity parameter
    # otimes = Float list of times when observations are made
    # outname = String name where the output will be written to hdf5 file
    # init_s = initial number of susceptibles
    # init_i = initial number of infecteds
    # seed = random seed
    
    # This script supports a HMFT coarse-graining of the graph into degree 
    # classes, meaning that we need to know all of the degree classes for each 
    # node, and the number of nodes in each degree class
    # d = list, where kth element is number of nodes in kth class
    d = degree_class_numbers(g)
    ks = length(d) # number of degree classes
    
    # define dictionary of SIRS model parameters
    @compat disease_exponential=Dict(
        # rate of infection of neighbor for fully mixed case
        # BE CAREFUL: this should be beta, divide out mean degree from R0!
        'i'=>beta, 
        'r'=>gam, # infectious to removed
        'w'=>rho, # removed to susceptible 
    )
    
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
    if nprocs() == 1
        set_rng(seed)
    end

    # create array for parallel simulations
    work=Array(Any,nruns);
    for i in 1:nruns
        # pass parameters dictionary, graph, and observation times
        # to the algorithm
        work[i]=(disease_exponential, g, init_s, init_i, otimes)
    end

    if nprocs() == 1
        # observed states
        osts = Array(Any, nruns);
        # observation times
        ots = Array(Any, nruns);
        for i in 1:nruns
            osts[i], ots[i] = apply(hmft_graph, (disease_exponential, g, 
                                      init_s, init_i, otimes))
        end
    else
        # apply mapping
        # execute simulation in parallel
        r = pmap(work) do package
            apply(hmft_graph, package)
        end
        # observed states
        osts = Array(Any, nruns);
        # observation times
        ots = Array(Any, nruns);
        for i in 1:nruns
            osts[i], ots[i] = r[i]
        end
    end

    # construct 4-d array for the results
    # 1st dimension is the run number
    # 2nd dimension is the state (SIR)
    # 3rd dimension is the degree class index
    # 4th dimension is observation time
    results=zeros(Int64, nruns, 4, ks, length(otimes))

    # fill in the results array
    for (run_idx, entry) in enumerate(osts)
      ss, ks, ts = size(entry)
      for k_idx in 1:ks
        for time_idx in 1:ts
          if ots[run_idx][time_idx] != otimes[time_idx]
            results[run_idx, 1, k_idx, time_idx] = entry[1, k_idx, time_idx]
            results[run_idx, 2, k_idx, time_idx] = entry[2, k_idx, time_idx]
            results[run_idx, 3, k_idx, time_idx] = entry[3, k_idx, time_idx]
            results[run_idx, 4, k_idx, time_idx] = time_idx
          else
            results[run_idx, 1, k_idx, time_idx] = -1
            results[run_idx, 2, k_idx, time_idx] = -1
            results[run_idx, 3, k_idx, time_idx] = -1
            results[run_idx, 4, k_idx, time_idx] = -1
          end
        end
      end
    end

    # Use hdf5 to transfer files from Julia environment
    f = h5open(outname, "w");
    d_write(f, "sirs", results);    
    d_write(f, "otimes", otimes);
    # write out parameters as attributes 
    # (can't seem to transfer dicts with hdf5)
    attrs(f)["nruns"] = nruns
    attrs(f)["graphsize"] = d
    attrs(f)["beta"] = beta
    attrs(f)["gamma"] = gam
    attrs(f)["rho"] = rho    
    attrs(f)["kmean"] = mean([k for k in values(graph_node_degree(g))])
    attrs(f)["otimes"] = otimes
    close(f);

end

function sirs_diagram(nruns, g, alphas, r0s, otimes, kmean, seed, gname)
    # default value for gamma
    gam = 1.;
    n = length(g.node);
    d = degree_class_numbers(g);
    # get list of files already in path
    path_files = output_path(gname)
    for rho_idx in range(1,length(alphas))
        for r0_idx in range(1,length(r0s))
            alpha = alphas[rho_idx]
            rho = alpha*gam;
            r0 = r0s[r0_idx];
            beta = r0/kmean*gam;
            outname = string(gname, "_rho_", rho_idx - 1, 
                             "_r0_", r0_idx - 1,".hdf5") 
            tic()
            println("rho = ", rho, ", beta = ", beta)
            # check whether or not this file has already been completed...
            
            # initial state, defaults to (S,I,R) being (90%,10%,0%) of all         
            # nodes -or- approximates the mean field steady state numbers at 
            # the start
            init_s = min(int(1.*d/r0), int(floor(9.*d/10)))
            init_i = max(int(1.*d*alpha/(1. + alpha)*(1 - 1./r0)), int(d/10.))
            if !in(outname, path_files)
                sirs_hmft(nruns, g, beta, gam, rho, init_s, init_i, 
                          otimes, outname, seed)
            end
            toc()
        end
    end
end

function output_path(gname)
    # return a list of all files currently in the output path
    # this will avoid repeating single output files
    r = split(gname, "/")
    p = ""
    for i in 1:length(r)-1
        p = p*r[i]*"/"
    end
    p = p[1:end-1]
    if p != ""
        [p*"/"*i for i in readdir(p)]
    else
        []
    end
end

export sirs_graph, sirs_diagram, output_path