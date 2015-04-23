include("tracing.jl")
using DataFrames
using Gadfly
using DataStructures
push!(LOAD_PATH, "../../src")
using CTDE
using CTDE.SmallGraphs
import CTDE: enabled_transitions, current_time, current_time!
import CTDE: fire, init
import Base: zero, convert, show

typealias Time Float64

####################################################
# All functions for defining graphs
####################################################

function complete_contact_graph(cnt::Int64)
    g=UndirectedGraph()
    for i in 1:cnt
        for j in 1:cnt
            if i!=j
                add_edge(g, i, j)
            end
        end
    end
    g
end


function contact_graph(edgedata::Array{Int64,2})
    # edgedata is an array, where each row is a pair of linked nodes
    # edgedata is generated from nx data in Python
    g = UndirectedGraph()
    for pair_idx in range(1, size(edgedata)[1])
        i, j = edgedata[pair_idx, :]
        if i != j
            add_edge(g, i, j)
        end
    end
    g
end

function graph_node_degree(g::UndirectedGraph)
    # returns a dictionary {node_index => node_degree}
    Dict{Int64,Int64}([key => length(g.edge[key]) for key in keys(g.node)])
end

function degree_class_numbers(g::UndirectedGraph)
    # returns a vector of number of nodes in the kth degree class
    # degree classes are ordered from k = 1 to k = k_max
    ks = graph_node_degree(g);
    pks = DefaultDict(Int64, Int64, 0)
    for k in values(ks)
       pks[k] += 1
    end
    d = Array{Int64}([pks[k] for k in sort([k for k in keys(pks)])])
    d
end

function graph_node_class_index(g::UndirectedGraph)
    # returns a dictionary {node_index => node degree class index}
    ks = graph_node_degree(g)
    kclasses = Dict{Int64, Int64}([i => 0 for i in values(ks)])
    sorted_degrees = sort([k for k in keys(kclasses)])
    for k in keys(kclasses)
       kclasses[k] = find(x->x==k, sorted_degrees)[1]
    end
    # now create a new dictionary
    Dict{Int64, Int64}([k => kclasses[ks[k]] for k in keys(ks)])
end

####################################################
# Define all functions for exponential reactions & transitions
####################################################

function individual_exponential_graph(params, contact::UndirectedGraph)
    state=TokenState(int_marking())
    model=ExplicitGSPNModel(state)
    cnt=length(contact)
    structure=model.structure
    for i in 1:cnt
        add_place(structure, (i,'s')) # susceptible
        add_place(structure, (i,'i')) # infectious
        add_place(structure, (i,'r')) # recovered
    end

    # The basic SIRS
    for (node, properties) in contact.node
        i=node
        recover=ConstExplicitTransition(
            (lm, when::Time)->begin
                (TransitionExponential(params['r'], when), Int[])
            end)
        wane=ConstExplicitTransition(
            (lm, when::Time)->begin
                (TransitionExponential(params['w'], when), Int[])
            end)
        add_transition(structure, (i, i, 'd'), recover,
            [((i,'i'), -1), ((i,'r'), 1)],
            [])
        add_transition(structure, (i, i, 'w'), wane,
            [((i,'r'), -1), ((i,'s'), 1)],
            [])
    end

    # loop over all edges
    for (source, targets) in contact.edge
        for (target, properties) in targets
            infect=ConstExplicitTransition((lm, when::Time)->begin
                (TransitionExponential(params['i'], when), Int[])
                end)
            (i, j)=(source, target)
            add_transition(structure, (i, j,'g'), infect,
                [((i,'i'),-1), ((j,'s'), -1), ((i,'i'),1), ((j,'i'),1)],
                [])
        end
    end
    model
end

function initialize_marking(model, contact)
    # assign initially infected node
    first_infection=1 # Could pick this randomly.
    node_idx=1
    for (node, properties) in contact.node
        if node_idx!=first_infection
            add_tokens(model, (node,'s'), 1)
        else
            add_tokens(model, (node,'i'), 1)
        end
        node_idx+=1
    end
end

function initialize_marking_multiple(model, contact, init_s, init_i)
    # takes as argument number of nodes to be initially infected
    snodes = [range(1,init_s);]
    inodes = [range(init_s+1, init_i);]
    #println(inodes) # print for debugging
    for (node, properties) in contact.node
        if in(node, snodes)
            # 1st argument is the "model" object constructed earlier
            # 2nd argument is the place where the token is added
            #       place includes location (node) and type
            # 3rd argument is the number of tokens added
            add_tokens(model, (node,'s'), 1)
        elseif in(node,inodes)
            add_tokens(model, (node,'i'), 1)
        else
            add_tokens(model, (node,'r'), 1)
        end
    end
end

function initialize_marking_hmft(model, contact, init_s::Array{Int64,1}, 
                                 init_i::Array{Int64,1})
    # want to create a list of nodes belonging to each degree class
    h = graph_node_class_index(contact)
    d = DefaultDict(Int64, Array{Int64, 1}, Array{Int64,1}[]);
    for item in h
       push!(d[item[2]], item[1])
    end
    # create lists of all nodes to be infected, susceptible, recovered
    inodes = flatten([d[k][1:init_i[k]] for k in keys(d)])
    snodes = flatten([d[k][init_i[k]+1:init_i[k]+init_s[k]] for k in keys(d)])
    # assign initial states to all nodes
#    println(snodes)
#    println(inodes)
    for (node, properties) in contact.node
#        if in(node, inodes)
#            add_tokens(model, (node, 'i'), 1)
#        else
#            add_tokens(model, (node, 's'), 1)
#        end
        if in(node, snodes)
            add_tokens(model, (node, 's'), 1)
        elseif in(node, inodes)
            add_tokens(model, (node, 'i'), 1)
        else
            add_tokens(model, (node, 'r'), 1)
        end
    end
end

flatten{T}(a::Array{T,1}) = any(map(x->isa(x,Array),a))? flatten(vcat(map(flatten,a)...)): a
flatten{T}(a::Array{T}) = reshape(a,prod(size(a)))
flatten(a)=a

export flatten

####################################################
# Define all objects for a Record SIRS simulation
# Gives full information about all events that have taken place
####################################################

type TrajectoryFull
  when::Float64
  event::Int64
  who::Int64
  actor::Int64
end

function getindex(self::TrajectoryFull, i::Int64)
  codes = Dict(1=>self.when, 2=>self.event, 3=>self.who, 4=>self.actor)
  codes[i]
end

type RecordObserver
  trajectory::Array{TrajectoryFull,1}
  endtime::Float64
  RecordObserver(maxtime) = new(
        Array(TrajectoryFull, 0), maxtime)
end

function observe(ro::RecordObserver, state)
  # what was the last event?
  last_fired=state.last_fired
  codes=Dict('g'=>0, 'd'=>1, 'w'=>2)
  push!(ro.trajectory,TrajectoryFull(
    state.current_time, codes[last_fired[3]],
    last_fired[2], last_fired[1]))
  # check that current time is before end time
  running = state.current_time<ro.endtime
  running
end

function full_record(params::Dict, contact::UndirectedGraph,        
                    initial_infected::Int, maxtime::Float64,
                    rng::MersenneTwister)
    model=individual_exponential_graph(params, contact)
    sampling=NextReactionHazards()
    # initialize the observer, returns full trajectory information
    observer=RecordObserver(maxtime)
    model.state=TokenState(int_marking())
    initialize_marking_multiple(model, contact, initial_infected)
    run_steps(model, sampling, s->observe(observer, s), rng)
    observer.trajectory
    # from this output, can use list comprehensions to extract
    # full trajectory
    # eg:
    # transitions=[data[i][2] for i in 1:length(data)];
    # future work: will need a way to translate this info into
    # actual SIRS trajectory information
end

function full_record(params::Dict, g::UndirectedGraph, 
                     ii::Int, maxtime::Float64)
    global rng
    full_record(params, g, ii, maxtime, rng)
end


####################################################
# Define all objects for a Herd Graph SIRS simulation
####################################################

typealias TrajectoryEntry (Int64,Int64,Int64,Time)
  zero(::Type{TrajectoryEntry})=(0,0,0,0.)

type TrajectoryStruct
  s::Int64
  i::Int64
  r::Int64
  t::Time
  TrajectoryStruct()=new(0,0,0,0.)
  TrajectoryStruct(s_,i_,r_,t_)=new(s_,i_,r_,t_)
end

function convert(::Type{TrajectoryEntry}, x::TrajectoryStruct)
    (x.s, x.i, x.r, x.t)
end

type HerdDiseaseObserver
    t::Array{TrajectoryEntry,1}
    cnt::Int
    sir::TrajectoryStruct
    previous_time::Time
    observation_times::Array{Time,1}
    observations_at_times::Array{TrajectoryEntry,1}
    # 10_000 = array size, can change if throws error 
    HerdDiseaseObserver(cnt, init_s, init_i, obs_times)=new(
            Array(TrajectoryEntry, 10_000), 1, 
            TrajectoryStruct(init_s, init_i, int(cnt-init_s-init_i), 0.),          
            0., obs_times, zeros(TrajectoryEntry, length(obs_times)))
end

function observe(eo::HerdDiseaseObserver, state)
    last_fired=state.last_fired
    delta=state.current_time-eo.previous_time
    running=true
    # specific times at which observations occur
    # digs through full past trajectories and match to observation times
    # if brackets observation time, use previous state
    # inside loop: we are checking on 
    for for_idx in 1:length(eo.observation_times)
        if (eo.previous_time<=eo.observation_times[for_idx] &&
                state.current_time>eo.observation_times[for_idx])
            eo.observations_at_times[for_idx]=convert(TrajectoryEntry, eo.sir)
            if for_idx==length(eo.observations_at_times)
                running=false
            end
            break
        end
    end

    # look at observer
    # xo.cnt = number of events
    shiftstore= xo->begin
        xo.sir.t=state.current_time
        xo.t[xo.cnt]=convert(TrajectoryEntry, xo.sir)
        xo.cnt+=1
    end
    # recover
    if last_fired[3]=='d'
        eo.sir.i-=1
        eo.sir.r+=1
        shiftstore(eo)
    # wane
    elseif last_fired[3]=='w'
        eo.sir.r-=1
        eo.sir.s+=1
        shiftstore(eo)
    # infect
    elseif last_fired[3]=='g'
        eo.sir.s-=1
        eo.sir.i+=1
        shiftstore(eo)
    else
        error("No transition known")
    end
    #println("eo.sir: ", eo.sir, " last ", last_fired)
    if eo.cnt>length(eo.t)
        new_len=2*eo.cnt
        new_t=Array(TrajectoryEntry, new_len)
        new_t[1:length(eo.t)]=eo.t
        eo.t=new_t
    end
    eo.previous_time=state.current_time
    # Condition for continuing simulation
    running
end

function show(eo::HerdDiseaseObserver)
  for i in 1:(eo.cnt-1)
      println(eo.t[i][1], " ", eo.t[i][2], " ", eo.t[i][3], " ", eo.t[i][4])
  end
end

function epidemic_size(eo::HerdDiseaseObserver)
    eo.t[eo.cnt-1][3]
end


function herd_graph(params::Dict, contact::UndirectedGraph,        
                    init_s::Int, init_i::Int, 
                    obs_times::Array{Time,1}, rng::MersenneTwister)
    model=individual_exponential_graph(params, contact);
    sampling=NextReactionHazards();
    cnt = length(contact.node); # graph size
    observer=HerdDiseaseObserver(cnt, init_s, init_i, obs_times);
    model.state=TokenState(int_marking());
    initialize_marking_multiple(model, contact, init_s, init_i);
    # initialize_marking(model, contact)
    run_steps(model, sampling, s->observe(observer, s), rng)
    observer.observations_at_times
end

# This one pulls rng from scope so that it can be initialized in parallel.
function herd_graph(params::Dict, g::UndirectedGraph, 
                    init_s::Int, init_i::Int,
                    obs_times::Array{Time,1})
    # ii is number of "initial infected" nodes
    global rng
    herd_graph(params, g, init_s, init_i, obs_times, rng)
end


####################################################
# Define all objects for HMFT observer
####################################################

typealias HMFState (Array{Int64,1},Array{Int64,1},Array{Int64,1},Time)
zero(::Type{HMFState})=(Array{Int64,1},Array{Int64,1},Array{Int64,1},0.)

type HMFTrajectoryStruct
  s::Array{Int64, 1}
  i::Array{Int64, 1}
  r::Array{Int64, 1}
  t::Time
  #HMFTrajectoryStruct()=new(Array{Int64,1},Array{Int64,1},Array{Int64,1},0.)
  HMFTrajectoryStruct(s_::Array{Int64,1},i_::Array{Int64,1},
                      r_::Array{Int64,1},t_::Float64)=new(s_,i_,r_,t_)
end

type HMFObserver
    # current state
    sir::HMFTrajectoryStruct
    # number of nodes in each degree class
    degree_class::Array{Int64, 1}
    # keep track of the index of each node's degree
    node_classes::Dict{Int64, Int64}
    previous_time::Time
    # specify times when we observe the state of the system
    otimes::Array{Time,1}
    # these are the actual times when observations take place
    observation_times::Array{Time,1}
    # array of system states at each of observation_times
    # 1st axis: SIR, 2nd axis: degree class, 3rd axis: time index
    state_observed_times::Array{Int64,3}
    # 10_000 = array size, can change if throws error 
    HMFObserver(g::UndirectedGraph, init_s::Array{Int64,1}, 
                init_i::Array{Int64,1}, obs_times::Array{Float64,1})=new( 
            HMFTrajectoryStruct(copy(init_s), copy(init_i), 
                            copy(degree_class_numbers(g)-init_s-init_i), 0.),          
            degree_class_numbers(g), graph_node_class_index(g), 0., 
            copy(obs_times), copy(obs_times), 
            zeros(Int64, 3, length(degree_class_numbers(g)), 
                  length(obs_times)))
end

function observe(ho::HMFObserver, state)
    last_fired=state.last_fired
    delta=state.current_time-ho.previous_time
    running=true
    # specific times at which observations occur
    # digs through full past trajectories and match to observation times
    # if brackets observation time, use previous state
    # inside loop: we are checking on 
    for t_index in 1:length(ho.otimes)
        if (ho.previous_time<=ho.otimes[t_index] &&
                state.current_time>ho.otimes[t_index])
            # store current state
            ho.state_observed_times[:,:,t_index] = 
                            [ho.sir.s ho.sir.i ho.sir.r]'
            # store current time
            ho.observation_times[t_index] = ho.sir.t
            if t_index>=length(ho.otimes)
                running=false
            end
            break
        end
    end

    # which node made a transition?
    who = last_fired[2]
    # what's the degree class of that node?
    kclass = ho.node_classes[who]
    # recover
    if last_fired[3]=='d'
        ho.sir.i[kclass]-=1
        ho.sir.r[kclass]+=1
        ho.sir.t = state.current_time
    # wane
    elseif last_fired[3]=='w'
        ho.sir.r[kclass]-=1
        ho.sir.s[kclass]+=1
        ho.sir.t = state.current_time
    # infect
    elseif last_fired[3]=='g'
        ho.sir.s[kclass]-=1
        ho.sir.i[kclass]+=1
        ho.sir.t = state.current_time
    else
        error("No transition known")
    end
    # update the time of the previous transition
    ho.previous_time=state.current_time
    # Condition for continuing simulation
    running
end

function hmft_graph(params::Dict, contact::UndirectedGraph,        
                    init_s::Array{Int64, 1}, init_i::Array{Int64, 1}, 
                    obs_times::Array{Time,1}, rng::MersenneTwister)
    model=individual_exponential_graph(params, contact);
    sampling=NextReactionHazards();
    observer=HMFObserver(contact, init_s, init_i, obs_times);
    model.state=TokenState(int_marking());
    initialize_marking_hmft(model, contact, init_s, init_i);
    run_steps(model, sampling, s->observe(observer, s), rng)
    observer.state_observed_times, observer.observation_times
end

# This one pulls rng from scope so that it can be initialized in parallel.
function hmft_graph(params::Dict, g::UndirectedGraph, init_s::Array{Int64,1}, 
                    init_i::Array{Int64, 1}, obs_times::Array{Time,1})
    # ii is number of "initial infected" nodes
    global rng
    hmft_graph(params, g, init_s, init_i, obs_times, rng)
end

####################################################
# Initialize pseudo-random number generator
####################################################

function set_rng(seed)
    global rng
    rng=MersenneTwister(seed+myid())
    nothing
end

export set_rng


####################################################
# Extract the actual state of the system
####################################################
function model_state_explicit(model::ExplicitGSPNModel)
    s = Int64[]
    i = Int64[]
    r = Int64[]
    for (j, k) in model.state.marking.dict
        nodestate = model.structure.id_to_pt[j]
        if nodestate[2] == 'i' d
            push!(i, nodestate[1])
        elseif nodestate[2] == 's'
            push!(s, nodestate[1])
        else
            push!(r, nodestate[1])
        end
    end
    s, i, r
end