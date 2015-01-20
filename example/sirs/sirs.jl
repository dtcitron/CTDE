include("tracing.jl")
using DataFrames
using Gadfly
push!(LOAD_PATH, "../../src")
using CTDE
using CTDE.SmallGraphs
import CTDE: enabled_transitions, current_time, current_time!
import CTDE: fire, init
import Base: zero, convert, show

typealias Time Float64

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

function initialize_marking_multiple(model, contact, init_infection = 1)
    # takes as argument number of nodes to be initially infected
    inodes = [range(1,init_infection)]
    #println(inodes) # print for debugging
    node_idx=1
    for (node, properties) in contact.node
        if !in(node_idx, inodes)
            # 1st argument is the "model" object constructed earlier
            # 2nd argument is the place where the token is added
            #       place includes location (node) and type
            # 3rd argument is the number of tokens added
            add_tokens(model, (node,'s'), 1)
        else
            add_tokens(model, (node,'i'), 1)
        end
        node_idx+=1
    end
end

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
    HerdDiseaseObserver(cnt, init_infecteds, obs_times)=new(
            Array(TrajectoryEntry, 10_000), 1, 
            TrajectoryStruct(int(cnt-init_infecteds), init_infecteds, 0, 0.),          
            0., obs_times, zeros(TrajectoryEntry, length(obs_times)))
end

function observe(eo::HerdDiseaseObserver, state)
    last_fired=state.last_fired
    delta=state.current_time-eo.previous_time
    running=true
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

    shiftstore= xo->begin
        xo.sir.t=state.current_time
        xo.t[xo.cnt]=convert(TrajectoryEntry, xo.sir)
        xo.cnt+=1
    end
    if last_fired[3]=='d'
        eo.sir.i-=1
        eo.sir.r+=1
        shiftstore(eo)
    elseif last_fired[3]=='w'
        eo.sir.r-=1
        eo.sir.s+=1
        shiftstore(eo)
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

function complete_contact_graph(cnt)
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


function contact_graph(edgedata)
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


function herd_model(params, cnt, run_cnt, obs_times, rng)
    contact=complete_contact_graph(cnt)
    model=individual_exponential_graph(params, contact)
    println("obs_times ", obs_times)
    for i in 1:run_cnt
        sampling=NextReactionHazards()
        observer=HerdDiseaseObserver(cnt, 1, obs_times)
        model.state=TokenState(int_marking())
        initialize_marking(model, contact)
        run_steps(model, sampling, s->observe(observer, s), rng)
        println(observer.observations_at_times)
    end
end


function herd_single(params::Dict, cnt::Int, initial_infected::Int,     
                     obs_times::Array{Time,1}, rng::MersenneTwister)
    contact=complete_contact_graph(cnt)
    model=individual_exponential_graph(params, contact)
    sampling=NextReactionHazards()
    observer=HerdDiseaseObserver(cnt, initial_infected, obs_times)
    model.state=TokenState(int_marking())
    initialize_marking_multiple(model, contact, initial_infected)
    #initialize_marking(model, contact)
    run_steps(model, sampling, s->observe(observer, s), rng)
    observer.observations_at_times
end

# This one pulls rng from scope so that it can be initialized in parallel.
function herd_single(params::Dict, cnt::Int, ii::Int, obs_times::Array{Time,1})
    # ii is number of "initial infected" nodes
    global rng
    herd_single(params, cnt, ii, obs_times, rng)
end


function herd_graph(params::Dict, contact::UndirectedGraph,        
                    initial_infected::Int, obs_times::Array{Time,1},
                    rng::MersenneTwister)
    model=individual_exponential_graph(params, contact)
    sampling=NextReactionHazards()
    cnt = length(contact.node)
    observer=HerdDiseaseObserver(cnt, initial_infected, obs_times)
    model.state=TokenState(int_marking())
    initialize_marking_multiple(model, contact, initial_infected)
    # initialize_marking(model, contact)
    run_steps(model, sampling, s->observe(observer, s), rng)
    observer.observations_at_times
end

# This one pulls rng from scope so that it can be initialized in parallel.
function herd_graph(params::Dict, g::UndirectedGraph, 
                    ii::Int, obs_times::Array{Time,1})
    # ii is number of "initial infected" nodes
    global rng
    herd_graph(params, g, ii, obs_times, rng)
end


function set_rng(seed)
    global rng
    rng=MersenneTwister(seed+myid())
    nothing
end

export set_rng
