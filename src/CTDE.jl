module CTDE
    using Logging
    @Logging.configure(level=INFO)
    include("smallgraph.jl")
    include("samplesemimarkov.jl")
    include("marking.jl")
    include("explicitmodel.jl")
    include("transitiondistributions.jl")
    include("category_fsm.jl")
end
