module SmallGraphs
# This is in the style of the NetworkX package in Python.

import Base: length
export SmallGraph, DirectedGraph, UndirectedGraph, UndirectedMultiGraph
export add_node, add_edge, out_degree, neighbors, length

abstract SmallGraph

#######################=============================#
# Dictionary-based graph implementation.
# g=DirectedGraph()
# add_node(g, 3)
# add_node(g, "hi", {"name"=>"joe"})
# add_edge(g, "chicago", "seattle", {"distance"=>1721})
# add_edge(g, 3, 9)
# g.node["hi"]["name"]=="joe"
# for target_node, properties in g.edge["chicago"]
#   @debug("Distance to ", target_node, " is ", properties["distance"])
# end
#####################################################
type DirectedGraph <: SmallGraph
	node::Dict{Any,Dict{Any,Any}}
	edge::Dict{Any,Dict{Any,Dict{Any,Any}}}
end

function DirectedGraph()
	nodes=Dict{Any,Dict{Any,Any}}()
	edges=Dict{Any,Dict{Any,Dict{Any,Any}}}()
	DirectedGraph(nodes, edges)
end

function add_node(g::SmallGraph, node, dict)
	g.node[node]=dict
	g.edge[node]=Dict{Any,Dict{Any,Dict{Any,Any}}}()
end

add_node(g::SmallGraph, node)=add_node(g, node, Dict{Any,Dict{Any,Any}}())

function add_edge(g::DirectedGraph, source, target, dict)
	if !haskey(g.node, source)
		add_node(g, source)
	end
	if !haskey(g.node, target)
		add_node(g, target)
	end
	g.edge[source][target]=dict
end

add_edge(g::DirectedGraph, source, target)=
	add_edge(g, source, target, Dict{Any,Any}())


#####################
type UndirectedGraph <: SmallGraph
	node::Dict{Any,Dict{Any,Any}}
	edge::Dict{Any,Dict{Any,Dict{Any,Any}}}
end

function UndirectedGraph()
	nodes=Dict{Any,Dict{Any,Any}}()
	edges=Dict{Any,Dict{Any,Dict{Any,Any}}}()
	UndirectedGraph(nodes, edges)
end

function add_edge(g::UndirectedGraph, source, target, dict)
	if !haskey(g.node, source)
		add_node(g, source)
	end
	if !haskey(g.node, target)
		add_node(g, target)
	end
	g.edge[source][target]=dict
	g.edge[target][source]=dict
end

add_edge(g::UndirectedGraph, source, target)=
	add_edge(g, source, target, Dict{Any,Any}())

####################
out_degree(g::SmallGraph, node)=length(g.edge[node])
neighbors(g::SmallGraph, node)=keys(g.edge[node])
length(g::SmallGraph)=length(g.node)
#mean_degree(g::SmallGraph) = 

###############################################################
# The multigraph has a list of edges from each vertex instead
# of a dictionary indexed by target node.
# Each entry in the list is the target node and a dictionary
# associated with the edge.
# add_edge(g, "chicago", "seattle", {"distance"=>1721, "by"=>"crow"})
# add_edge(g, "chicago", "seattle", {"distance"=>2069, "by"=>"car"})
# for target_node, properties in g.edge["chicago"]
#   @debug("Distance is ", properties["distance"], " by ", properties["by"])
# end
###############################################################
type UndirectedMultiGraph <: SmallGraph
	node::Dict{Any,Dict{Any,Any}}
	edge::Dict{Any,Array{(Any, Dict{Any,Any}),1}}
end

function UndirectedMultiGraph()
	nodes=Dict{Any,Dict{Any,Any}}()
	edges=Dict{Any,Array{(Any, Dict{Any,Any}),1}}()
	UndirectedMultiGraph(nodes, edges)
end

function add_node(g::UndirectedMultiGraph, key, dict)
	g.node[key]=dict
	g.edge[key]=Array((Any, Dict{Any,Any}),0)
end

function add_edge(g::UndirectedMultiGraph, source, target, dict)
	if !haskey(g.node, source)
		add_node(g, source)
	end
	if !haskey(g.node, target)
		add_node(g, target)
	end
	push!(g.edge[source], (target, dict))
	push!(g.edge[target], (source, dict))
end

add_edge(g::UndirectedMultiGraph, source, target)=
	add_edge(g, source, target, Dict{Any,Any}())

neighbors(g::UndirectedMultiGraph, node)=Set([x[1] for x in g.edge[node]])

end # module SmallGraphs
