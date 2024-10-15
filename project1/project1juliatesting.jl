using Pkg
Pkg.activate(".")

using Graphs
using Printf

"""
    write_gph(dag::DiGraph, idx2names, filename)

Takes a DiGraph, a Dict of index to names and a output filename to write the graph in `gph` format.
"""
function write_gph(dag::DiGraph, idx2names, filename)
    open(filename, "w") do io
        for edge in edges(dag)
            @printf(io, "%s,%s\n", idx2names[src(edge)], idx2names[dst(edge)])
        end
    end
end


function compute(infile, outfile)

    # WRITE YOUR CODE HERE
    # FEEL FREE TO CHANGE ANYTHING ANYWHERE IN THE CODE
    # THIS INCLUDES CHANGING THE FUNCTION NAMES, MAKING THE CODE MODULAR, BASICALLY ANYTHING

end

# if length(ARGS) != 2
#     error("usage: julia project1.jl <infile>.csv <outfile>.gph")
# end

# inputfilename = ARGS[1]
# outputfilename = ARGS[2]

# compute(inputfilename, outputfilename)



### GRAPH SAMPLE 2 ###
# using GraphPlot

# g = smallgraph(:chvatal)
# node_names = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]

# p = gplot(g; nodelabel=node_names)

# # Save using Compose
# using Compose, Cairo, Fontconfig
# draw(PDF("chvatal_graphplot.pdf", 16cm, 16cm), p)

# println("GraphPlot worked")

# # Test directed
# # Create a simple directed graph with 4 vertices
# g = SimpleDiGraph(4)

# # Add directed edges (from node to node)
# add_edge!(g, 1, 2)
# add_edge!(g, 2, 3)
# add_edge!(g, 3, 4)
# add_edge!(g, 1, 4)

# # Plot the directed graph with arrows
# dirgplot = gplot(g; nodelabel=[:A, :B, :C, :D])
# draw(PDF("dirgplot.pdf", 16cm, 16cm), dirgplot)
# println("Directed GraphPlot worked")



# ### GRAPH SAMPLE 3 ###
using Plots
using GraphRecipes


g = smallgraph(:chvatal)
node_names = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]

p = graphplot(g; names=node_names, node_size=0.2)

savefig(p, "chvatal_graphrecipes.pdf")
println("GraphRecipes worked")

# Test directed
# Create a simple directed graph with 4 vertices
g = SimpleDiGraph(4)

# Add directed edges (from node to node)
add_edge!(g, 1, 2)
add_edge!(g, 2, 3)
add_edge!(g, 3, 4)
add_edge!(g, 1, 4)

# Plot the directed graph with arrows
dirgraphplot = graphplot(g; names=[:A, :B, :C, :D])
savefig(dirgraphplot, "dirgraphplot.pdf")
println("Directed GraphRecipes worked")

println("At end")









### GRAPH SAMPLE 1 - Never worked ###
# using Graphs  # for DiGraph and add_edge!
# using TikzGraphs   # for TikZ plot output

# # An example [Chvatal Graph](https://en.wikipedia.org/wiki/Chv%C3%A1tal_graph)
# g = smallgraph(:chvatal) 

# # Create notional names for the nodes
# node_names = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]

# # Create TikZ plot with node labels
# p = plot(g, node_names)

# # Save as PDF (using TikzPictures)
# using TikzPictures # to save TikZ as PDF
# save(PDF("/Users/trevorperey/chvatal_tikz.pdf"), p)

# println("Tikz worked")