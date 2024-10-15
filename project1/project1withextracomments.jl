using Pkg
Pkg.activate(".")

using Graphs
using Plots
using Printf
using DataFrames
using CSV
# using GraphPlot
# using Compose, Cairo, Fontconfig
using GraphRecipes
using LinearAlgebra
using SpecialFunctions

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

#=====================================
  STRUCTURES
======================================#

# Variable struct
struct Variable
    name::Symbol ## Variable name
    r::Int ## Number of possible values (1:r) for that variable
end

# Variable
struct K2Search
    ordering::Vector{Int} # Topological sort to use for K2.
    # Ints are not necessarily sorted
    # Correspond to order in original csv
end

# # Basic Variable test
# var = Variable(:X, 2)
# println(var.name)
# println(var.r)

#=====================================
  EXTRACTION - Helper functions
======================================#

### DATA EXTRACTION ###
function extract_data(filepath)
    alldata = CSV.read(filepath, DataFrame) # Get dataframe with all data

    # Build up Vector of variables
    var_names = Symbol.(names(alldata)) # Extract column names into list of vars

    vars = Vector{Variable}(undef, length(var_names)) # Initialize a vector for storing Variables
    index = 1
    for col in names(alldata) # Iterate over each column
        vars[index] = Variable(Symbol(col), maximum(alldata[!,col])) # Create Variable with name and r
        index += 1 # Move to next index of vars
    end

    # Generate matrix of data
    data_matrix = Matrix(alldata)
    data_matrix = collect(transpose(data_matrix)) # statistics func expects vars x data

    return vars, data_matrix
end

# vars, D = extract_data("./example/example.csv") # Test data extraction
# println(vars)
# println(D)

### GRAPH EXTRACTION (from .gph file) ###
function extract_graph(graphpath)

    # Extract edges from graphpath
    lines = readlines(graphpath) # Read in the file
    edges = [ (Symbol(split(line, ",")[1]), Symbol(split(line, ",")[2])) for line in lines] #Convert to symbol pairs 

    # Extract variables
    var_symbols = vcat([var for e in edges for var in e]...) # Use splat and vcat to break edges pairs into single array of symbols
    var_symbols = unique(var_symbols) # Reduce down to unique vars

    # Build graph
    var_index_map = Dict(var_symbols => index for (index, var_symbols) in enumerate(var_symbols))
    # Create tuples of (index, var) for all variables,
    # and then a mapping from each variable to its index

    g = SimpleDiGraph(length(var_symbols)) # Instantiate directed graph with nodes for unique variables 
    for (parent, child) in edges # For all edges in list
        add_edge!(g, var_index_map[parent], var_index_map[child]) # Use mapping to create that edge
    end

    return var_symbols, g

end

# nodes, g_test = extract_graph("./example/example.gph") # Test extracting graph
# gt_plot = gplot(g_test; nodelabel= nodes)
# draw(PDF("gt_plot.pdf", 16cm, 16cm), gt_plot, )
# println(nodes)

# indices = Dict(i => variable for (i, variable) in enumerate(nodes)) # Test reconstructing output
# println(indices)
# write_gph(g_test, indices, "./test_ex_build.gph")

# nodes_again, g_again = extract_graph("./test_ex_build.gph") # Extract graph again to confirm
# gagain_plot = gplot(g_again, nodelabel = nodes_again) 
# draw(PDF("gagain_plot.pdf", 16cm, 16cm), gt_plot, )
# println("Everything completed")

#=====================================
  SCORE - Helper functions
======================================#

### STATISTICS EXTRACTION (per pg. 75 of the text) ###

# Helper function for determining parental instatiation
function sub2ind(sze, x)
    k = vcat(1, cumprod(sze[1:end-1])) # Compute cumulative index product for each x (e.g. parent) value
    return dot(k, x.-1) + 1 # Compute linear index for particular parental instantiation
end


# Function for extracting counts M
function statistics(vars, G, D::Matrix{Int})

    n = size(D,1) # Number of variables
    r = [vars[i].r for i in 1:n] # Extract vector of ri's (number of values for each variable)
    q = [prod([r[j] for j in inneighbors(G,i)]) for i in 1:n] # Extract vector of qi's
    # (number of parental instantiations for each variable, which is product of r's of parents)

    # Build up counts matrix
    M = [zeros(q[i], r[i]) for i in 1:n] # n x qi x ri tensor. ith is qi x ri matrix of counts for variable i

    for o in eachcol(D) # Iterate through variables
        for i in 1:n # Iterate through values for a variable
            k = o[i] # Get current value

            # Determine parental instantiation
            parents = inneighbors(G,i) # Get parent indices
            j = 1 # Initialize j to 1 (in case no parents)
            if !isempty(parents)
                j = sub2ind(r[parents], o[parents]) # Pass the possible and actual parent values to sub2ind
            end
            M[i][j,k] += 1.0 # Have determined current value for variable i to be for parental instantiation j and value k. Increment.
        end
    end

    return M
end

# # Basic test (from textbook pg. 76)
# G = SimpleDiGraph(3)
# add_edge!(G, 1, 2)
# add_edge!(G, 3, 2)
# vars = [Variable(:A,2), Variable(:B,2), Variable(:C,2)]
# D = [1 2 2 1; 1 2 2 1; 2 2 2 2]
# M = statistics(vars, G, D)
# print(M) # Works

# # Test with example data
# nodes, G_test = extract_graph("./example/example.gph") # Extract ex graph
# vars, D_test = extract_data("./example/example.csv") # Extract ex data
# # println(vars)
# # println(G_test)
# # println(D_test)
# M = statistics(vars, G_test, D_test)
# println(M)


### UNIFORM PRIOR (per pg. 81 of the text) ### 
function prior(vars, G)
    n = length(vars) # Get number of variables
    r = [vars[i].r for i in 1:n] # Get vector of possible values ri for each variable idx2names
    q = [prod([r[j] for j in inneighbors(G,i)]) for i in 1:n] # Get vector of number of parental instantiations for each variable
    # qij = product of r's for parents of variable i
    return [ones(q[i], r[i]) for i in 1:n] # return qi x ri matrix of 1's (uniform) for each variable i
end

# nodes, G_test = extract_graph("./example/example.gph") # Extract ex graph
# vars, D_test = extract_data("./example/example.csv") # Extract ex data
# # println(vars)
# # println(G_test)
# alpha = prior(vars, G_test)
# println(alpha)
# together = M + alpha
# println(together)

### BAYESIAN SCORE (per pg. 98 of text) ### <-- REWRITE

# Helper for each ith Bayes score component
function bayesian_score_component(M, α)
    p = sum(loggamma.(α+M)) # Equ 5.5, term 2 numerator
    # Adds priors and counts, takes loggamma of each element, then sums matrix elements.
    # This sums over all qi and ri as desired.

    p -= sum(loggamma.(α)) # Equ 5.5, term 2 denominator
    # Takes loggamma of prior elements, then sums matrix elements.
    # This sums over all qi and ri as desired.

    p += sum(loggamma.(sum(α,dims=2))) # Equ 5.5, term 1 numerator
    # First, sums α over all ri for each j in qi. Gives vector of α_ij0 for each j up to qi.
    # Then, takes loggamma of vector elements and sums them. This sums loggamma(α_ij0) for all j up to qi as desired.

    p -= sum(loggamma.(sum(α,dims=2) + sum(M,dims=2))) # Equ 5.5, term 1 denominator
    # First, sum α and M over all ri for each j in qi. Gives vectors α_ij0 and m_ij0 for each j up to qi and sums them.
    # Then, takes loggamma of vector elements and sums them. This sums loggamma(α_ij0 + m_ij0) for each j up to qi as desired.
end

# Overall Bayes score
function bayesian_score(vars, G, D)
    n = length(vars) # Get number of variables
    M = statistics(vars, G, D) # Get counts m_ijk
    α = prior(vars, G) # Define uniform prior

    return sum(bayesian_score_component(M[i],α[i]) for i in 1:n)
    # Compute score components for each variable and sum them to return total score
end

# # Test with example
# nodes, G_test = extract_graph("./example/example.gph") # Extract ex graph
# vars, D_test = extract_data("./example/example.csv") # Extract ex data
# # println(vars)
# # println(G_test)

# score = bayesian_score(vars, G_test, D_test)
# println(score)

# # Test with empty graph (expect worse)
# G_empty = SimpleDiGraph(length(nodes))
# score_empty = bayesian_score(vars,G_empty,D_test)
# println(score_empty)

# # Test a different graph (chain)
# G_chain = SimpleDiGraph(length(nodes))

# for i in 1:(length(nodes)-1)
#     add_edge!(G_chain,i, (i+1))
# end

# score_chain = bayesian_score(vars,G_chain,D_test)
# println(score_chain)
# chainplot = gplot(G_chain; nodelabel = nodes)
# draw(PDF("chainplot.pdf", 16cm, 16cm), chainplot)

# # # Test with graph with random nodes (could be better or worse)
# # G_rand = SimpleDiGraph(length(nodes))
# # n = length(nodes)

# # for i in 1:n
# #     add_edge!(G_rand,rand(1:n),rand(1:n))
# # end

# # score_rand = bayesian_score(vars,G_rand,D_test)
# # println(score_rand)

# # randplot = gplot(G_rand; nodelabel = nodes)
# # draw(PDF("randplot.pdf", 16cm, 16cm), randplot)

# # Test with Ed example
# G_ed = SimpleDiGraph(3)
# add_edge!(G_ed,1,3)
# add_edge!(G_ed,2,3)
# edplot = gplot(G_ed; nodelabel = [:A, :B, :C])
# draw(PDF("edplot.pdf", 16cm, 16cm), edplot)

# vars_ed = [Variable(:A,2), Variable(:B,2), Variable(:C,3)]
# # println(vars_ed)
# D_ed = [1 2 1 2 1 1 1 2 1 2 1; 1 1 1 2 2 1 2 2 2 1 1; 3 1 1 2 3 3 2 1 3 2 1]
# score_ed = bayesian_score(vars_ed, G_ed, D_ed)
# println(score_ed)

# soln_from_ed = -1*loggamma(13) + loggamma(8) + loggamma(5) - loggamma(13) + loggamma(7)
# soln_from_ed = soln_from_ed + loggamma(6) + loggamma(3) - loggamma(7) + loggamma(3)
# soln_from_ed = soln_from_ed + loggamma(3) + loggamma(3) - loggamma(6) + loggamma(3)
# soln_from_ed = soln_from_ed + loggamma(3) - loggamma(5) + loggamma(3) - loggamma(5)
# println(soln_from_ed)
# # Match as expected

#=====================================
  FINDING BEST GRAPH
======================================#

function K2fit(method::K2Search, vars, D, parent_lim::Int = 2)
    G = SimpleDiGraph(length(vars)) # Instantiate a graph with nodes for vars
    for (k,i) in enumerate(method.ordering[2:end]) # Iterate through nodes.
        # Note k = (1, length(ordering) - 1 ) corresponds to i = ordering(2:end)

        y = bayesian_score(vars, G, D) # Starting Bayes Score for assessing parents for node i

        while true
            y_best, j_best = -Inf, 0 # Initialize new best Bayes score, parent index

            for j in method.ordering[1:k] # Loop through allowable (previous) parents
                # Evalute each edge from parent j to current node i, and save best
                if !has_edge(G, j, i) # Verify edge doesn't already exist

                    add_edge!(G,j,i)
                    y_new = bayesian_score(vars, G, D) # Add and eval edge

                    if y_new > y_best # If score improvement
                        y_best, j_best = y_new, j #Update best score, parent index
                    end

                    rem_edge!(G, j, i) # Remove and test others
                end
            end
            # Now, have best possible parent

            # FIRST, enforce parent limit
            if ( (length(inneighbors(G,i))) >= parent_lim )
                break
            end

            # If within limit, decide to keep this edge
            if y_best > y # If improvement over prior score
                y = y_best # Update current score
                add_edge!(G, j_best, i) # Actually keep that edge

                # 'while true' will lead to checking of additional parents (from allowable)

            else # NO IMPROVEMENT, so assess next node
                break
            end

        end
        #println("Node $i is done")
        # Repeat for all nodes

    end

    return G # Return final graph
end




#=====================================
  MAIN COMPUTATION
======================================#
function compute(infile, outfile)

    # WRITE YOUR CODE HERE
    # FEEL FREE TO CHANGE ANYTHING ANYWHERE IN THE CODE
    # THIS INCLUDES CHANGING THE FUNCTION NAMES, MAKING THE CODE MODULAR, BASICALLY ANYTHING

    #===== K2 SEARCH with most basic ordering =====#
    # K2 search
    vars, D = extract_data(infile)
    ordering = collect(1:length(vars))
    search = K2Search(ordering)

    println("About to search")
    @time begin
        G = K2fit(search, vars, D, length(vars) + 1)
    end

    # Plot
    nodes = [vars[i].name for i in 1:length(vars)]
    plotname, _ = splitext(split(outfile, "/")[end]) # Parse out data file name
    plotname = plotname * "_plot.pdf"

    #plot = gplot(G; nodelabel = nodes)
    #draw(PDF(plotname), plot)
    visualize_graph = plot( graphplot(G; names = nodes), size = (1100,850) )
    #visualize_graph = graphplot(G; names = nodes)
    savefig(visualize_graph,plotname)

    # compute and print score
    print("Score = ")
    println(bayesian_score(vars,G,D))

    # Write to file
    indices = Dict(i => variable for (i, variable) in enumerate(nodes))
    write_gph(G, indices, outfile)

end

if length(ARGS) != 2
    error("usage: julia project1.jl <infile>.csv <outfile>.gph")
end

inputfilename = ARGS[1]
outputfilename = ARGS[2]

compute(inputfilename, outputfilename)