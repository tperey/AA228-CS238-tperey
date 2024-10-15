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