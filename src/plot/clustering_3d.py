#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np

import atomgraph
import npdb.db_inter


def build_atomgraph(bimetallic_result):
    """
    Returns an atomgraph object from the result of a bimetallic result query.

    Args:
        bimetallic_result (BimetallicResults): the result of a bimetallic result query

    Returns:
         an atomgraph object
    """
    bondlist = bimetallic_result.nanoparticle.bonds_list
    assert bondlist is not None
    return atomgraph.AtomGraph(bondlist,
                               kind0=bimetallic_result.metal1,
                               kind1=bimetallic_result.metal2)


def plot_results(atomgraphs, orderings, system, color, marker, size, maxsize, morphology):
    coords = np.stack(list(map(lambda i,j: i.countMixing(j), atomgraphs, orderings)))
    x = coords[:,0]
    x = x / max(x)
    y = coords[:,1]
    y = y / max(y)
    z = coords[:,2]
    z = z / max(z)

    # Do the actual plotting
    if size != maxsize:
        label = ""
    else:
        label = system + "_" + morphology[:4]

    ax.scatter(xs=x, ys=y, zs=z, c=color, s=size, marker=marker, label=label)
    ax.set_xlabel("A-A Bonds")
    ax.set_ylabel("B-B Bonds")
    ax.set_zlabel("A-B Bonds")




if __name__ == "__main__":
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    systems = ["AgCu", "AgAu", "AuCu"]
    colors = ["red", "green", "blue"]

    morphologies = ["icosahedron", "cuboctahedron", "elongated-pentagonal-bipyramid"]
    markers = ["h", "s", "^"]

    sizes = [13, 55, 147, 309, 561, 923, 1415, 2057, 2869, 3871]
    pointsizes = range(1,len(sizes)+1)
    maxsize = max(pointsizes)


    for system, color in zip(systems, colors):
        print(system, color)
        for morph, marker in zip(morphologies, markers):
            print(morph, marker)
            for size, pointsize in zip(sizes, pointsizes):
                print(size, pointsize)
                testcases = npdb.db_inter.get_bimet_result(metals=system, shape=morph, num_atoms=size)
                if len(testcases) == 0:
                    continue
                graphs = [None] * len(testcases)
                orderings = [None] * len(testcases)

                for count, case in enumerate(testcases):
                    if case.nanoparticle.bonds_list is None:
                        case.nanoparticle.load_bonds_list()
                    graphs[count] = build_atomgraph(case)
                    orderings[count] = np.array(list(case.ordering), dtype=int)
                plot_results(list(graphs), orderings, system, color, marker, pointsize, maxsize, morph)
    plt.legend()
    plt.show()

