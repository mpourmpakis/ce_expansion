import numpy as np

import ce_expansion.npdb.db_inter
import ce_expansion.plots
from ce_expansion.atomgraph import atomgraph


def build_atomgraph(bimetallic_result):
    """
    Returns an atomgraph object from the result of a bimetallic result query.

    :param bimetallic_result: the result of a bimetallic result query

    :return: an atomgraph object
    """
    bondlist = bimetallic_result.nanoparticle.bonds_list
    assert bondlist is not None
    return atomgraph.AtomGraph(bondlist, kind0=bimetallic_result.metal1, kind1=bimetallic_result.metal2)


systems = ["AgCu" , "AgAu", "AuCu"]
system_colors = ["#FF5555" , "#55FF55", "#5555FF"]

morphologies = ["icosahedron", "cuboctahedron", "elongated-pentagonal-bipyramid"]
morphology_markers = ["h", "s", "^"]

sizes = [13, 55, 147, 309, 561, 923, 1415, 2057, 2869, 3871]
min_pointsize = 5
max_pointsize = min_pointsize + len(sizes) + 30
point_sizes = np.linspace(min_pointsize, max_pointsize, len(sizes))

# Passed to plot function
bondcounts = []
labels = []
colors = []
markers = []
marker_sizes = []

# Atomgraphs for speedup
graphs = {"icosahedron":{},
          "cuboctahedron":{},
          "elongated-pentagonal-bipyramid":{}}

for system, color in zip(systems, system_colors):
    print(system)
    for morph, marker in zip(morphologies, morphology_markers):
        print(morph)
        labelmap = {"icosahedron": "Ico",
                    "cuboctahedron": "Cuboct",
                    "elongated-pentagonal-bipyramid": "EPB"}
        label = "_".join([system, labelmap[morph]])
        for size, pointsize in zip(sizes, point_sizes):
            print(size)

            # Database query for this alloy/morphology/size
            testcases = ce_expansion.npdb.db_inter.get_bimet_result(metals=system, shape=morph, num_atoms=size)
            if len(testcases) == 0:
                # Check that we actually returned results from the query
                continue

            # Bondcounts
            for count, case in enumerate(testcases):
                if case.nanoparticle.bonds_list is None:
                    case.nanoparticle.load_bonds_list()
                # Check if the graph already exists:
                try:
                    graph = graphs[morph][size]
                except KeyError:
                    graphs[morph][size] = build_atomgraph(case)
                    graph = graphs[morph][size]

                ordering = np.array(list(case.ordering), dtype=int)
                bondcount = graph.countMixing(ordering)

                # Append to lists
                bondcounts.append(bondcount)
                labels.append(label)
                colors.append(color)
                markers.append(marker)
                marker_sizes.append(pointsize)

ce_expansion.plots.plot_bond_types_2D(bondcounts, labels, colors, markers, marker_sizes, projection=["AA", "BB"], verbose=False, scale=True)

# for i in [True, False]:
#     plots.plot_bond_types_2D(systems, system_colors, morphologies, morphology_markers, sizes, point_sizes,
#                              projection=["AA", "AB"], verbose=True, scale=i)
#     plots.plot_bond_types_2D(systems, system_colors, morphologies, morphology_markers, sizes, point_sizes,
#                              projection=["BB", "AB"], verbose=True, scale=i)
#     plots.plot_bond_types_2D(systems, system_colors, morphologies, morphology_markers, sizes, point_sizes,
#                              projection=["AA", "BB"], verbose=True, scale=i)
