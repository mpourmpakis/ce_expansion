#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np

import atomgraph
import npdb


def _build_atomgraph(bimetallic_result):
    """
    Returns an atomgraph object from the result of a bimetallic result query.

    Args:
        bimetallic_result (BimetallicResults): the result of a bimetallic result query

    Returns:
         an atomgraph object
    """
    bondlist = bimetallic_result.nanoparticle.bonds_list
    assert bondlist is not None
    return atomgraph.AtomGraph(bondlist, kind0=bimetallic_result.metal1, kind1=bimetallic_result.metal2)

def _verbose_printer(verbose):
    if verbose:
        def inner(string):
            print(string)
    else:
        def inner(string):
            pass
    return inner


def plot_bond_types_3D(systems, system_colors,
                       morphologies, morphology_markers,
                       sizes, point_sizes,
                       scale=False,
                       verbose=False):
    """
    Plots the bond types in 3D. A-B versus A-A versus B-B

    :param systems: Systems to be investigated
    :param system_colors: Corresponding list of colors for those systems
    :param morphologies: Morphologies to be investigated
    :param morphology_markers: Corresponding list of markers for those morphologies
    :param sizes: Sizes to be investigated
    :param point_sizes: Corresponding list of point sizes
    :param scale: Whether to scale the 3D axes to range from 0 to 1
    :param verbose: Whether to verbosely print what the system is doing.
    :return:
    """
    logger = _verbose_printer(verbose)

    # Initialize the pyplot state machine
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("A-A Bonds")
    ax.set_ylabel("B-B Bonds")
    ax.set_zlabel("A-B Bonds")

    def add_series(atomgraphs, orderings, system, color, marker, size, maxsize, morphology, scale=False):
        """
        Inner function used to repeatedly add series to the Plt state machine in the below for-loop
        """
        # Set up coordinates for the 3D plot
        coords = np.stack(list(map(lambda i, j: i.countMixing(j), atomgraphs, orderings)))
        x = coords[:, 0]
        y = coords[:, 1]
        z = coords[:, 2]
        if scale:
            x = x / max(x)
            y = y / max(y)
            z = z / max(z)

        # Do the actual plotting
        if size != maxsize:
            label = ""
        else:
            label = system + "_" + morphology[:4]

        ax.scatter(xs=x, ys=y, zs=z, c=color, s=size, marker=marker, label=label)

    # Scales point sizes
    maxsize = max(point_sizes)

    for system, color in zip(systems, system_colors):
        logger(system)
        for morph, marker in zip(morphologies, morphology_markers):
            logger(morph)
            for size, pointsize in zip(sizes, point_sizes):
                logger(size)
                # Database query for this alloy/morphology/size
                testcases = npdb.db_inter.get_bimet_result(metals=system, shape=morph, num_atoms=size)
                if len(testcases) == 0:
                    # Check that we actually returned results from the query
                    continue
                # Construct the atomgraphs and their chemical orderings
                graphs = [None] * len(testcases)
                orderings = [None] * len(testcases)
                for count, case in enumerate(testcases):
                    if case.nanoparticle.bonds_list is None:
                        case.nanoparticle.load_bonds_list()
                    graphs[count] = _build_atomgraph(case)
                    orderings[count] = np.array(list(case.ordering), dtype=int)
                # Add the series to the 3D plot
                add_series(list(graphs), orderings, system, color, marker, pointsize, maxsize, morph, scale)

    plt.legend()
    plt.show()
