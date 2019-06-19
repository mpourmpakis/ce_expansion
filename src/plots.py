#!/usr/bin/env python
import operator

import matplotlib.lines
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


def _darken(color):
    """
    Takes a hexidecimal color and makes it a shade darker
    :param color: The hexidecimal color to darken
    :return: A darkened version of the hexidecimal color
    """
    # Get the edge color
    darker = "#"
    hex1 = color[1:3]
    hex2 = color[3:5]
    hex3 = color[5:7]
    for val in [hex1, hex2, hex3]:
        if val == "00":
            darker += "00"
        else:
            x = int(val, base=16)
            x -= int("11", base=16)
            x = str(hex(x))[2:].upper()
            darker += x
    return darker


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
    plt.close()


def plot_bond_types_2D(systems, system_colors,
                       morphologies, morphology_markers,
                       sizes, point_sizes,
                       projection=None,
                       scale=False,
                       verbose=False):
    """

    Plots the bond types in 2D.

    :param systems: Systems to be investigated
    :param system_colors: Corresponding list of colors for those systems
    :param morphologies: Morphologies to be investigated
    :param morphology_markers: Corresponding list of markers for those morphologies
    :param sizes: Sizes to be investigated
    :param point_sizes: Corresponding list of point sizes
    :param projection: The two axes to project down.
    :param scale: Whether to scale the 3D axes to range from 0 to 1
    :param verbose: Whether to verbosely print what the system is doing.
    """
    logger = _verbose_printer(verbose)

    # Figure out which axes are where
    axes = [None, None]
    if "AB" in projection or "BA" in projection:
        axes[1] = "AB"
        if "AA" in projection:
            axes[0] = "AA"
        elif "BB" in projection:
            axes[0] = "BB"
    else:
        if "AA" in projection:
            axes[0] = "AA"
        if "BB" in projection:
            axes[1] = "BB"
    assert None not in axes

    # Initialize the pyplot state machine
    fig, ax = plt.subplots()
    title = axes[1] + " vs " + axes[0] + " Bonds"
    plt.title(title)
    ax.set_xlabel(axes[0] + " Bonds")
    ax.set_ylabel(axes[1] + " Bonds")

    def add_series(atomgraphs, orderings, system, color, marker, size, maxsize, morphology, scale=False):
        """
        Inner function used to repeatedly add series to the Plt state machine in the below for-loop
        """
        # Set up coordinates for the 3D plot
        coords = np.stack(list(map(lambda i, j: i.countMixing(j), atomgraphs, orderings)))
        ax_mappings = {"AA": 0,
                       "BB": 1,
                       "AB": 2}  # the order in which atomgraph.countMixing returns bonds

        if scale:
            nbonds = map(sum,coords)
            scaled = np.stack(list(map(operator.truediv, coords, nbonds)))
            x = scaled[:, ax_mappings[axes[0]]]
            y = scaled[:, ax_mappings[axes[1]]]
        else:
            x = coords[:, ax_mappings[axes[0]]]
            y = coords[:, ax_mappings[axes[1]]]


        # Do the actual plotting
        if size != maxsize:
            label = ""
        else:
            label = system + "_" + morphology[:4]

        ax.scatter(x, y, c=color, edgecolors=_darken(color), s=size, marker=marker, label=label)

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

    # Draw hypothetical-structure lines if set to scale
    legend_lines = []
    legend_lines_labels = []
    if scale:
        mix_color = "#1C2957"
        nomix_color = "#CDB87D"
        linealpha = 1
        linewidth = 1
        linestyle = ":"
        if axes[0] == "AA" and axes[1] == "BB":
            ax.plot([0.0, 1.0], [1.0, 0.0], color=nomix_color, alpha=linealpha, lw=linewidth, linestyle=linestyle)
            ax.plot([0.0, 0.0], [0.0, 1.0], color=mix_color, alpha=linealpha, lw=linewidth, linestyle=linestyle)
            ax.plot([0.0, 1.0], [0.0, 0.0], color=mix_color, alpha=linealpha, lw=linewidth, linestyle=linestyle)
        else:
            ax.plot([0.0, 1.0], [1.0, 0.0], color=mix_color, alpha=linealpha, lw=linewidth, linestyle=linestyle)
            ax.plot([0.0, 0.0], [0.0, 1.0], color=mix_color, alpha=linealpha, lw=linewidth, linestyle=linestyle)
            ax.plot([0.0, 1.0], [0.0, 0.0], color=nomix_color, alpha=linealpha, lw=linewidth, linestyle=linestyle)
        legend_lines.append(matplotlib.lines.Line2D([0], [0], color=nomix_color, lw=linewidth, linestyle=linestyle))
        legend_lines_labels.append("Hypothetical Perfect Non-Mixing")
        legend_lines.append(matplotlib.lines.Line2D([0], [0], color=mix_color, lw=linewidth, linestyle=linestyle))
        legend_lines_labels.append("Hypothetical Perfect Mixing")

    # Create legend
    labels = []
    entries = []
    for system, color in zip(systems, system_colors):
        for morph, marker in zip(morphologies, morphology_markers):
            morphmap = {"icosahedron": "Ico",
                        "cuboctahedron": "Cuboct",
                        "elongated-pentagonal-bipyramid": "EPB"}
            label = system + "_" + morphmap[morph]
            entry = matplotlib.lines.Line2D([0], [0], lw=0, color=_darken(color), marker=marker)
            labels.append(label)
            entries.append(entry)
    for line, label in zip(legend_lines, legend_lines_labels):
        entries.append(line)
        labels.append(label)

    plt.legend(entries, labels)

    # Layout and save
    plt.tight_layout()
    if scale:
        title += " (Normalized)"
    plt.savefig(title + ".png", dpi=600)
    plt.close()
