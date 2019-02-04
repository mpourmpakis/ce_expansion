#!/usr/bin/env python3

import ase

from . import atomgraph


def randomizer(ordering):
    """
    Randomizes the structure with some selected atomic symbols.
    """
    return 1


def fileprint():
    """
    Prints information to file.
    """

    return 1


def boundce():
    """
    Returns a summary of the current run.
    """

    return 1

# Example: Check an 147-atom and 309-atom icosahedral nanoparticle with Cu-Ag, Ag-Cu, and Au-Ag alloys.
# Naively randomize 1,000 times for each ordering.

# First, we generate our skeletons
smallest_NP = 2
largest_NP = 3
skeletons = []
for size in range(smallest_NP, largest_NP+1):
    skeletons.append(ase.cluster.Icosahedron("Cu", size))

# Next, we generate graphs for these skeletons.
graphs = []
for skeleton in skeletons:
    graph = atomgraph.buildAdjacencyList(skeleton)