#!/usr/bin/env python

import os

import ase.lattice.cubic
import ase.neighborlist

from ce_expansion.atomgraph import adjacency, atomgraph
from ce_expansion.ga import ga

# Make a generic FCC cell
for cell_size in range(5, 10):
    pairs = [["Ag", "Au"], ["Ag", "Cu"], ["Au", "Cu"]]
    fcc_cell = ase.lattice.cubic.FaceCenteredCubic(symbol="Cu", size=[cell_size, cell_size, cell_size])
    bonds = adjacency.buildBondsList(fcc_cell)
    for pair in pairs:
        metal1 = pair[0]
        metal2 = pair[1]
        graph = atomgraph.AtomGraph(bonds.copy(), metal1, metal2)

        # Genetic Algorithm
        n_atoms = len(fcc_cell)
        known_comps = [None] * 101
        for composition in range(0, 101):
            percent = composition / 100
            n_metal1 = int(percent * n_atoms)
            if n_metal1 in known_comps:
                continue
            else:
                known_comps[composition] = n_metal1
            n_metal2 = n_atoms - n_metal1
            formula = "".join(map(str, [metal1, n_metal1, metal2, n_metal2]))
            if os.path.exists("cells/" + formula + ".cif"):
                continue
            print("Composition: " + str(composition) + "%, " + formula)
            pop = ga.Pop(fcc_cell,
                         bond_list=bonds.copy(),
                         metals=(metal1, metal2),
                         shape="fcc_cell",
                         n_metal2=n_metal2,
                         atomg=graph)
            pop.run(max_nochange=2000)
            ga.make_file(pop.atom, pop[0], "cells/" + formula + ".cif", filetype="cif")
