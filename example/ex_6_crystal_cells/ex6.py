#!/usr/bin/env python

import ase.lattice.cubic
import ase.neighborlist

import adjacency
import atomgraph
import ga

# Make a generic FCC cell
cell_size = 1
fcc_cell = ase.lattice.cubic.FaceCenteredCubic(symbol="Cu", size=[cell_size, cell_size, cell_size])
bonds = adjacency.buildBondsList(fcc_cell)
metal1 = "Au"
metal2 = "Ag"
graph = atomgraph.AtomGraph(bonds.copy(), metal1, metal2)

# Genetic Algorithm
n_atoms = len(fcc_cell)
known_comps = [None] * 101
for composition in range(0,101,1):
    percent = composition / 100
    n_metal1 = int(percent * n_atoms)
    if n_metal1 in known_comps:
        continue
    else:
        known_comps[composition] = n_metal1
    n_metal2 = n_atoms - n_metal1
    formula = f"{metal1}{n_metal1}{metal2}{n_metal2}"
    print(f"Composition: {composition}%, {formula}")
    pop = ga.Pop(fcc_cell,
                 bond_list=bonds.copy(),
                 metals=(metal1, metal2),
                 shape="fcc_cell",
                 n_metal2=n_metal2,
                 atomg=graph)
    pop.run(max_nochange=1)
    ga.make_file(pop.atom, pop[0], f"cells/{formula}.cif", filetype="cif")
