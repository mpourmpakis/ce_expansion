#!/usr/bin/env python

import adjacency, atomgraph
import ase.cluster
import numpy as np
import time

atom0 = "Cu"
atom1 = "Ag"

delta_c = [None]*18
c_energies = [None]*18
delta_og = [None]*18
og_energies = [None]*18



def set_offset(size):
  nanoparticle = ase.cluster.Icosahedron("Cu", size)

  bondList = adjacency.buildBondsList(nanoparticle)
  adjList = adjacency.buildAdjacencyList(nanoparticle)

  c_graph = atomgraph.c_AtomGraph(bondList, atom0, atom1)
  og_graph = atomgraph.AtomGraph(adjList, atom0, atom1)


  ordering = np.random.choice([0,1], size=og_graph.n_atoms)

  return (c_graph, og_graph, ordering)
  
