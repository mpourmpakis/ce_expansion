#!/usr/bin/env python

import ctypes
import os
import sys

import ase.cluster
import ase.io
import numpy as np

# Set up the path to src so we can import the relevant files
path = os.path.realpath(__file__)
srcpath = os.sep.join(path.split(os.sep)[:-3] + ["src"])
sys.path.append(srcpath)

import atomgraph
from atomgraph import adjacency


def deterministic_sequence(length):
  """
  Returns a pattern of 1's and 0's to test the NP. Pattern takes the form:
    010100101001010...
    f(n) = (n mod 5) mod 2

  Args:
  length(int) : How long the sequence should be

  Returns:
  A numpy array containing the sequence with type c_long
  """
  result = np.zeros(length, dtype=ctypes.c_long)
  for i in range(0, length):
    result[i] = i%5%2
  return result

with open("results.csv", "w") as outp:
  outp.write("Layers,N_Atoms,Energy (ev)\n")
  for layers in range(2,20):
    print(layers)
    outp.write(str(layers) + ",")
    icos = ase.cluster.Icosahedron("Cu", layers)
    print(icos)
    outp.write(icos.get_chemical_formula() + ",")
    bondlist = adjacency.buildBondsList(icos)
    # Copper-silver
    graph = atomgraph.AtomGraph(bondlist, "Cu", "Ag")
    ordering = deterministic_sequence(len(icos))
    energy = graph.getTotalCE(ordering)
    print(energy)
    outp.write(str(energy) + "\n")
    ase.io.write(icos.get_chemical_formula() + ".xyz", icos)
    
