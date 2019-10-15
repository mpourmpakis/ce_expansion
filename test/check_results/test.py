#!/usr/bin/env python

import os
import re
import sys

import numpy as np

# Get path to /ce_expansion/
path = os.path.realpath(__file__)
ce_expansionpath = os.sep.join(path.split(os.sep)[:-3] + ["ce_expansion"])
sys.path.append(ce_expansionpath)

import atomgraph

class Job(object):
  def __init__(self, filename):
    # Get bonds_list and chemical ordering
    self.kind0 = None
    self.kind1 = None
    with open(filename, 'r') as inp:
      adjacency_list = []
      num_bonds = 0
      for count, line in enumerate(inp):
          if count == 0:
            num_atoms = int(line.strip())
            self.ordering = np.ones(num_atoms)
          if count < 2:
            continue
          elif re.match("^\s+?$", line):
            continue
          else:
            # Grabs the element type
            atom_type = re.search("(?:^(\s+)?)([A-z][A-z]?)(?=\s)", line)[0]
            matched = False
            if self.kind0 is None:
              matched = True
              self.kind0 = atom_type
              self.ordering[count-2] = 0
            elif self.kind1 is None and self.kind0 != atom_type:
              matched = True
              self.kind1 = atom_type
              self.ordering[count-2] = 1
            else:
              if atom_type == self.kind0:
                matched = True
                self.ordering[count-2] = 0
              else:
                matched = True
                self.ordering[count-2] = 1

            # Regex below matches a comma-separated list of numbers enclosed in [] brackets.
            # This list of numbers is then converted to a list
            bonds = list(map(lambda i: int(i.strip()), re.search("(?<=\[)(((\d+)(,?\s+)?)+)(?=\])", line)[0].split(",")))
            adjacency_list.append(bonds)
            num_bonds += len(bonds)
      self.bonds_list = np.zeros([num_bonds, 2], dtype=int)
      bond_count = 0
      for source_index, bonds in enumerate(adjacency_list):
        for bond in bonds:
          self.bonds_list[bond_count, 0] = source_index
          self.bonds_list[bond_count, 1] = bond
          bond_count += 1

    # Get reference DFT energy
    with open("BC_energies_Clusters_DFT_CE_refs.txt", "r") as inp:
      # Matches on filenames.xyz
      for line in inp:
        if re.search("((?<=\s)\w+\.xyz(?=\s))", line)[0] == filename:
          # Finds the energy
          self.DFT_energy = re.search("(-?\d+\.\d+(?=(\s+eV)))", line)[1]
          break

    # Get reference experimental energy
    with open("BC_energies_Clusters_experimental_CE_refs.txt", "r") as inp:
      for line in inp:
        if re.search("((?<=\s)\w+\.xyz(?=\s))", line)[0] == filename:
          self.exp_energy = re.search("(-?\d+\.\d+(?=(\s+eV)))", line)[0]
          break

  def calculate_energy(self):
    graph = atomgraph.AtomGraph(self.bonds_list, kind0 = self.kind0, kind1 = self.kind1)
    self.energy = graph.getTotalCE(self.ordering)
  
with open("results.csv", "w") as outp:
  outp.write("Cluster,DFT Energy,Experiment,Model\n")
  for filename in os.listdir():
    if ".xyz" in filename:
      job = Job(filename)
      print(filename)
      job.calculate_energy()
      outp.write(",".join([filename[:-4], job.DFT_energy, job.exp_energy, str(job.energy)]) + "\n")

        

