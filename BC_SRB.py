#!/usr/bin/env python
# Code for calculating the cohesive energy (CE), using the BC/SRB models, of Arbitrary Metal Nanoparticles (MNP)
# For details surrounding the module see 10.1021/acs.nanolett.8b00670

# Original version published in:
#     Yan, Z.; Taylor, M. G.; Mascareno, A.; Mpourmpakis, G.
#     "Size-, Shape-, and Composition-Dependent Model for Metal Nanoparticle Stability Prediction"
#     Nano Lett 2018, 18 (4), 2696-2704
# Minor modifications made for readability and interoperability with our new code

import numpy as np
import sys
import csv
import ase
import ase.neighborlist
import math
from ase.io import read, write
import ase.calculators.calculator


class SRB_Model(ase.calculators.calculator.Calculator):


    # We can't relax every structure, so we just take the known Cu-Cu bond length of 2.55, and round up to 2.6 for every
    # structure.
    vdw_radius=2.6

    implemented_properties = ["energy"]

    def __init__(self, **kwargs):
        ase.calculators.calculator.Calculator.__init__(self, **kwargs)

    def calculate(self, atoms=None, properties=['energy'], system_changes=None):
        """
        Returns the cohesive energy as calculated by:

        """
        ase.calculators.calculator.Calculator.calculate(self, atoms, properties, system_changes)
        atoms = ase.Atoms()

        # Calculate coordination numbers
        cnList = np.bincount(ase.neighborlist.neighbor_list("i",atoms,self.vdw_radius))

        # Calculate the SRB model energy

        sys.exit()

        atomicSymbols = list(set(atoms.get_chemical_symbols()))


for item in range(1, len(sys.argv)):
    moleculename = sys.argv[item]
original = read(moleculename)
atomicsymbols = original.get_chemical_symbols()
atom_types = list(set(atomicsymbols))
file1 = open(moleculename, 'r')
# For ease of processing neighbors load all the lines into memory
lines = file1.readlines()
file1.close()
count = 0
BE = 0
for line in lines:
    if
count <= 1:
count += 1










else:
vals = line.split()
# Isolate CN value of atom i from modified .xyz file
CN_i = float(vals[4])
name_i = vals[0]
BCE_i = BCE[name_i]
t_metal_i = t_metals[name_i]  # Pull out transition metal reference number
i = count - 2
# Isolate neighborlist for each atom from modified .xyz file
vect = line.split("[")
nl_i_raw = vect[1][0:-2]
nl_i_raw_2 = nl_i_raw.split(",")
nl_i = [int(i) for i in nl_i_raw_2]
# Iterate over all nieghbors of i to evaluate the BC model.
for j in nl_i:  # j is a neighbor of i
    k = j + 2
line_j = lines[k]
vals_j = line_j.split()
CN_j = float(vals_j[4])
name_j = vals_j[0]
BCE_j = BCE[name_j]
t_metal_j = t_metals[name_j]
t = WF_1[t_metal_i][t_metal_j]
if t != 0:  # reference 1 is in priority since it's experimental value
    AB = t
AA = WF_1[t_metal_i][t_metal_i]
BB = WF_1[t_metal_j][t_metal_j]
else:  # Switch to reference 2 if reference 1 doesn't have the data
AB = WF_2[t_metal_i][t_metal_j]
AA = WF_2[t_metal_i][t_metal_i]
BB = WF_2[t_metal_j][t_metal_j]
if AA == BB:  # i and j are same metal, SRB model
    bl_i = 1
bl_j = 1
else:  # i and j are different metals, BC model
# Calculate the gamma, weight factors
bl_i = 2(AB - BB) / (AA - BB)
bl_j = 2 - bl_i
# Apply the BC/SRB model to add up over all the bonds
BE += (((BCE_i * (CN_i ** 0.5) / (12 ** 0.5)) / CN_i)) * (bl_i / float(bl_i + bl_j)) + ((BCE_j * (
    CN_j ** 0.5) / (12 ** 0.5)) / CN_j) * (bl_j / float(bl_i + bl_j))
count += 1
# Print out the evaluated CE of the system.
print 'The CE = {} eV for the {} system'.format(BE / (count - 2), moleculename)
