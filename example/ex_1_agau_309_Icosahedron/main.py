#!/usr/bin/env python

import os
import sys

import ase.io
import numpy as np

from atomgraph import atomgraph, adjacency

# Tell the program where /ce_expansion/ and /data/ can be found
data = os.path.join(os.path.realpath(__file__), '..', '..', '..', 'data', 'larson_et_al')
sys.path.append(data)

# Calculating the bonding is fairly slow in ASE's libraries.
# Fortunately, every XYZ in the dataset has the same set of coordinates,
# so we can use the same atomgraph for every system.
print("Building atom graph...")
filename = "Ag0Au309.xyz"  # chosen completely arbitrarily
path = os.path.join(data, filename)
atoms = ase.io.read(path)
bondlist = adjacency.buildBondsList(atoms, radius_dictionary={("Ag", "Ag"): 3,
                                                              ("Ag", "Au"): 3,
                                                              ("Au", "Au"): 3})
# Make the atomgraph; set "0" to Ag and set "1" to Au
graph = atomgraph.AtomGraph(bondlist, kind0="Ag", kind1="Au")

# Array to hold mixing parameters and energies
csv_data = [None] * 310

# Now we'll iterate over every xyz file in the directory
print("Calculating mixing parameters and cohesive energies...")
for i in range(0, 310):
    # Specify the path to the file and open it as an ASE_Atoms Object
    filename = "Ag" + str(i) + "Au" + str(309 - i) + ".xyz"
    print(filename[:-4])
    path = os.path.join(data, filename)
    atoms = ase.io.read(path)

    # Make a holder array for the chemical ordering
    ordering = np.zeros(309)

    # Iterate over every atom to get its chemical ordering
    for index, atom in enumerate(atoms):
        assert atom.symbol in ["Ag", "Au"]
        # Recall when we made the atomgraph, we said a "0" is Ag, and a "1" is Au
        if atom.symbol == "Ag":
            ordering[index] = 0
        elif atom.symbol == "Au":
            ordering[index] = 1

    # Calculate the mixing parameter
    mixing_parameter = graph.calcMixing(ordering)

    # Calculate the cohesive energy
    cohesive_energy = graph.getTotalCE(ordering)

    # CSV will have the following columns:
    #   1 - Chemical formula
    #   2 - Number of Ag atoms
    #   3 - Number of Au atoms
    #   4 - Mixing Parameter
    #   5 - BCM_BCM_CE_eV_eV
    #   6 - SE_Energy_eV
    #   7 - BCM_EE_eV

    csv_data[i] = [atoms.get_chemical_formula(),
                   i,
                   309 - i,
                   mixing_parameter,
                   cohesive_energy,
                   atoms.get_total_energy() * -1]

# We need the monometallic cohesive energies to calculate excess energy
for i in csv_data:
    if i[1] == 0:
        mono_au = i[4]
    elif i[1] == 309:
        mono_ag = i[4]

# Calculate excess energy
for entry in csv_data:
    ag_count = entry[1]
    au_count = entry[2]
    cohesive_energy = entry[4]

    excess_energy = cohesive_energy - mono_ag * (ag_count / 309) - mono_au * (au_count / 309)

    if abs(excess_energy) < 1E-10:
        excess_energy = 0
    entry.append(excess_energy)

# Write to file and call it a day
print("Writing to File mixing_parameter_data.csv")
with open("mixing_parameter_data.csv", "w") as outp:
    outp.write("Chemical_Formula,Ag,Au,Mixing_Parameter,BCM_CE_eV,SE_Energy_eV,BCM_EE_eV\n")
    for entry in csv_data:
        outp.write(",".join(map(str, entry)) + "\n")
