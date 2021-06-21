from ce_expansion.npdb import db_inter
from pprint import pprint
from ce_expansion.atomgraph.bcm import BCModel
import numpy as np
import os
from pprint import pprint

import ase.units as units

this_dir = os.path.dirname(os.path.abspath(__file__))

res = db_inter.get_bimet_result('AuCu', 'icosahedron', 55, n_metal1=22)

atoms = res.atoms_obj
bond_list = res.nanoparticle.bonds_list
metal_types = ['Cu', 'Au', "Ag"]

import ase.visualize
#ase.visualize.view(atoms)

# STEP 1
# create a BCModel object


#TODO: account for this case in bcm
#metal_types = ['Cu', 'Au', 'au', 'ag']

bcm = BCModel(atoms, bond_list, metal_types)
assert bcm.metal_types == ['Ag', 'Au', 'Cu']

print("STEP 1")

# STEP 2
# make sure everything passed in becomes an attribute
print(bcm.atoms)
print(bcm.bond_list.shape)
print(bcm.metal_types)
print("STEP 2")

# STEP 2.5 DONE
# this should print the total number of atoms
print(len(bcm))
print("STEP 2.5")

# STEP 3
# should calculate the following during initialization
print("CE_BULK")
pprint(bcm.ce_bulk)
print("GAMMAS")
pprint(bcm.gammas)
print("STEP 3")

# STEP 4
# calculate precomputed factors array
pprint(bcm.precomps)
print("STEP 4")

# STEP 5
import os
import ase.io as io

ordering1 = np.zeros(len(bcm),int)
ordering2 = np.ones(len(bcm),int)
ordering3 = (atoms.symbols == metal_types[1]).astype(int)
ce1 = bcm.calc_ce(ordering1)
ce2 = bcm.calc_ce(ordering2)
ce3 = bcm.calc_ce(ordering3)
print(ce1)
print(ce2)
print(ce3)
print("STEP 5")

# STEP 6
# calculate excess energy (EE) using the self.ce_calc
# for monometallic NPs, EE = 0
ee1 = bcm.calc_ee(ordering1)
ee2 = bcm.calc_ee(ordering2)
ee3 = bcm.calc_ee(ordering3)

print(ee1)
print(ee2)
print(ee3)
print("STEP 6")

print(bcm.calc_smix(ordering3) , "= smix of ordering3")
print(bcm.calc_gmix(ordering3) , "= gmix of ordering3")
print("STEP 7")

#Below is mutliple tested smix values when given the fractions by the smix_test_data.txt file:


data = np.loadtxt(os.path.join(this_dir, "smix_test_data.txt") , dtype=float)

"""
compositions = np.bincount(ordering) / len(ordering)

kb = ase.units.Kb
smix = -kb * sum(xi * ln(xi) for xi in compositions)
sum(compositions) == 1

Boltzmann constant (look it up on wiki)

smix units = eV / (K * atom)

Returns: entropy of mixing (smix)

"""
comps = []


for row in data:
    comps = row[:-1]
    x_i = comps[comps != 0]
    smix_actual = abs(row[-1])

    smix = -units.kB * sum(x_i * np.log(x_i))
    assert abs(smix - smix_actual) < 1E-8
print('All smix tests passed!')
