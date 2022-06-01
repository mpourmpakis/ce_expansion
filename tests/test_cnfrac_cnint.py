import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from ce_expansion.atomgraph.bcm import BCModel
from ce_expansion.atomgraph.adjacency import build_bonds_arr

from ase.io import read
import ase

import ase.neighborlist

import pytest

def get_ordering(atoms):
    atom_types = list(np.unique(atoms.symbols))
    atom_types.sort()
    order = []
    for sym in atoms.symbols:
        order.append(atom_types.index(sym))
    return np.array(order)

def get_cutoffs(atoms,x):
    """Custom Cutoffs from custom Radii"""
    radii = {'Au':1.47*x,
             'Pd':1.38*x,
             'Pt':1.38*x}
    return [radii[atom_type] for atom_type in atoms.symbols]

def make_bcm(atoms,x=1.200,CN_Method = 'int'):
    # radii = ase.neighborlist.natural_cutoffs(atoms,x)
    radii = get_cutoffs(atoms,x)
    bonds = build_bonds_arr(atoms,radii)
    bcm = BCModel(atoms,bond_list=bonds,CN_Method=CN_Method)
    return bcm

def calc_ce1(atoms,gammas=False,ce_bulk=False,CN_Method='int'):
    # Getting integer CNs
    bcm_1 = make_bcm(atoms,x=1.20)
    cns = bcm_1.cn

    # Radii calculated from Average Distance / 2 between atoms in a 147 icosahedron (PBE-D3)
    radius = {'Au':1.47,'Pd':1.38,'Pt':1.38} 

    # Getting NP average radius
    AVERAGE_RADS = []
    for i, CN in enumerate(cns):
        AVERAGE_RADS.append(radius[atoms.symbols[i]])
    AVERAGE_RAD = np.mean(AVERAGE_RADS)
    

    cn_frac = []
    for i,cn in enumerate(cns):
        atom_type = atoms.symbols[i]
        if CN_Method == 'frac':
            cn_frac.append(cn * (radius[atom_type]/AVERAGE_RAD))
        else:
            cn_frac.append(cn)

    

    if ce_bulk==False:
        ce_bulk = bcm_1.ce_bulk
    if gammas == False:
        gammas = bcm_1.gammas
    
    num_sum = 0
    for i,j in bcm_1.bond_list:
        A = atoms.symbols[i]
        B = atoms.symbols[j]
        part_1 = gammas[A][B]*(ce_bulk[A]/cn_frac[i])*np.sqrt(cn_frac[i]/12) 
        part_2 = gammas[B][A]*(ce_bulk[B]/cn_frac[j])*np.sqrt(cn_frac[j]/12) 
        num_sum += (part_1+part_2) 
    return num_sum/(len(atoms)*2) 

# Checking the int method 
def test_check_bcm_int_calc_ce():
    atoms = read('tests/Test_Data/Pt73Pd74.xyz')
    BCM = make_bcm(atoms,x=1.200,CN_Method = 'int')
    order = get_ordering(atoms)
    assert np.isclose(calc_ce1(atoms,CN_Method='int'),BCM.calc_ce(order))

def test_check_bcm_frac_calc_ce():
    atoms = read('tests/Test_Data/Pt73Pd74.xyz')
    BCM = make_bcm(atoms,x=1.200,CN_Method = 'frac')
    order = get_ordering(atoms)
    assert np.isclose(calc_ce1(atoms,CN_Method='frac'),BCM.calc_ce(order))