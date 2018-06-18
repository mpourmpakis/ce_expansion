#!/usr/bin/env python
import ce_shuffle
import structure_gen
import time
from itertools import combinations

#shell_nums is a list with the shell sizes
def structure(shape, shell_nums):
    atoms_list=[]
    for s in shell_nums:
        if shape == 'Icosahedron':
            shell_args = [s]
        elif shape in ['Decahedron','Lattice','Sphere']:
            shell_args = [s] * 3

        atoms_list.append(structure_gen.structuregen(shape, shell_args))
        #last_atom_obj = atoms_list[-1]
        #last_atom_obj.shape = shape
        atoms_list[-1].shape=shape
        atoms_list[-1].shell_num=s

    return atoms_list
 


def randomizer(atoms):   
    """
            replaces generic structure with selected atomic symbols; randomizes; prints CE into excel file
    """


    t = time.clock()
    print 'start = ' + str(t)
    ce_shuffle.atomchanger(atoms, 'Cu', 'Ag', atoms.shape , atoms.shell_num)
    t2 = time.clock()
    print 'I1= ' + str(t2 - t)
    ce_shuffle.atomchanger(atoms, 'Cu', 'Au', atoms.shape , atoms.shell_num)
    t2 = time.clock()
    print 'I2= ' + str(t2 - t)
    ce_shuffle.atomchanger(atoms, 'Au', 'Ag', atoms.shape , atoms.shell_num)
    t2 = time.clock()
    print 'I3 = ' + str(t2 - t)
    

def main():
    """
        runs the selected structure 
    """
    atoms_list=structure("Icosahedron", [2,3])
    for atom in atoms_list:
       randomizer(atom)

    print 'Done'


main()