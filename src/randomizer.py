#!/usr/bin/env python3
# TODO: Clean this up; remove repetition. Remove repetition.

from __future__ import absolute_import
import structure_gen
import ce_shuffle
import time
import itertools


def i_structure():
    """
        pre-defined structural parameters ; returns atoms objects 
    """
    atoms_i2 = structure_gen.structuregen("Icosahedron", [2])
    atoms_i3 = structure_gen.structuregen("Icosahedron", [3])
    atoms_i4 = structure_gen.structuregen("Icosahedron", [4])
    atoms_i5 = structure_gen.structuregen("Icosahedron", [5])
    return atoms_i2, atoms_i3, atoms_i4, atoms_i5


def d_structure():
    """
        pre-defined structural parameters ; returns atoms objects
    """
    atoms_d2 = structure_gen.structuregen("Decahedron", [2, 2, 2])
    atoms_d3 = structure_gen.structuregen("Decahedron", [3, 3, 3])
    atoms_d4 = structure_gen.structuregen("Decahedron", [4, 4, 4])
    atoms_d5 = structure_gen.structuregen("Decahedron", [5, 5, 5])
    return atoms_d2, atoms_d3, atoms_d4, atoms_d5


def c_structure():
    """
        pre-defined structural parameters ; returns atoms objects
    """
    atoms_c2 = structure_gen.structuregen("Lattice", [2, 2, 2])
    atoms_c3 = structure_gen.structuregen("Lattice", [3, 3, 3])
    atoms_c4 = structure_gen.structuregen("Lattice", [4, 4, 4])
    atoms_c5 = structure_gen.structuregen("Lattice", [5, 5, 5])
    return atoms_c2, atoms_c3, atoms_c4, atoms_c5


def s_structure():
    """
        pre-defined structural parameters ; returns atoms objects
    """
    atoms_s2 = structure_gen.structuregen("Lattice", [2, 2, 2])
    atoms_s3 = structure_gen.structuregen("Lattice", [3, 3, 3])
    atoms_s4 = structure_gen.structuregen("Lattice", [4, 4, 4])
    atoms_s5 = structure_gen.structuregen("Lattice", [5, 5, 5])
    return atoms_s2, atoms_s3, atoms_s4, atoms_s5


def icosa(atoms_i2, atoms_i3, atoms_i4, atoms_i5):
    """
        replaces generic structure with selected atomic symbols; randomizes; prints CE into excel file
    """
    t = time.clock()
    icosalist = [atoms_i2, atoms_i3, atoms_i4, atoms_i5]
    timei = 1
    metals = ['Au', 'Cu', 'Ag']
    for i in range(2, 6):
        for a1, a2 in itertools.combinations(metals, 2):
            ce_shuffle.atomchanger(icosalist[i - 2], a1, a2, 'Icosahedron', i)
            t2 = time.clock()
            print('I' + str(timei) + ' = ' + str(t2 - t))
            timei += 1
    return t2


def deca(atoms_d2, atoms_d3, atoms_d4, atoms_d5, t):
    """
        replaces generic structure with selected atomic symbols; randomizes; prints CE into excel file
    """
    decalist = [atoms_d2, atoms_d3, atoms_d4, atoms_d5]
    timei = 1
    metals = ['Au', 'Cu', 'Ag']
    for i in range(2, 6):
        for a1, a2 in itertools.combinations(metals, 2):
            ce_shuffle.atomchanger(decalist[i - 2], a1, a2, 'Decahedron', i)
            t2 = time.clock()
            print('D' + str(timei) + ' = ' + str(t2 - t))
            timei += 1


def cubic(atoms_c2, atoms_c3, atoms_c4, atoms_c5, t):
    """
        replaces generic structure with selected atomic symbols; randomizes; prints CE into excel file
    """

    cubelist = [atoms_c2, atoms_c3, atoms_c4, atoms_c5]
    timei = 1
    metals = ['Au', 'Cu', 'Ag']
    for i in range(2, 6):
        for a1, a2 in itertools.combinations(metals, 2):
            ce_shuffle.atomchanger(cubelist[i - 2], a1, a2, 'Cube', i)
            t2 = time.clock()
            print('C' + str(timei) + ' = ' + str(t2 - t))
            timei += 1
    return t2


def sphere(atoms_s2, atoms_s3, atoms_s4, atoms_s5, t):
    """
        replaces generic structure with selected atomic symbols; randomizes; prints CE into excel file
    """
    spherelist = [atoms_s2, atoms_s3, atoms_s4, atoms_s5]
    timei = 1
    metals = ['Au', 'Cu', 'Ag']
    for i in range(2, 6):
        for a1, a2 in itertools.combinations(metals, 2):
            ce_shuffle.atomchanger(spherelist[i - 2], a1, a2, 'Sphere', i)
            t2 = time.clock()
            print('S' + str(timei) + ' = ' + str(t2 - t))
            timei += 1
    return t2


def main():
    """
        runs the selected structure 
    """
    [c2, c3, c4, c5] = c_structure()
    [i2, i3, i4, i5] = i_structure()
    [d2, d3, d4, d5] = d_structure()
    [s2, s3, s4, s5] = s_structure()

    t2 = icosa(i2, i3, i4, i5)
    t3 = cubic(c2, c3, c4, c5, t2)
    t4 = sphere(s2, s3, s4, s5, t3)
    deca(d2, d3, d4, d5, t4)

    print('Done')


main()
