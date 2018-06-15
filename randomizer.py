#!/usr/bin/env python
import ce_shuffle
import structure_gen
import time


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
    print 'start = ' + str(t)
    ce_shuffle.atomchanger(atoms_i2, 'Cu', 'Ag', 'Icosahedron', 2)
    t2 = time.clock()
    print 'I1= ' + str(t2 - t)
    ce_shuffle.atomchanger(atoms_i2, 'Cu', 'Au', 'Icosahedron', 2)
    t2 = time.clock()
    print 'I2= ' + str(t2 - t)
    ce_shuffle.atomchanger(atoms_i2, 'Au', 'Ag', 'Icosahedron', 2)
    t2 = time.clock()
    print 'I3 = ' + str(t2 - t)

    ce_shuffle.atomchanger(atoms_i3, 'Cu', 'Ag', 'Icosahedron', 3)
    t2 = time.clock()
    print 'I4 = ' + str(t2 - t)
    ce_shuffle.atomchanger(atoms_i3, 'Cu', 'Au', 'Icosahedron', 3)
    t2 = time.clock()
    print 'I5 = ' + str(t2 - t)
    ce_shuffle.atomchanger(atoms_i3, 'Au', 'Ag', 'Icosahedron', 3)
    t2 = time.clock()
    print 'I6 = ' + str(t2 - t)

    ce_shuffle.atomchanger(atoms_i4, 'Cu', 'Ag', 'Icosahedron', 4)
    t2 = time.clock()
    print 'I7 = ' + str(t2 - t)
    ce_shuffle.atomchanger(atoms_i4, 'Cu', 'Au', 'Icosahedron', 4)
    t2 = time.clock()
    print 'I8 = ' + str(t2 - t)
    ce_shuffle.atomchanger(atoms_i4, 'Au', 'Ag', 'Icosahedron', 4)
    t2 = time.clock()
    print 'I9 = ' + str(t2 - t)

    ce_shuffle.atomchanger(atoms_i5, 'Cu', 'Ag', 'Icosahedron', 5)
    t2 = time.clock()
    print 'I10 = ' + str(t2 - t)
    ce_shuffle.atomchanger(atoms_i5, 'Cu', 'Au', 'Icosahedron', 5)
    t2 = time.clock()
    print 'I11 = ' + str(t2 - t)
    ce_shuffle.atomchanger(atoms_i5, 'Au', 'Ag', 'Icosahedron', 5)
    t2 = time.clock()
    print 'I12 = ' + str(t2 - t)
    return t2


def deca(atoms_d2, atoms_d3, atoms_d4, atoms_d5, t):
    """
        replaces generic structure with selected atomic symbols; randomizes; prints CE into excel file
    """
    ce_shuffle.atomchanger(atoms_d2, 'Cu', 'Ag', 'Decahedron', 2)
    t2 = time.clock()
    print 'D1 = ' + str(t2 - t)
    ce_shuffle.atomchanger(atoms_d2, 'Cu', 'Au', 'Decahedron', 2)
    t2 = time.clock()
    print 'D2 = ' + str(t2 - t)
    ce_shuffle.atomchanger(atoms_d2, 'Au', 'Ag', 'Decahedron', 2)
    t2 = time.clock()
    print 'D3 = ' + str(t2 - t)

    ce_shuffle.atomchanger(atoms_d3, 'Cu', 'Ag', 'Decahedron', 3)
    t2 = time.clock()
    print 'D4 = ' + str(t2 - t)
    ce_shuffle.atomchanger(atoms_d3, 'Cu', 'Au', 'Decahedron', 3)
    t2 = time.clock()
    print 'D5 = ' + str(t2 - t)
    ce_shuffle.atomchanger(atoms_d3, 'Au', 'Ag', 'Decahedron', 3)
    t2 = time.clock()
    print 'D6 = ' + str(t2 - t)

    ce_shuffle.atomchanger(atoms_d4, 'Cu', 'Ag', 'Decahedron', 4)
    t2 = time.clock()
    print 'D7 = ' + str(t2 - t)
    ce_shuffle.atomchanger(atoms_d4, 'Cu', 'Au', 'Decahedron', 4)
    t2 = time.clock()
    print 'D8 = ' + str(t2 - t)
    ce_shuffle.atomchanger(atoms_d4, 'Au', 'Ag', 'Decahedron', 4)
    t2 = time.clock()
    print 'D9 = ' + str(t2 - t)

    ce_shuffle.atomchanger(atoms_d5, 'Cu', 'Ag', 'Decahedron', 5)
    t2 = time.clock()
    print 'D10 = ' + str(t2 - t)
    ce_shuffle.atomchanger(atoms_d5, 'Cu', 'Au', 'Decahedron', 5)
    t2 = time.clock()
    print 'D11 = ' + str(t2 - t)
    ce_shuffle.atomchanger(atoms_d5, 'Au', 'Ag', 'Decahedron', 5)
    t2 = time.clock()
    print 'D12 = ' + str(t2 - t)


def cubic(atoms_c2, atoms_c3, atoms_c4, atoms_c5, t):
    """
        replaces generic structure with selected atomic symbols; randomizes; prints CE into excel file
    """
    ce_shuffle.atomchanger(atoms_c2, 'Cu', 'Ag', 'Cube', 2)
    t2 = time.clock()
    print 'C1 = ' + str(t2 - t)
    ce_shuffle.atomchanger(atoms_c2, 'Cu', 'Au', 'Cube', 2)
    t2 = time.clock()
    print 'C2 = ' + str(t2 - t)
    ce_shuffle.atomchanger(atoms_c2, 'Au', 'Ag', 'Cube', 2)
    t2 = time.clock()
    print 'C3 = ' + str(t2 - t)

    ce_shuffle.atomchanger(atoms_c3, 'Cu', 'Ag', 'Cube', 3)
    t2 = time.clock()
    print 'C4 = ' + str(t2 - t)
    ce_shuffle.atomchanger(atoms_c3, 'Cu', 'Au', 'Cube', 3)
    t2 = time.clock()
    print 'C5 = ' + str(t2 - t)
    ce_shuffle.atomchanger(atoms_c3, 'Au', 'Ag', 'Cube', 3)
    t2 = time.clock()
    print 'C6 = ' + str(t2 - t)

    ce_shuffle.atomchanger(atoms_c4, 'Cu', 'Ag', 'Cube', 4)
    t2 = time.clock()
    print 'C7 = ' + str(t2 - t)
    ce_shuffle.atomchanger(atoms_c4, 'Cu', 'Au', 'Cube', 4)
    t2 = time.clock()
    print 'C8 = ' + str(t2 - t)
    ce_shuffle.atomchanger(atoms_c4, 'Au', 'Ag', 'Cube', 4)
    t2 = time.clock()
    print 'C9 = ' + str(t2 - t)
    ce_shuffle.atomchanger(atoms_c5, 'Cu', 'Ag', 'Cube', 5)
    t2 = time.clock()
    print 'C10 = ' + str(t2 - t)
    ce_shuffle.atomchanger(atoms_c5, 'Cu', 'Au', 'Cube', 5)
    t2 = time.clock()
    print 'C11 = ' + str(t2 - t)
    ce_shuffle.atomchanger(atoms_c5, 'Au', 'Ag', 'Cube', 5)
    t2 = time.clock()
    print 'C12 = ' + str(t2 - t)
    return t2

def sphere(atoms_s2, atoms_s3, atoms_s4, atoms_s5, t):
    """
        replaces generic structure with selected atomic symbols; randomizes; prints CE into excel file
    """
    ce_shuffle.atomchanger(atoms_s2, 'Cu', 'Ag', 'Sphere', 2)
    t2 = time.clock()
    print 'S1 = ' + str(t2 - t)
    ce_shuffle.atomchanger(atoms_s2, 'Cu', 'Au', 'Sphere', 2)
    t2 = time.clock()
    print 'S2 = ' + str(t2 - t)
    ce_shuffle.atomchanger(atoms_s2, 'Au', 'Ag', 'Sphere', 2)
    t2 = time.clock()
    print 'S3 = ' + str(t2 - t)

    ce_shuffle.atomchanger(atoms_s3, 'Cu', 'Ag', 'Sphere', 3)
    t2 = time.clock()
    print 'S4 = ' + str(t2 - t)
    ce_shuffle.atomchanger(atoms_s3, 'Cu', 'Au', 'Sphere', 3)
    t2 = time.clock()
    print 'S5 = ' + str(t2 - t)
    ce_shuffle.atomchanger(atoms_s3, 'Au', 'Ag', 'Sphere', 3)
    t2 = time.clock()
    print 'S6 = ' + str(t2 - t)

    ce_shuffle.atomchanger(atoms_s4, 'Cu', 'Ag', 'Sphere', 4)
    t2 = time.clock()
    print 'S7 = ' + str(t2 - t)
    ce_shuffle.atomchanger(atoms_s4, 'Cu', 'Au', 'Sphere', 4)
    t2 = time.clock()
    print 'S8 = ' + str(t2 - t)
    ce_shuffle.atomchanger(atoms_s4, 'Au', 'Ag', 'Sphere', 4)
    t2 = time.clock()
    print 'S9 = ' + str(t2 - t)

    ce_shuffle.atomchanger(atoms_s5, 'Cu', 'Ag', 'Sphere', 5)
    t2 = time.clock()
    print 'S10 = ' + str(t2 - t)
    ce_shuffle.atomchanger(atoms_s5, 'Cu', 'Au', 'Sphere', 5)
    t2 = time.clock()
    print 'S11 = ' + str(t2 - t)
    ce_shuffle.atomchanger(atoms_s5, 'Au', 'Ag', 'Sphere', 5)
    t2 = time.clock()
    print 'S12 = ' + str(t2 - t)
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
    #t3 = cubic(c2, c3, c4, c5, t2)
    #t4 = sphere(s2, s3, s4, s5, t3)
    #deca(d2, d3, d4, d5, t4)

    print 'Done'


main()
