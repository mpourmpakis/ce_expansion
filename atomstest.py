#!/usr/bin/env python
from random import shuffle
import csv
import time
from ase import Atoms, Atom
from ase.cluster.icosahedron import Icosahedron
from ase.cluster.decahedron import Decahedron
from ase.cluster.cubic import FaceCenteredCubic
from math import floor


def check_distance(atoms, radius, index):
    distance = atoms.get_distance(-1, index)
    print distance
    if distance <= radius and distance != 0:
        return True
    else:
        return False


def structuregen(shape, parameters):
    if shape == "Icosahedron":
        atoms = Icosahedron('Cu', noshells=parameters[0])
        return atoms
    elif shape == "Decahedron":
        atoms = Decahedron('Cu', p=parameters[0], q=parameters[1], r=parameters[2])
        return atoms
    elif shape == "Lattice":
        atoms = FaceCenteredCubic('Cu', ((1, 0, 0), (0, 1, 0), (0, 0, 1)), parameters)
        return atoms
    elif shape == "Sphere":
        atoms = FaceCenteredCubic('Cu', ((1, 0, 0), (0, 1, 0), (0, 0, 1)), parameters)
        cen = atoms.positions.mean(0)
        radius = (parameters[0] - 1) * 3.61 / 2
        atoms.append(Atom("Cu", cen))
        # Filter
        # 1st arg is a function returning T or F
        #    Check_distance needs 3 args: atoms object, radius, and a count
        # The rest of the arguments are lists of things that get plugged in

        my_cluster = Atoms(filter(lambda x: check_distance(atoms, radius, x.index), atoms))
        print my_cluster
        return my_cluster


def atomchanger(nanop, el1, el2, shape, shells):
    # nanop = ase Atom object
    # el1 = first element requested
    # el2 = second element requested

    nanoplist = nanop.get_chemical_symbols()
    nanonum = len(nanoplist)
    # Read in data from atom object.
    name = el1 + el2 + shape + str(shells) + '.csv'

    with open(name, 'wb') as csvfile:
        atomscribe = csv.DictWriter(csvfile, delimiter=',',
                                    fieldnames=['Element 1', 'Percentage 1', 'Element 2', 'Percentage 2',
                                                'Cohesive Energy', 'Excess Energy'])
        atomscribe.writeheader()
        for j in range(0, 101):
            divider = floor(j * nanonum / 100)
            x = 0
            while x < nanonum:
                if x < divider:
                    nanoplist[x] = el1
                elif x >= divider:
                    nanoplist[x] = el2
                x = x + 1

            modlist = list(nanoplist)

            for i in range(0, 1000):
                shuffle(modlist)
                nanop.set_chemical_symbols(modlist)
                # C,E = Func(nanop)
                atomscribe.writerow({'Element 1': el1, 'Percentage 1': j, 'Element 2': el2, 'Percentage 2': 100 - j,
                                     'Cohesive Energy': i, 'Excess Energy': i})


def i_structure():
    atoms_i2 = structuregen("Icosahedron", [2])
    atoms_i3 = structuregen("Icosahedron", [3])
    atoms_i4 = structuregen("Icosahedron", [4])
    atoms_i5 = structuregen("Icosahedron", [5])
    return atoms_i2, atoms_i3, atoms_i4, atoms_i5


def d_structure():
    atoms_d2 = structuregen("Decahedron", [2, 2, 2])
    atoms_d3 = structuregen("Decahedron", [3, 3, 3])
    atoms_d4 = structuregen("Decahedron", [4, 4, 4])
    atoms_d5 = structuregen("Decahedron", [5, 5, 5])
    return atoms_d2, atoms_d3, atoms_d4, atoms_d5


def c_structure():
    atoms_c2 = structuregen("Lattice", [2, 2, 2])
    atoms_c3 = structuregen("Lattice", [3, 3, 3])
    atoms_c4 = structuregen("Lattice", [4, 4, 4])
    atoms_c5 = structuregen("Lattice", [5, 5, 5])
    return atoms_c2, atoms_c3, atoms_c4, atoms_c5


def s_structure():
    atoms_s2 = structuregen("Lattice", [2, 2, 2])
    atoms_s3 = structuregen("Lattice", [3, 3, 3])
    atoms_s4 = structuregen("Lattice", [4, 4, 4])
    atoms_s5 = structuregen("Lattice", [5, 5, 5])
    return atoms_s2, atoms_s3, atoms_s4, atoms_s5


def icosa(atoms_i2, atoms_i3, atoms_i4, atoms_i5):
    t = time.clock()
    print 'start = ' + str(t)
    atomchanger(atoms_i2, 'Cu', 'Ag', 'Icosahedron', 2)
    t2 = time.clock()
    print 'I1= ' + str(t2 - t)
    atomchanger(atoms_i2, 'Cu', 'Au', 'Icosahedron', 2)
    t2 = time.clock()
    print 'I2= ' + str(t2 - t)
    atomchanger(atoms_i2, 'Au', 'Ag', 'Icosahedron', 2)
    t2 = time.clock()
    print 'I3 = ' + str(t2 - t)

    atomchanger(atoms_i3, 'Cu', 'Ag', 'Icosahedron', 3)
    t2 = time.clock()
    print 'I4 = ' + str(t2 - t)
    atomchanger(atoms_i3, 'Cu', 'Au', 'Icosahedron', 3)
    t2 = time.clock()
    print 'I5 = ' + str(t2 - t)
    atomchanger(atoms_i3, 'Au', 'Ag', 'Icosahedron', 3)
    t2 = time.clock()
    print 'I6 = ' + str(t2 - t)

    atomchanger(atoms_i4, 'Cu', 'Ag', 'Icosahedron', 4)
    t2 = time.clock()
    print 'I7 = ' + str(t2 - t)
    atomchanger(atoms_i4, 'Cu', 'Au', 'Icosahedron', 4)
    t2 = time.clock()
    print 'I8 = ' + str(t2 - t)
    atomchanger(atoms_i4, 'Au', 'Ag', 'Icosahedron', 4)
    t2 = time.clock()
    print 'I9 = ' + str(t2 - t)

    atomchanger(atoms_i5, 'Cu', 'Ag', 'Icosahedron', 5)
    t2 = time.clock()
    print 'I10 = ' + str(t2 - t)
    atomchanger(atoms_i5, 'Cu', 'Au', 'Icosahedron', 5)
    t2 = time.clock()
    print 'I11 = ' + str(t2 - t)
    atomchanger(atoms_i5, 'Au', 'Ag', 'Icosahedron', 5)
    t2 = time.clock()
    print 'I12 = ' + str(t2 - t)
    return t2


def deca(atoms_d2, atoms_d3, atoms_d4, atoms_d5, t):
    atomchanger(atoms_d2, 'Cu', 'Ag', 'Decahedron', 2)
    t2 = time.clock()
    print 'D1 = ' + str(t2 - t)
    atomchanger(atoms_d2, 'Cu', 'Au', 'Decahedron', 2)
    t2 = time.clock()
    print 'D2 = ' + str(t2 - t)
    atomchanger(atoms_d2, 'Au', 'Ag', 'Decahedron', 2)
    t2 = time.clock()
    print 'D3 = ' + str(t2 - t)

    atomchanger(atoms_d3, 'Cu', 'Ag', 'Decahedron', 3)
    t2 = time.clock()
    print 'D4 = ' + str(t2 - t)
    atomchanger(atoms_d3, 'Cu', 'Au', 'Decahedron', 3)
    t2 = time.clock()
    print 'D5 = ' + str(t2 - t)
    atomchanger(atoms_d3, 'Au', 'Ag', 'Decahedron', 3)
    t2 = time.clock()
    print 'D6 = ' + str(t2 - t)

    atomchanger(atoms_d4, 'Cu', 'Ag', 'Decahedron', 4)
    t2 = time.clock()
    print 'D7 = ' + str(t2 - t)
    atomchanger(atoms_d4, 'Cu', 'Au', 'Decahedron', 4)
    t2 = time.clock()
    print 'D8 = ' + str(t2 - t)
    atomchanger(atoms_d4, 'Au', 'Ag', 'Decahedron', 4)
    t2 = time.clock()
    print 'D9 = ' + str(t2 - t)

    atomchanger(atoms_d5, 'Cu', 'Ag', 'Decahedron', 5)
    t2 = time.clock()
    print 'D10 = ' + str(t2 - t)
    atomchanger(atoms_d5, 'Cu', 'Au', 'Decahedron', 5)
    t2 = time.clock()
    print 'D11 = ' + str(t2 - t)
    atomchanger(atoms_d5, 'Au', 'Ag', 'Decahedron', 5)
    t2 = time.clock()
    print 'D12 = ' + str(t2 - t)


def cubic(atoms_c2, atoms_c3, atoms_c4, atoms_c5, t):
    atomchanger(atoms_c2, 'Cu', 'Ag', 'Cube', 2)
    t2 = time.clock()
    print 'C1 = ' + str(t2 - t)
    atomchanger(atoms_c2, 'Cu', 'Au', 'Cube', 2)
    t2 = time.clock()
    print 'C2 = ' + str(t2 - t)
    atomchanger(atoms_c2, 'Au', 'Ag', 'Cube', 2)
    t2 = time.clock()
    print 'C3 = ' + str(t2 - t)

    atomchanger(atoms_c3, 'Cu', 'Ag', 'Cube', 3)
    t2 = time.clock()
    print 'C4 = ' + str(t2 - t)
    atomchanger(atoms_c3, 'Cu', 'Au', 'Cube', 3)
    t2 = time.clock()
    print 'C5 = ' + str(t2 - t)
    atomchanger(atoms_c3, 'Au', 'Ag', 'Cube', 3)
    t2 = time.clock()
    print 'C6 = ' + str(t2 - t)

    atomchanger(atoms_c4, 'Cu', 'Ag', 'Cube', 4)
    t2 = time.clock()
    print 'C7 = ' + str(t2 - t)
    atomchanger(atoms_c4, 'Cu', 'Au', 'Cube', 4)
    t2 = time.clock()
    print 'C8 = ' + str(t2 - t)
    atomchanger(atoms_c4, 'Au', 'Ag', 'Cube', 4)
    t2 = time.clock()
    print 'C9 = ' + str(t2 - t)
    atomchanger(atoms_c5, 'Cu', 'Ag', 'Cube', 5)
    t2 = time.clock()
    print 'C10 = ' + str(t2 - t)
    atomchanger(atoms_c5, 'Cu', 'Au', 'Cube', 5)
    t2 = time.clock()
    print 'C11 = ' + str(t2 - t)
    atomchanger(atoms_c5, 'Au', 'Ag', 'Cube', 5)
    t2 = time.clock()
    print 'C12 = ' + str(t2 - t)
    return t2

def sphere(atoms_s2, atoms_s3, atoms_s4, atoms_s5, t):
    atomchanger(atoms_s2, 'Cu', 'Ag', 'Sphere', 2)
    t2 = time.clock()
    print 'S1 = ' + str(t2 - t)
    atomchanger(atoms_s2, 'Cu', 'Au', 'Sphere', 2)
    t2 = time.clock()
    print 'S2 = ' + str(t2 - t)
    atomchanger(atoms_s2, 'Au', 'Ag', 'Sphere', 2)
    t2 = time.clock()
    print 'S3 = ' + str(t2 - t)

    atomchanger(atoms_s3, 'Cu', 'Ag', 'Sphere', 3)
    t2 = time.clock()
    print 'S4 = ' + str(t2 - t)
    atomchanger(atoms_s3, 'Cu', 'Au', 'Sphere', 3)
    t2 = time.clock()
    print 'S5 = ' + str(t2 - t)
    atomchanger(atoms_s3, 'Au', 'Ag', 'Sphere', 3)
    t2 = time.clock()
    print 'S6 = ' + str(t2 - t)

    atomchanger(atoms_s4, 'Cu', 'Ag', 'Sphere', 4)
    t2 = time.clock()
    print 'S7 = ' + str(t2 - t)
    atomchanger(atoms_s4, 'Cu', 'Au', 'Sphere', 4)
    t2 = time.clock()
    print 'S8 = ' + str(t2 - t)
    atomchanger(atoms_s4, 'Au', 'Ag', 'Sphere', 4)
    t2 = time.clock()
    print 'S9 = ' + str(t2 - t)

    atomchanger(atoms_s5, 'Cu', 'Ag', 'Sphere', 5)
    t2 = time.clock()
    print 'S10 = ' + str(t2 - t)
    atomchanger(atoms_s5, 'Cu', 'Au', 'Sphere', 5)
    t2 = time.clock()
    print 'S11 = ' + str(t2 - t)
    atomchanger(atoms_s5, 'Au', 'Ag', 'Sphere', 5)
    t2 = time.clock()
    print 'S12 = ' + str(t2 - t)
    return t2

def main():
    [c2, c3, c4, c5] = c_structure()
    [i2, i3, i4, i5] = i_structure()
    [d2, d3, d4, d5] = d_structure()
    [s2, s3, s4, s5] = s_structure()

    t2 = icosa(i2, i3, i4, i5)
    t3 = cubic(c2, c3, c4, c5, t2)
    t4 = sphere(s2, s3, s4, s5, t3)
    deca(d2, d3, d4, d5, t4)

    print 'Done'


main()
