#!/usr/bin/env python3
# TODO: Clean this code up, make it more readable

import ase.cluster


def check_distance(atoms, radius, index):
    distance = atoms.get_distance(-1, index)
    print(distance)
    if distance <= radius and distance != 0:
        return True
    else:
        return False


# structures
def structuregen(shape, parameters):
    if shape == "Icosahedron":
        atoms = ase.cluster.Icosahedron('Cu', noshells=parameters[0])
        return atoms
    elif shape == "Decahedron":
        atoms = ase.cluster.Decahedron('Cu', p=parameters[0], q=parameters[1], r=parameters[2])
        return atoms
    elif shape == "Lattice":
        atoms = ase.cluster.FaceCenteredCubic('Cu', ((1, 0, 0), (0, 1, 0), (0, 0, 1)), parameters)
        return atoms
    elif shape == "Sphere":
        atoms = ase.cluster.FaceCenteredCubic('Cu', ((1, 0, 0), (0, 1, 0), (0, 0, 1)), parameters)
        cen = atoms.positions.mean(0)
        radius = (parameters[0] - 1) * 3.61 / 2
        atoms.append(ase.Atom("Cu", cen))
        # Filter
        # 1st arg is a function returning T or F
        #    Check_distance needs 3 args: atoms object, radius, and a count
        # The rest of the arguments are lists of things that get plugged in

        my_cluster = ase.Atoms(filter(lambda x: check_distance(atoms, radius, x.index), atoms))
        print(my_cluster)
        return my_cluster
