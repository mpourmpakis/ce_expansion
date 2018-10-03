#!/usr/bin/env python3
# TODO: Clean this code up, make it more readable

import ase.cluster


def create_lattice(parameters):
    atoms = ase.cluster.FaceCenteredCubic('Cu', ((1, 0, 0), (0, 1, 0), (0, 0, 1)), parameters)
    return atoms


def create_sphere(parameters):
    atoms = ase.cluster.FaceCenteredCubic('Cu', ((1, 0, 0), (0, 1, 0), (0, 0, 1)), parameters)
    cen = atoms.positions.mean(0)
    radius = (parameters[0] - 1) * 3.61 / 2
    atoms.append(ase.Atom("Cu", cen))
    # Filter
    # 1st arg is a function returning T or F
    #    Check_distance needs 3 args: atoms object, radius, and a count
    # The rest of the arguments are lists of things that get plugged in

    my_cluster = 0#ase.Atoms(filter(lambda x: check_distance(atoms, radius, x.index), atoms))
    return my_cluster
