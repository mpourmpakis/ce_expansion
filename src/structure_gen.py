#!/usr/bin/env python3
# TODO: Clean this code up, make it more readable

import ase.cluster, ase.lattice


def create_cube(num_layers: "int", kind: "str" = "Cu") -> "ase.Atoms":
    """
    Creates an FCC cube with faces on the {100} family of planes.

    :param num_layers: Number of unit cells along each side of the cube.
    :type num_layers: int
    :param kind: The element making up the skeleton. Defaults to "Cu"
    :type kind: str

    :return: An ASE atoms object containing the cube skeleton
    """

    lattice = ase.lattice.cubic.FaceCenteredCubic(kind, size=[num_layers]*3)
    cube = ase.build.cut(lattice, extend=1.01)
    return cube


def create_sphere(num_layers: "int", kind: "str" = "Cu") -> "ase.Atoms":
    """
    Inscribes a sphere inside a cube and makes it a nanoparticle. Perfect symmetry not guaranteed.

    :param num_layers: The size of the lattice containing the inscribed sphere.
    :type num_layers: int
    :param kind: The element making up the skeleton. Defaults to "Cu"
    :type kind: str

    :return: An ASE atoms object containing the sphere skeleton.
    """

    atoms = create_cube(num_layers, kind)

    cen = atoms.positions.mean(0)

    radius = (num_layers - 1) * 3.61 / 2

    atoms.append(ase.Atom(kind, cen))


    my_cluster = 0  ase.Atoms(filter(lambda x: check_distance(atoms, radius, x.index), atoms))
    return my_cluster
