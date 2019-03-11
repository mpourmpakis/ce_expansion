#!/usr/bin/env python3

import os
import ase.cluster
import ase.lattice
import pathlib
import pickle
import numpy as np
import adjacency
from npdb import db_inter


def build_structure_sql(shape, num_shells, return_bond_list=True):
    """
    Creates Atoms obj of specified shape and size (based on nshell)

    Args:
    shape (str): name of shape for atoms obj
                 NOTE: currently supports
                        - icosahedron
                        - cuboctahedron
                        - fcc-cube
                        - elongated-trigonal-pyramic
    nshell (int): number of shells used to generate atom size
                  e.g. icosahedron with 3 shells makes a 55-atom object
                       ( 1 in core + 12 in shell_1 + 42 in shell_2)

    Kargs:
    return_bond_list (bool): if True, also returns bond_list of Atoms obj
                             (default: True)

    Returns:
            if return_bond_list:
                (ase.Atoms), (list): atom obj and bond_list
            else:
                (ase.Atoms): atom obj of structure

    Raises:
            NotImplementedError: given shape has not been implemented
    """
    np = db_inter.get_nanoparticle(shape, num_shells=num_shells, lim=1)
    if np:
        return np
    else:
        # build atom object
        if shape == 'icosahedron':
            atom = ase.cluster.Icosahedron('Cu', num_shells)
        elif shape == 'fcc-cube':
            atom = ase.cluster.FaceCenteredCubic('Cu', [(1, 0, 0),
                                                        (0, 1, 0),
                                                        (0, 0, 1)],
                                                 [num_shells] * 3)
        elif shape == 'cuboctahedron':
            atom = ase.cluster.Octahedron('Cu', 2 * num_shells + 1,
                                          cutoff=num_shells)
        elif shape == 'elongated-pentagonal-bipyramid':
            atom = ase.cluster.Decahedron('Cu', num_shells, num_shells, 0)
        else:
            raise NotImplementedError('%s has not been implemented')

        np = db_inter.insert_nanoparticle(atom, shape, num_shells)

    # can return atoms obj and bond list or just atoms obj
    if return_bond_list:

        # make sure bond_list directory exists (if not, make one)
        bond_list_path = '../data/bond_lists/%s/' % shape
        pathlib.Path(bond_list_path).mkdir(parents=True, exist_ok=True)

        # if bond_list file (fname) exists, read it in
        # else make and save bond_list
        fname = bond_list_path + '%i.npy' % nshell
        if os.path.isfile(fname):
            bond_list = np.load(fname)
        else:
            bond_list = adjacency.buildBondsList(atom)
            np.save(fname, bond_list)
        return atom, bond_list
    else:
        return atom


def build_structure(shape, nshell, return_bond_list=True):
    """
    Creates Atoms obj of specified shape and size (based on nshell)

    Args:
    shape (str): name of shape for atoms obj
                 NOTE: currently supports
                        - icosahedron
                        - cuboctahedron
                        - fcc-cube
                        - elongated-trigonal-pyramic
    nshell (int): number of shells used to generate atom size
                  e.g. icosahedron with 3 shells makes a 55-atom object
                       ( 1 in core + 12 in shell_1 + 42 in shell_2)

    Kargs:
    return_bond_list (bool): if True, also returns bond_list of Atoms obj
                             (default: True)

    Returns:
            if return_bond_list:
                (ase.Atoms), (list): atom obj and bond_list
            else:
                (ase.Atoms): atom obj of structure

    Raises:
            NotImplementedError: given shape has not been implemented
    """
    # ensure necessary directories exist within local repository
    pathlib.Path('../data/atom_objects/%s/' % shape).mkdir(parents=True,
                                                           exist_ok=True)
    apath = '../data/atom_objects/%s/%i.pickle' % (shape, nshell)
    if os.path.isfile(apath):
        with open(apath, 'rb') as fidr:
            atom = pickle.load(fidr)
    else:

        # build atom object
        if shape == 'icosahedron':
            atom = ase.cluster.Icosahedron('Cu', nshell)
        elif shape == 'fcc-cube':
            atom = ase.cluster.FaceCenteredCubic('Cu', [(1, 0, 0),
                                                        (0, 1, 0),
                                                        (0, 0, 1)],
                                                 [nshell] * 3)
        elif shape == 'cuboctahedron':
            atom = ase.cluster.Octahedron('Cu', 2 * nshell + 1,
                                          cutoff=nshell)
        elif shape == 'elongated-pentagonal-bipyramid':
            atom = ase.cluster.Decahedron('Cu', nshell, nshell, 0)
        else:
            raise NotImplementedError('%s has not been implemented')

        with open(apath, 'wb') as fidw:
            pickle.dump(atom, fidw)

    # can return atoms obj and bond list or just atoms obj
    if return_bond_list:

        # make sure bond_list directory exists (if not, make one)
        bond_list_path = '../data/bond_lists/%s/' % shape
        pathlib.Path(bond_list_path).mkdir(parents=True, exist_ok=True)

        # if bond_list file (fname) exists, read it in
        # else make and save bond_list
        fname = bond_list_path + '%i.npy' % nshell
        if os.path.isfile(fname):
            bond_list = np.load(fname)
        else:
            bond_list = adjacency.buildBondsList(atom)
            np.save(fname, bond_list)
        return atom, bond_list
    else:
        return atom


def cube(num_layers: "int", kind: "str" = "Cu") -> "ase.Atoms":
    """
    Creates an FCC cube with faces on the {100} family of planes.

    :param num_layers: Number of unit cells along each side of the cube.
    :type num_layers: int
    :param kind: The element making up the skeleton. Defaults to "Cu"
    :type kind: str

    :return: An ASE atoms object containing the cube skeleton
    """

    lattice = ase.lattice.cubic.FaceCenteredCubic(kind, size=[num_layers] * 3)
    cube = ase.build.cut(lattice, extend=1.01)
    return cube


def sphere(num_layers: "int", kind: "str" = "Cu", unit_cell_length: "float" = 3.61) -> "ase.Atoms":
    """
    Inscribes a sphere inside a cube and makes it a nanoparticle. Perfect symmetry not guaranteed.

    :param num_layers: The size of the lattice containing the inscribed sphere.
    :type num_layers: int
    :param kind: The element making up the skeleton. Defaults to "Cu"
    :type kind: str
    :param unit_cell_length: The edge-length of the unit cell.
    :type unit_cell_length: float

    :return: An ASE atoms object containing the sphere skeleton.
    """

    # Create the cube
    trimmed_cube = cube(num_layers, kind)

    # Simple geometry
    center = trimmed_cube.positions.mean(0)
    cutoff_radius = num_layers * unit_cell_length / 1.99
    distance_list = map(ase.np.linalg.norm,
                        ase.geometry.get_distances(trimmed_cube.get_positions(), p2=center)[1])

    # Build the sphere using atoms that are within the cutoff
    sphere = ase.Atoms()
    for atom, distance in zip(trimmed_cube, distance_list):
        if distance <= cutoff_radius:
            sphere += atom

    return sphere


def elongated_pentagonal_bipyramid(num_layers: "int", kind: "str" = "Cu") -> "ase.Atoms":
    """
    Creates an elongated pentagonal bipyramidal NP.

    :param num_layers: Number of layers in the EPB
    :type num_layers: int
    :param kind: These are monometallic. What element will be used. Defaults to "Cu"
    :type kind: str

    :return: An ASE atoms object containing the elongated pentagonal bipyramid skeleton.
    """
    return ase.cluster.Decahedron("Cu", num_layers, num_layers, 0)


def cuboctahedron(num_layers: "int", kind: "str" = "Cu") -> "ase.Atoms":
    """
    Creates a cuboctahedral NP.

    :param num_layers: Number of layers in the cuboctahedron.
    :type num_layers: int
    :param kind: These are monometallic. What element will bee used. Defaults to "Cu".
    :type kind: str

    :return: An ASE atoms object containing the cuboctahedron skeleton.
    """
    return ase.cluster.Octahedron(kind, 2 * num_layers + 1, cutoff=num_layers)


if __name__ == '__main__':
    np = build_structure_sql('icosahedron', 25)
