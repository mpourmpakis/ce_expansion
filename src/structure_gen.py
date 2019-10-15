#!/usr/bin/env python3
import os
import pathlib
import pickle

import ase.cluster
import ase.lattice
import numpy as np

from atomgraph import adjacency
from npdb import db_inter

# build paths
datapath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        '..', 'data')
atompath = os.path.join(datapath, 'atom_objects')
bondpath = os.path.join(datapath, 'bond_lists')


def build_structure_sql(shape, num_shells,
                        build_bonds_list=True):
    """
    Creates NP of specified shape and size (based on num_shells)

    Args:
    shape (str): shape of NP
                 NOTE: currently supported methods (found in NPBuilder)
                 - cuboctahedron
                 - elongated-trigonal-pyramid
                 - fcc-cube
                 - icosahedron
    num_shells (int): Number of shells in NP
                      e.g. icosahedron with 2 shells makes a 55-atom object
                      ( 1 in core (shell_0) + 12 in shell_1 + 42 in shell_2)

    Kargs:
    build_bonds_list (bool): if True, builds bonds list attribute
                             (default: True)

    Returns:
    (npdb.datatables.Nanoparticle)

    Raises:
            NotImplementedError: given shape has not been implemented
    """
    if num_shells <= 0:
        raise ValueError('Can only build NPs with at least one shell.')

    nanop = db_inter.get_nanoparticle(shape, num_shells=num_shells, lim=1)
    atom = None
    if nanop:
        atom = nanop.get_atoms_obj_skel()
    else:
        try:
            # build atom object
            atom = getattr(NPBuilder, shape.replace('-', '_'))(num_shells)
        except:
            raise NotImplementedError('%s has not been implemented' % shape)

        # insert nanoparticle into DB
        nanop = db_inter.insert_nanoparticle(atom, shape, num_shells)

    # can return atoms obj and bond list or just atoms obj
    if build_bonds_list:

        # make sure bond_list directory exists (if not, make one)
        pathlib.Path(os.path.join(bondpath, shape)).mkdir(parents=True,
                                                          exist_ok=True)

        # if bond_list file (fname) exists, read it in
        # else make and save bond_list
        fname = os.path.join(bondpath, shape, '%i.npy' % num_shells)
        if os.path.isfile(fname):
            bonds_list = np.load(fname)
        else:
            bonds_list = adjacency.buildBondsList(atom)
            np.save(fname, bonds_list)
        nanop.bonds_list = bonds_list
    return nanop


def build_structure(shape, num_shells,
                    return_bonds_list=True):
    """
    Creates NP of specified shape and size (based on num_shells)

    Args:
    shape (str): shape of NP
                 NOTE: currently supported methods (found in NPBuilder)
                 - cuboctahedron
                 - elongated-trigonal-pyramid
                 - fcc-cube
                 - icosahedron
    num_shells (int): Number of shells in NP
                      e.g. icosahedron with 2 shells makes a 55-atom object
                      ( 1 in core (shell_0) + 12 in shell_1 + 42 in shell_2)

    Kargs:
    return_bonds_list (bool): if True, also returns bond_list of Atoms obj
                             (default: True)

    Returns:
            if return_bond_list:
                (ase.Atoms), (list): atom obj and bond_list
            else:
                (ase.Atoms): atom obj of structure

    Raises:
            NotImplementedError: given shape has not been implemented
    """
    if num_shells <= 0:
        raise ValueError('Can only build NPs with at least one shell.')

    # ensure necessary directories exist within local repository
    pathlib.Path(os.path.join(atompath, shape)).mkdir(parents=True,
                                                      exist_ok=True)

    apath = os.path.join(atompath, shape, '%i.pickle' % num_shells)
    if os.path.isfile(apath):
        with open(apath, 'rb') as fidr:
            atom = pickle.load(fidr)
    else:
        try:
            # build atom object
            atom = getattr(NPBuilder, shape.replace('-', '_'))(num_shells)
        except:
            raise NotImplementedError('%s has not been implemented' % shape)

        # only save NPs with at least 1 shell
        if num_shells > 0:
            with open(apath, 'wb') as fidw:
                pickle.dump(atom, fidw)

    # can return atoms obj and bond list or just atoms obj
    if return_bonds_list:

        # make sure bond_list directory exists (if not, make one)
        pathlib.Path(os.path.join(bondpath, shape)).mkdir(parents=True,
                                                          exist_ok=True)

        # if bond_list file (fname) exists, read it in
        # else make and save bond_list
        fname = os.path.join(bondpath, shape, '%i.npy' % num_shells)
        if os.path.isfile(fname):
            bond_list = np.load(fname)
        else:
            bond_list = adjacency.buildBondsList(atom)
            np.save(fname, bond_list)
        return atom, bond_list
    else:
        return atom


class NPBuilder(object):
    """
    Static class that contains methods to build NPs of various shapes

    Args:
    num_shells (int): Number of shells in NP

    KArgs:
    kind (str): What element will be used for the monometallic NPs
                (DEFAULT: Cu)

    Returns:
    (ase.Atoms): the NP skeleton"""

    def cuboctahedron(num_shells, kind="Cu"):
        """
        Creates a cuboctahedral NP.

        Args:
        num_shells (int): Number of shells in NP

        KArgs:
        kind (str): What element will be used for the monometallic NPs
                    (DEFAULT: Cu)

        Returns:
        (ase.Atoms): the NP skeleton
        """
        assert num_shells >= 0
        if num_shells == 0:
            return ase.Atoms(kind)
        return ase.Atoms(ase.cluster.Octahedron(kind, 2 * num_shells + 1,
                                                cutoff=num_shells), pbc=False)

    def elongated_pentagonal_bipyramid(num_shells,
                                       kind="Cu"):
        """
        Creates an elongated-pentagonal-bipyramidal NP.

        Args:
        num_shells (int): Number of shells in NP

        KArgs:
        kind (str): What element will be used for the monometallic NPs
                    (DEFAULT: Cu)

        Returns:
        (ase.Atoms): the NP skeleton
        """
        num_shells += 1
        return ase.Atoms(
            ase.cluster.Decahedron("Cu", num_shells, num_shells, 0),
            pbc=False)

    def fcc_cube(num_units, kind="Cu"):
        """
        Creates an FCC-cube with faces on the {100} family of planes.

        Args:
        num_units (int): Number of primitive FCC units across each side of NP
                        - NOTE: units share a face

        KArgs:
        kind (str): What element will be used for the monometallic NPs
                    (DEFAULT: Cu)

        Returns:
        (ase.Atoms): the NP skeleton
        """
        assert num_units >= 0
        if num_units == 0:
            return ase.Atoms(kind)
        atom = ase.Atoms(ase.cluster.FaceCenteredCubic('Cu', [(1, 0, 0),
                                                              (0, 1, 0),
                                                              (0, 0, 1)],
                                                       [num_units] * 3))
        return atom

    def icosahedron(num_shells, kind="Cu"):
        """
        Creates an icosahedral NP.

        Args:
        num_shells (int): Number of shells in NP

        KArgs:
        kind (str): What element will be used for the monometallic NPs
                    (DEFAULT: Cu)

        Returns:
        (ase.Atoms): the NP skeleton
        """
        assert num_shells >= 0
        return ase.Atoms(ase.cluster.Icosahedron(kind, num_shells + 1),
                         pbc=False)


# WIP - sphere is not perfectly symmetric
def sphere(num_layers, kind="Cu",
           unit_cell_length=3.61):
    """
    Inscribes a sphere inside a cube and makes it a nanoparticle.
    NOTE: Perfect symmetry not guaranteed.

    :param num_layers: The size of the lattice containing the inscribed sphere.
    :type num_layers: int
    :param kind: The element making up the skeleton. Defaults to "Cu"
    :type kind: str
    :param unit_cell_length: The edge-length of the unit cell.
    :type unit_cell_length: float

    :return: An ASE atoms object containing the sphere skeleton.
    """
    raise NotImplementedError
    # Create the cube
    trimmed_cube = fcc_cube(num_layers, kind)

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


if __name__ == '__main__':
    use_sql = False
    shapes = ['cuboctahedron', 'elongated-pentagonal-bipyramid',
              'fcc-cube', 'icosahedron']
    for shape in shapes:
        print('-' * 50)
        print(shape)
        for num_shells in range(1, 16):
            if use_sql:
                nanop = build_structure_sql(shape, num_shells,
                                            build_bonds_list=True)
                atom = nanop.get_atoms_obj_skel()
                bonds = nanop.load_bonds_list()
            else:
                atom, bonds = build_structure(shape, num_shells,
                                              return_bonds_list=True)
            print('%02i: %i' % (num_shells, len(atom)))
        print('-' * 50)
