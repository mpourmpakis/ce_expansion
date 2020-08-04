#!/usr/bin/env python3

# Library for buiding bonding lists/tables/matrices
# James Dean, 2019

import os
import pathlib

import ase.neighborlist
from ase.data import covalent_radii
import numpy as np

# Functions below
def buildBondsList(atoms_object,
                   radius_dictionary=None):
    """
    2D bonds list from an ASE atoms object.

    NOTE: incredibly slow if code has interacted with SQL DB
          in that case, use build_bonds_list

    Args:
    atoms_object (ase.Atoms): An ASE atoms object representing the system of interest
    radius_dictionary (dict): A dictionary with the atom-atom radii at-which a bond is considered a
      bond. If no dict is supplied (default behavior), the bonds are calculated automatically.

    Returns:
    np.ndarray : A numpy array representing the bonds list.
    """
    if radius_dictionary is None:
        radius_dictionary = ase.neighborlist.natural_cutoffs(atoms_object, 1.2)
    sources, destinations = ase.neighborlist.neighbor_list("ij", atoms_object, radius_dictionary)
    return np.column_stack((sources, destinations))


def build_bonds_list(atoms_object, radius_dictionary):
    """
    ALTERNATIVE FUNCTION WHEN INTERACTING WITH SQLite DB
    Finds bonds between atoms based on bonding radii
    Default radii to determine bonding: covalent radii * 1.25

    Args:
    atoms_object (ase.Atoms): atoms object

    KArgs:
    radius_dictionary (dict): A dictionary with the atom-atom radii at-which a bond is considered a
      bond. If no dict is supplied (default behavior), the bonds are calculated automatically.
    """
    # remove periodic boundaries
    atoms_object = atoms_object.copy()
    atoms_object.pbc = False

    # use default radius_arr = covalent_radii
    if radius_dictionary is None:
        radius_dictionary = ase.neighborlist.natural_cutoffs(atoms_object, 1.2)

    # create neighborlist object
    n = ase.neighborlist.NeighborList(radius_dictionary, skin=0,
                                      self_interaction=False)
    n.update(atoms_object)
    if not n.nneighbors:
        return []

    bonds = np.zeros((n.nneighbors, 2), int)
    spot1 = 0
    for atomi in range(len(atoms_object)):
        # get neighbors of atomi
        neighs = n.get_neighbors(atomi)[0]

        # find second cutoff in matrix
        spot2 = spot1 + len(neighs)

        # add bonds to matrix
        bonds[spot1:spot2, 0] = atomi
        bonds[spot1:spot2, 1] = neighs

        # shift down matrix
        spot1 = spot2

        # once all bonds have been found break loop
        if spot1 == n.nneighbors:
            break

    return np.concatenate((bonds, bonds[:, ::-1]))


def buildAdjacencyMatrix(atoms_object,
                         radius_dictionary=None):
    """
    Sparse matrix representation from an ase atoms object.

    Args:
    atoms_object (ase.Atoms): An ASE atoms object representing the system of interest
    radius_dictionary (dict): A dictionary with the atom-atom radii at-which a bond is considered a
                              bond. If no dict is supplied (default behavior), the bonds are calculated automatically.

    Returns:
    np.ndarray : A numpy array representing the sparse matrix of the ase object
    """
    if radius_dictionary is None:
        radius_dictionary = ase.neighborlist.natural_cutoffs(atoms_object, 1.2)
    # Construct the list of bonds
    sources, destinations = ase.neighborlist.neighbor_list("ij", atoms_object, radius_dictionary)
    # Generate the matrix
    adjacency_matrix = np.zeros((len(atoms_object), len(atoms_object)))
    for bond in zip(sources, destinations):
        adjacency_matrix[bond[0], bond[1]] += 1

    return adjacency_matrix


def buildAdjacencyList(atoms_object,
                       atom_name=None,
                       radius_dictionary=None):
    """
      Adjacency list representation for an ase atoms object.


      atoms_object (ase.Atoms): An ASE atoms object representing the system of interest
      radius_dictionary (dict): A dictionary with the atom-atom radii at-which a bond is considered a
                                bond. If no dict is supplied (default behavior), the bonds are calculated automatically.

      Returns:
      np.ndarray : A numpy array representing the adjacency list of the ase object

    """
    # Check to see if adjacency list has already been generated
    # Current folder structure:
    #   Project
    #   |---bin
    #       |---lib.dll
    #   |---ce_expansion
    #       |---atomgraph
    #           |----interface.py
    if radius_dictionary is None:
        radius_dictionary = ase.neighborlist.natural_cutoffs(atoms_object, 1.2)
    path = os.path.realpath(__file__)
    data_directory = os.sep.join(path.split(os.sep)[:-3] + ["data"])

    pathlib.Path(os.sep.join([data_directory, 'adjacency_lists'])).mkdir(parents=True, exist_ok=True)
    fpath = os.sep.join([data_directory, '%s.npy']) % atom_name
    if os.path.isfile(fpath) and 0:
        adj = np.load(fpath)
        return [[i for i in a] for a in adj]

    # Construct the list of bonds
    sources, destinations = ase.neighborlist.neighbor_list("ij", atoms_object, radius_dictionary)
    # Sort along our destinations
    # Mergesort has a slightly better worst-case time complexity than quicksort or heapsort, and is stable
    sorted_destinations = destinations[sources.argsort(kind='mergesort')]

    # Figure out how the list of bonds will be sliced, and slice it
    bins = np.bincount(sources)
    splitting = np.zeros(len(bins), dtype=int)
    for count, item in enumerate(bins):
        if count == 0:
            splitting[count] = item
        else:
            splitting[count] = item + splitting[count - 1]

    # Slice the list of bonds to get the adjacency list
    adjacency_list = np.split(sorted_destinations, splitting)

    # Check that the final entry is an empty list, otherwise something weird happened
    if len(adjacency_list[-1]) != 0:
        raise ValueError(
            "The following atoms have bonds yet do not appear to be bound to any item: " + str(adjacency_list[-1]))
    else:
        result = np.delete(adjacency_list, -1)
        if atom_name:
            np.save(os.sep.join([data_directory, '%s.npy']) % atom_name, result)
        return [[i for i in a] for a in result]


if __name__ == '__main__':
    import ase.cluster

    nanoparticle = ase.cluster.Icosahedron('Cu', 2)
    adjacency_list = buildAdjacencyList(nanoparticle)
    adjacency_matrix = buildAdjacencyMatrix(nanoparticle)
    bonds_list = buildBondsList(nanoparticle)

    print(adjacency_list, adjacency_matrix, bonds_list)
