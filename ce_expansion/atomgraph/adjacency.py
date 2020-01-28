#!/usr/bin/env python3

# Library for buiding bonding lists/tables/matrices
# James Dean, 2019

import os
import pathlib

import ase.neighborlist
import numpy as np

# Set up globals for defaults
DEFAULT_ELEMENTS = ("Cu", "Cu")
DEFAULT_RADIUS = 2.8


# Functions below
def buildBondsList(atoms_object,
                   radius_dictionary={DEFAULT_ELEMENTS: DEFAULT_RADIUS}):
    """
    2D bonds list from an ASE atoms object.

    Args:
    atoms_object (ase.Atoms): An ASE atoms object representing the system of interest
    radius_dictionary (dict): A dictionary with the atom-atom radii at-which a bond is considered a
      bond. If no dict is supplied, Cu-Cu bonds of a max-len 2.8 are assumes.

    Returns:
    np.ndarray : A numpy array representing the bonds list.
    """
    sources, destinations = ase.neighborlist.neighbor_list("ij", atoms_object, radius_dictionary)
    return np.column_stack((sources, destinations))


def buildAdjacencyMatrix(atoms_object,
                         radius_dictionary={DEFAULT_ELEMENTS: DEFAULT_RADIUS}):
    """
    Sparse matrix representation from an ase atoms object.

    Args:
    atoms_object (ase.Atoms): An ASE atoms object representing the system of interest
    radius_dictionary (dict): A dictionary with the atom-atom radii at-which a bond is considered a
                              bond. If no dict is supplied, Cu-Cu bonds of max-len 2.8 are assumed.

    Returns:
    np.ndarray : A numpy array representing the sparse matrix of the ase object
    """
    # Construct the list of bonds
    sources, destinations = ase.neighborlist.neighbor_list("ij", atoms_object, radius_dictionary)
    # Generate the matrix
    adjacency_matrix = np.zeros((len(atoms_object), len(atoms_object)))
    for bond in zip(sources, destinations):
        adjacency_matrix[bond[0], bond[1]] += 1

    return adjacency_matrix


def buildAdjacencyList(atoms_object,
                       atom_name=None,
                       radius_dictionary={DEFAULT_ELEMENTS: DEFAULT_RADIUS}):
    """
      Adjacency list representation for an ase atoms object.


      atoms_object (ase.Atoms): An ASE atoms object representing the system of interest
      radius_dictionary (dict): A dictionary with the atom-atom radii at-which a bond is considered a
                                bond. If no dict is supplied, Cu-Cu bonds of max-len 2.8 are assumed.

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
