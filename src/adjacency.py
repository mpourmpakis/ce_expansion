#!/usr/bin/env python3

import pickle
import os
import ase.neighborlist
import numpy as np

DEFAULT_ELEMENTS = ("Cu", "Cu")
DEFAULT_RADIUS = 2.8
with open("../data/precalc_coeffs.pickle", "rb") as precalcs:
    DEFAULT_BOND_COEFFS = pickle.load(precalcs)


def buildAdjacencyMatrix(atoms_object: "ase.Atoms",
                         radius_dictionary: "dict" = {DEFAULT_ELEMENTS: DEFAULT_RADIUS}) -> "np.ndarray":
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


def buildAdjacencyList(atoms_object: "ase.Atoms",
                       atom_name: "str" = None,
                       radius_dictionary: "dict" = {DEFAULT_ELEMENTS: DEFAULT_RADIUS}) -> "list":
    """
      Adjacency list representation for an ase atoms object.

      Args:
      atoms_object (ase.Atoms): An ASE atoms object representing the system of interest
      radius_dictionary (dict): A dictionary with the atom-atom radii at-which a bond is considered a
                                bond. If no dict is supplied, Cu-Cu bonds of max-len 2.8 are assumed.

      Returns:
      np.ndarray : A numpy array representing the adjacency list of the ase object

    """
    # Check to see if adjacency list has already been generated
    # NOTE: based on data file paths, not ideal but functional for the moment
    fpath = '../data/adjacency_lists/%s.npy' % atom_name
    if os.path.isfile(fpath):
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
            np.save('../data/adjacency_lists/%s.npy' % atom_name, result)
        return [[i for i in a] for a in result]


def buildBondList(atoms_object: "ase.Atoms",
                  atom_name: "str" = None,
                  radius_dictionary: "dict" = {DEFAULT_ELEMENTS:
                                               DEFAULT_RADIUS}) -> "dict":
    """
      Adjacency *bond* list representation based on "buildAdjacencyList"

      Args:
      atoms_object (ase.Atoms): An ASE atoms object representing the system of
                                  interest
      radius_dictionary (dict): A dictionary with the atom-atom radii at-which
                                  a bond is considered a bond. If no dict is
                                  supplied, Cu-Cu bonds of max-len 2.8 are
                                  assumed.

      Returns:
      dict : A dictonary with an atom index as a key and a list of atoms it's
               bonded to as the value
    """
    adjacency_list = buildAdjacencyList(atoms_object,
                                        atom_name,
                                        radius_dictionary)
    bonds = {}
    for i in range(len(adjacency_list)):
        for con in adjacency_list[i]:
            pair = sorted([i, con])
            # firstcn = [len(a[p]) for p in pair]
            if pair[0] in bonds:
                if pair[1] not in bonds[pair[0]]:
                    bonds[pair[0]].append(pair[1])
            elif pair[1] in bonds:
                if pair[0] not in bonds[pair[1]]:
                    bonds[pair[1]].append(pair[0])
            else:
                bonds[pair[0]] = [pair[1]]
    return bonds


if __name__ == '__main__':
    import ase.cluster
    atom = ase.cluster.Icosahedron('Cu', 3)
    a = buildAdjacencyList(atom)
    b = buildBondList(atom)
