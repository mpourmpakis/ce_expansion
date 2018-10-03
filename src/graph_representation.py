#!/usr/bin/env python

import ase.neighborlist
import numpy as np
import pickle

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
                       radius_dictionary: "dict" = {DEFAULT_ELEMENTS: DEFAULT_RADIUS}) -> "np.ndarray":
    """
      Adjacency list representation for an ase atoms object.

      Args:
      atoms_object (ase.Atoms): An ASE atoms object representing the system of interest
      radius_dictionary (dict): A dictionary with the atom-atom radii at-which a bond is considered a
                                bond. If no dict is supplied, Cu-Cu bonds of max-len 2.8 are assumed.

      Returns:
      np.ndarray : A numpy array representing the adjacency list of the ase object

    """

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
        return np.delete(adjacency_list, -1)


class AtomGraph(object):
    def __init__(self, adj_list: "np.array",
                 colors: "np.array",
                 kind0: "str",
                 kind1: "str",
                 coeffs: "dict" = DEFAULT_BOND_COEFFS):
        """
        A graph representing the ase Atoms object to be investigated. First axis is the atom index, second axis containst bonding information.
        First entry of the second axis corresponds to a 1/0 representing the atomic kind
        Second entry of the second axis corresponds to the indices of the atoms that atom is bound to

        Args:
        adj_list (np.array) : A numpy array containing adjacency information. Assumed to be produced by the buildAdjacencyList function
        colors (np.array) : A numpy array containing a binary representation of the molecule
        kind0 (str) : Atomic symbol indicating what a "0" in colors means
        kind1 (str) : Atomic symbol indicating what a "1" in colors means
        coeffs (dict) : A dictionary of the various bond coefficients we have precalculated, using coeffs.py. Defaults to the global DEFAULT_BOND_COEFFS.


        Attributes:
        adj_list (np.array) : A numpy array containing adjacency information
        colors (np.array) : A numpy array containing a binary representation of the molecule
        symbols (tuple) : Atomic symbols indicating what the binary representations of the elements in self.colors means

        """

        self.adj_list = adj_list
        self.colors = colors
        self.coeffs = coeffs
        self.symbols = (kind0, kind1)

    def __len__(self):
        return len(self.adj_list)

    def __getitem__(self, atom_key: "int") -> "tuple":
        return (self.symbols[self.colors[atom_key]], self.adj_list[atom_key])

    def getCN(self, atom_key: "int") -> "int":
        """
        Returns the coordination number of the given atom.

        Args:
        atom_key (int) : Index of the atom of interest
        """
        return len(self[atom_key][1])

    def getAllCNs(self) -> "np.array":
        """
        Returns a numpy array containing all CN's in the cluster.
        """
        return np.array([entry.size for entry in self.adj_list])

    def getHalfBond(self, atom_key: "int") -> "float":
        """
        Returns the half-bond energy of a given bond for a certain atom.

        Args:
        atom_key (int) : Index of the atom of interest
        bond_key (int) : Index of the bond of interest

        Returns:
        float : The half-bond energy of that bond at that atom, in units of eV
        """
        atom1 = self.colors[atom_key]
        atom2 = self.colors[self.adj_list[atom_key]]
        return self.coeffs[atom1][atom2][self.getCN(atom_key)]

    def getLocalCE(self, atom_key: "int") -> "float":
        """
        Returns the sum of half-bond energies for a particular atom.

        Args:
        atom_key : Index of the atom of interest

        Returns:
        float : The sum of all half-bond energies at that atom, in units of eV
        """
        local_CE = 0
        for bond_key in range(len(self.adj_list[atom_key])):
            local_CE += self.getHalfBond(atom_key, bond_key)
        return local_CE

    def getTotalCE(self) -> "float":
        """
        Returns the cohesive energy of the cluster as a whole.

        Returns
        float : The CE, in units of eV
        """
        total_CE = 0
        for atom_key in len(self):
            total_CE += self.getLocalCE(atom_key)
        return total_CE
