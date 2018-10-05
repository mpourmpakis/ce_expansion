#!/usr/bin/env python3

import pickle

import numpy as np

DEFAULT_ELEMENTS = ("Cu", "Cu")
DEFAULT_RADIUS = 2.8
with open("../data/precalc_coeffs.pickle", "rb") as precalcs:
    DEFAULT_BOND_COEFFS = pickle.load(precalcs)


class AtomGraph(object):
    def __init__(self, adj_list: "np.array",
                 ordering: "np.array",
                 kind0: "str",
                 kind1: "str",
                 coeffs: "dict" = DEFAULT_BOND_COEFFS):
        """
        A graph representing the ase Atoms object to be investigated. First axis is the atom index, second axis contains
        bonding information.
        First entry of the second axis corresponds to a 1/0 representing the atomic kind
        Second entry of the second axis corresponds to the indices of the atoms that atom is bound to

        Args:
        adj_list (np.array) : A numpy array containing adjacency information. Assumed to be produced by the
        buildAdjacencyList function
        ordering (np.array) : A numpy array containing a binary representation of the molecule
        kind0 (str) : Atomic symbol indicating what a "0" in ordering means
        kind1 (str) : Atomic symbol indicating what a "1" in ordering means
        coeffs (dict) : A dictionary of the various bond coefficients we have precalculated, using coeffs.py. Defaults
                        to the global DEFAULT_BOND_COEFFS.


        Attributes:
        adj_list (np.array) : A numpy array containing adjacency information
        ordering (np.array) : A numpy array containing a binary representation of the molecule
        symbols (tuple) : Atomic symbols indicating what the binary representations of the elements in self.ordering
                        means

        """

        self.adj_list = adj_list
        self.ordering = ordering
        self.coeffs = coeffs
        self.symbols = (kind0, kind1)

    def __len__(self):
        return len(self.adj_list)

    def __getitem__(self, atom_key: "int") -> "tuple":
        return self.symbols[self.ordering[atom_key]], self.adj_list[atom_key]

    def getCN(self, atom_key: "int") -> "int":
        """
        Returns the coordination number of the given atom.

        Args:
        atom_key (int) : Index of the atom of interest
        """
        return self.adj_list[atom_key].size

    def getAllCNs(self) -> "np.array":
        """
        Returns a numpy array containing all CN's in the cluster.
        """
        return np.array([entry.size for entry in self.adj_list])

    def get_chemical_symbol(self, index: "int") -> "str":
        """
        Returns the chemical symbol of an atom at the particular index.

        :param index: Index of the atom of interest.

        :return: A string containing the atomic symbol of interest.
        """
        return self.symbols[self.ordering[index]]

    def getHalfBond(self, atom_key: "int", bond_key: "int") -> "float":
        """
        Returns the half-bond energy of a given bond for a certain atom.

        Args:
        atom_key (int) : Index of the atom of interest
        bond_key (int) : Index of the bond of interest

        Returns:
        float : The half-bond energy of that bond at that atom, in units of eV
        """

        atom1 = self.symbols[self.ordering[atom_key]]
        atom2 = self.symbols[self.ordering[self.adj_list[atom_key][bond_key]]]
        return self.coeffs[atom1][atom2][self.getCN(atom_key)]

    def getAtomicCE(self, atom_key: "int") -> "float":
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
        total_energy = 0
        for atom in range(0, len(self.ordering)):
            total_energy += self.getAtomicCE(atom)
        total_CE = total_energy / len(self.ordering)
        return total_CE
