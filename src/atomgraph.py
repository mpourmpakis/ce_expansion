#!/usr/bin/env python3

import pickle
import numpy as np

DEFAULT_ELEMENTS = ("Cu", "Cu")
DEFAULT_RADIUS = 2.8
with open("../data/precalc_coeffs.pickle", "rb") as precalcs:
    DEFAULT_BOND_COEFFS = pickle.load(precalcs)


class AtomGraph(object):
    def __init__(self, adj_list: "np.array",
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
        kind0 (str) : Atomic symbol indicating what a "0" in ordering means
        kind1 (str) : Atomic symbol indicating what a "1" in ordering means
        coeffs (dict) : A dictionary of the various bond coefficients we have precalculated, using coeffs.py. Defaults
                        to the global DEFAULT_BOND_COEFFS.


        Attributes:
        adj_list (np.array) : A numpy array containing adjacency information
        symbols (tuple) : Atomic symbols indicating what the binary representations of the elements in ordering
                        means

        """

        self.adj_list = adj_list
        self.cns = [len(a) for a in adj_list]
        self.coeffs = coeffs
        self.symbols = (kind0, kind1)

        # total number of atoms
        self.n_atoms = len(adj_list)

    def __len__(self):
        return len(self.adj_list)

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
        return np.array([len(entry) for entry in self.adj_list])

    def __getHalfBond__(self, atom_key: "int", bond_key: "int",
                        ordering) -> "float":
        """
        Returns the half-bond energy of a given bond for a certain atom.

        Args:
        atom_key (int) : Index of the atom of interest
        bond_key (int) : Index of the bond of interest

        Returns:
        float : The half-bond energy of that bond at that atom, in units of eV
        """

        atom1 = self.symbols[ordering[atom_key]]
        atom2 = self.symbols[ordering[self.adj_list[atom_key][bond_key]]]
        return self.coeffs[atom1][atom2][self.cns[atom_key]]

    def getAtomicCE(self, atom_key: "int", cn, ordering) -> "float":
        """
        Returns the sum of half-bond energies for a particular atom.

        Args:
        atom_key : Index of the atom of interest

        Returns:
        float : The sum of all half-bond energies at that atom, in units of eV
        """
        local_CE = 0
        atom1 = self.symbols[ordering[atom_key]]
        adjs = self.adj_list[atom_key]
        for bond_key in range(cn):
            atom2 = self.symbols[ordering[adjs[bond_key]]]
            local_CE += self.coeffs[atom1][atom2][cn]
            # local_CE += self.__getHalfBond__(atom_key, bond_key, ordering)
        return local_CE

    def getTotalCE(self, ordering) -> "float":
        """
        Returns the cohesive energy of the cluster as a whole.

        Returns
        float : The CE, in units of eV
        """
        total_energy = 0
        for atom, cn in enumerate(self.cns):
            total_energy += self.getAtomicCE(atom, cn, ordering)
        total_CE = total_energy / self.n_atoms
        return total_CE

    def get_total_ce2(self, ordering, bonds):
        total = 0
        for i1 in bonds:
            a1 = self.symbols[ordering[i1]]
            for i2 in bonds[i1]:
                a2 = self.symbols[ordering[i2]]
                total += self.coeffs[a1][a2][self.cns[i1]]
                total += self.coeffs[a2][a1][self.cns[i2]]
        return total / self.n_atoms

if __name__ == '__main__':
    from adjacency import buildAdjacencyList
    import ase.cluster
    import time
    import sys
    import matplotlib.pyplot as plt

    natoms = []
    old = []
    new = []
    for nshells in range(2, 41):
        atom = ase.cluster.Icosahedron('Cu', nshells)
        natoms.append(len(atom))
        a = buildAdjacencyList(atom)
        bonds = {}
        for i in range(len(a)):
            for con in a[i]:
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
                # key = '_'.join([str(z) for z in sorted([i, con])])
                # if key not in bonds:
                #    bonds[key] = [len(a[i]), len(a[con])]

        print('Total half bonds: %i' % sum(len(i) for i in a))
        print('Total full bonds: %i' % sum(len(bonds[j]) for j in bonds))

        x = AtomGraph(a, 'Cu', 'Au')

        od = [0] * len(atom)
        od[3] = 1

        nn = 5

        start1 = time.time()
        for y in range(nn):
            val1 = x.getTotalCE(od)
        tot1 = (time.time() - start1) / nn
        print('Old: %.3f s/run' % tot1)

        start2 = time.time()
        for z in range(nn):
            val2 = x.get_total_ce2(od, bonds)
        tot2 = (time.time() - start2) / nn
        print('New: %.3f s/run' % tot2)
        print('CE diff (New - Old) = %.9f eV' % (val2 - val1))
        old.append(tot1)
        new.append(tot2)

    plt.plot(natoms, old, label='Old')
    plt.plot(natoms, new, label='New')
    plt.xlabel('N Atoms')
    plt.ylabel('Runtime per CE calc (s)')
    plt.legend()
    plt.show()
