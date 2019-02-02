#!/usr/bin/env python3

import pickle
import ctypes
import interface
import numpy as np

DEFAULT_ELEMENTS = ("Cu", "Cu")
DEFAULT_RADIUS = 2.8
with open("../data/precalc_coeffs.pickle", "rb") as precalcs:
    DEFAULT_BOND_COEFFS = pickle.load(precalcs)


class c_AtomGraph(object):
    def __init__(self, bond_list: "np.array",
                 kind0: "str",
                 kind1: "str",
                 coeffs: "dict" = DEFAULT_BOND_COEFFS):
        """
        A graph representing the ASE Atoms object to be investigated. This graph is represented as
        an Nx2 list of edges in the nanoparticle. Note that the function signature is slightly
        different than AtomGraph.

        Args:
        bond_list (np.array): An Nx2 numpy array containing a list of bonds in the nanoparticle.
                              Zero indexed. Assumed to come from adjacency.buildBondList.
        kind0 (str): A string indicating the atomic symbol a "0" represents.
        kind1 (str): A string indicating the atomic symbol a "1" represents.
        coeffs (dict): A nested dictionary of bond coefficients, of the format dict[source][destination][CN]. Source and
            destination are strings indicating the element of interest. Defaults to the global DFTAULT_BOND_COEFFS.
            Coefficients can be calculated using /tools/gen_coeffs.py.

        Attributes:
        symbols (tuple): A tuple containing the compositional information of the NP. Index 0 is the element a "0"
            represents. Index 1 is the element a "1" represents. This attribute is updated whenever the NP's composition
            is updated via the "set_composition" method.
        coeffs (dict): The nested dictionary passed to the constructor as "coeffs"
        n_atoms (int): The number of atoms in the NP.
        cns (np.array): An array containing the coordination number of each atom.
        """
        self._bond_list = bond_list
        self._num_bonds = len(bond_list)
        self.symbols = (None, None)
        self.coeffs = coeffs

        self._bond_energies = np.zeros((2, 2, 13), dtype=np.float64)
        self.set_composition(kind0, kind1)

        self.n_atoms = len(set(bond_list[:, 0]))
        self.cns = np.bincount(bond_list[:, 0])

        self._bond_energies = self.set_composition(kind0, kind1)

        # Create pointers
        self._long_num_atoms = ctypes.c_long(self.n_atoms)
        self._p_cns = self.cns.ctypes.data_as(ctypes.POINTER(ctypes.c_long))
        self._long_num_bonds = ctypes.c_long(self._num_bonds)
        self._p_bond_list = self._bond_list.ctypes.data_as(ctypes.POINTER(ctypes.c_long))

    def __len__(self):
        return self.num_bonds

    def set_composition(self, kind0: "str", kind1: "str"):
        """
        Sets the bond energies to be passed to the C library. Energies come from the coeffs
        attribute.

        Args:
        kind0 (str): The element a "0" represents.
        kind1 (str): The element a "1" represents.
        """
        if (kind0, kind1) == self.symbols:
            return self._bond_energies
        else:
            self.symbols = (kind0, kind1)
            for i, element1 in enumerate(self._kinds):
                for j, element2 in enumerate(self._kinds):
                    for cn in range(0, 13):
                        coefficient = self.coeffs[element1][element2][cn]
                        self._bond_energies[i][j][cn] = coefficient

        # Create pointer
        self._p_bond_energies = self._bond_energies.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    def getTotalCE(self, ordering: "np.narray"):
        """
        Calculates the cohesive energy of the NP using the BC model, as implemented in interface.py
        and lib.c
        """
        # Pointerize ordering
        p_ordering = ordering.ctypes.data_as(ctypes.POINTER(ctypes.c_long))
        return interface.pointerized_calculate_ce(self._p_bond_energies,
                                                  self._long_num_atoms,
                                                  self._p_cns,
                                                  self._long_num_bonds,
                                                  self._p_bond_list,
                                                  p_ordering)



class AtomGraph(object):
    def __init__(self, adj_list: "list",
                 kind0: "str",
                 kind1: "str",
                 coeffs: "dict" = DEFAULT_BOND_COEFFS):
        """
        A graph representing the ase Atoms object to be investigated. First axis is the atom index, second axis contains
        bonding information.
        First entry of the second axis corresponds to a 1/0 representing the atomic kind
        Second entry of the second axis corresponds to the indices of the atoms that atom is bound to

        Args:
        adj_list (list) : A numpy array containing adjacency information. Assumed to be produced by the
        buildAdjacencyList function
        kind0 (str) : Atomic symbol indicating what a "0" in ordering means
        kind1 (str) : Atomic symbol indicating what a "1" in ordering means
        coeffs (dict) : A dictionary of the various bond coefficients we have precalculated, using coeffs.py. Defaults
                        to the global DEFAULT_BOND_COEFFS.


        Attributes:
        adj_list (list) : A numpy array containing adjacency information
        symbols (tuple) : Atomic symbols indicating what the binary representations of the elements in ordering
                        means

        """

        self.adj_list = adj_list
        self.cns = [len(a) for a in adj_list]
        self.coeffs = coeffs
        self.symbols = (kind0, kind1)

        # total number of atoms
        self.n_atoms = len(adj_list)
        self.bonds = buildBonds(adj_list)

    def __len__(self):
        return len(self.adj_list)

    def getCN(self, atom_key: "int") -> "int":
        """
        Returns the coordination number of the given atom.

        Args:
        atom_key (int) : Index of the atom of interest
        """
        return self.adj_list[atom_key].size

    def getAllCNs(self) -> "list":
        """
        Returns a numpy array containing all CN's in the cluster.
        """
        return [len(entry) for entry in self.adj_list]

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

    def get_total_ce2(self, ordering):
        total = 0
        for i1 in self.bonds:
            a1 = self.symbols[ordering[i1]]
            for i2 in self.bonds[i1]:
                a2 = self.symbols[ordering[i2]]
                total += self.coeffs[a1][a2][self.cns[i1]]
                total += self.coeffs[a2][a1][self.cns[i2]]
        return total / self.n_atoms


def buildBonds(adjacency_list: "list") -> "dict":
    """
      Adjacency *bond* list representation based on "buildAdjacencyList"

      Args:
      adjacency_list (list): A list that is outputted from
                               adjacency.buildAdjacencyList

      Returns:
      dict : A dictonary with an atom index as a key and a list of atoms it's
               bonded to as the value
    """

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
    from adjacency import buildAdjacencyList
    import ase.cluster

    atom = ase.cluster.Icosahedron('Cu', 10)

    a = buildAdjacencyList(atom)

    x = AtomGraph(a, 'Cu', 'Au')
    ordering = [0] * x.n_atoms
    ordering[7] = 1
    ce1 = x.getTotalCE(ordering)
    ce2 = x.get_total_ce2(ordering)
    print('%.2e' % abs(ce1 - ce2))
