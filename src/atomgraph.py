#!/usr/bin/env python3

import ctypes

import numpy as np

import interface
from npdb import db_inter


class AtomGraph(object):
    """
    A graph representing the ASE Atoms object to be investigated.
    This graph is represented as an Nx2 list of edges in the nanoparticle.
    Note that the function signature is slightly different than AtomGraph.

    Args:
    bond_list (np.array): An Nx2 numpy array containing a list of bonds in the
                          nanoparticle. Zero indexed. Assumed to come from
                          adjacency.buildBondList.
    kind0 (str): A string indicating the atomic symbol a "0" represents.
    kind1 (str): A string indicating the atomic symbol a "1" represents.

    Attributes:
    symbols (tuple): A tuple containing the compositional information of the
                     NP. Index 0 is the element a "0" represents. Index 1 is
                     the element a "1" represents. This attribute is updated
                     whenever the NP's composition is updated via the
                     "set_composition" method.
    coeffs (dict): A nested dictionary of bond coefficients, of the format
                   dict[source][destination][CN]. Source and destination are
                   strings indicating the element of interest. Defaults to the
                   global DFTAULT_BOND_COEFFS. Coefficients can be calculated
                   using /tools/gen_coeffs.py.
    num_atoms (int): The number of atoms in the NP.
    cns (np.array): An array containing the coordination number of each atom.
    """

    def __init__(self, bond_list: "np.array", kind0: "str", kind1: "str"):

        self._bond_list = bond_list.astype(ctypes.c_long)
        self._num_bonds = len(bond_list)

        # Public attributes
        self.symbols = (None, None)
        self.coeffs = db_inter.build_coefficient_dict(kind0 + kind1)
        self.num_atoms = len(set(bond_list[:, 0]))

        self.cns = np.bincount(bond_list[:, 0])
        self.cns = self.cns.astype(ctypes.c_long)

        # Set up the matrix of bond energies
        self._bond_energies = np.zeros((2, 2, 13), dtype=ctypes.c_double)
        self._p_bond_energies = None
        self.set_composition(kind0, kind1)

        # Create pointers
        self._long_num_atoms = ctypes.c_long(self.num_atoms)
        self._p_cns = self.cns.ctypes.data_as(ctypes.POINTER(ctypes.c_long))  # Windows compatibility?
        self._long_num_bonds = ctypes.c_long(self._num_bonds)
        self._p_bond_list = self._bond_list.ctypes.data_as(ctypes.POINTER(ctypes.c_long))

    def __len__(self):
        return self._num_bonds

    def set_composition(self, kind0: "str", kind1: "str") -> "None":
        """
        Sets the bond energies to be passed to the C library. Energies come from the coeffs
        attribute.

        Args:
        kind0 (str): The element a "0" represents.
        kind1 (str): The element a "1" represents.
        """
        if (kind0, kind1) == self.symbols:
            pass
        else:
            self.symbols = (kind0, kind1)
            for i, element1 in enumerate(self.symbols):
                for j, element2 in enumerate(self.symbols):
                    for cn in range(0, 13):
                        coefficient = self.coeffs[element1][element2][cn]
                        self._bond_energies[i][j][cn] = coefficient

        # Create pointer
        self._p_bond_energies = self._bond_energies.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    def getTotalCE(self, ordering: "np.array") -> "float":
        """
        Calculates the cohesive energy of the NP using the BC model,
        as implemented in interface.py and lib.c
        """
        ordering = ordering.astype(ctypes.c_long)
        # Pointerize ordering
        p_ordering = ordering.ctypes.data_as(ctypes.POINTER(ctypes.c_long))
        return interface.pointerized_calculate_ce(self._p_bond_energies,
                                                  self._long_num_atoms,
                                                  self._p_cns,
                                                  self._long_num_bonds,
                                                  self._p_bond_list,
                                                  p_ordering)

    def get_adjacency_list(self):
        '''
        Calculates an adjacency list given the bonds list.

        Returns:
        The NxM adjacency list represented by the bonds list
        '''
        adjacency_list = [[]] * self.num_atoms
        for bond in self._bond_list:
            adjacency_list[bond[0]] =  adjacency_list[bond[0]] + [bond[1]]
        return adjacency_list

    def metropolis_movement(self, initial_ordering,
                             num_steps=1000, accept_move_rate=0.8):
        '''
        Metropolis-Hastings-based simulation of atomic diffusion in the NP

        Args:
        atomgraph (atomgraph.AtomGraph) : An atomgraph representing the NP
        initial_ordering (np.array) : 1D chemical ordering array
        kinds (tuple) : Tuple. Index0 is the element a 0 represents. Same for 1 and Index1.
        num_steps (int) : How many steps to simulate for
        accept_good_rate (float) : How often a step is accepted. 0=0%, 1=100%.

        '''

        step = 0
        num_rerolls = 0
        adj_list = self.get_adjacency_list()
        ordering = initial_ordering
        best_ordering = ordering
        energy = self.getTotalCE(ordering)
        best_energy = energy
        energy_history = np.zeros(num_steps)
        while step < num_steps:
            prev_ordering = ordering
            prev_energy = energy
            # Choose random atom in the NP
            moved_atom = np.random.choice(range(0, len(initial_ordering)))

            # Choose the atom to swap with
            destination = np.random.choice(adj_list[moved_atom])

            # Swap them
            if ordering[destination] == ordering[moved_atom]:
                # They are the same
                num_rerolls += 1
                if num_rerolls < 100:
                    # To make the simulation go faster, try to force it to swap
                    # If no swaps happen after a while, let it be
                    continue
                else:
                    energy_history[step] = prev_energy
                    step += 1

            else:
                # They are different
                num_rerolls = 0
                # 1-x flips between 0 and 1
                ordering[destination] = 1 - ordering[destination]
                ordering[moved_atom] = 1 - ordering[moved_atom]

                # Check the energy
                energy = self.getTotalCE(ordering)
                dice_roll = np.random.uniform()
                if dice_roll < accept_move_rate:
                    # Step is accepted
                    energy_history[step] = energy
                    if energy < best_energy:
                      best_energy = energy
                      best_ordering = ordering
                else:
                    # Step is declined
                    energy_history[step] = prev_energy
                    energy = prev_energy
                    ordering = prev_ordering
                step += 1
        return best_ordering, best_energy, energy_history


if __name__ == '__main__':
    import ase.cluster
    import adjacency

    # Create a nanoparticle and its graph object
    nanoparticle = ase.cluster.Icosahedron('Cu', 2)
    bond_list = adjacency.buildBondsList(nanoparticle)
    graph = AtomGraph(bond_list, 'Cu', 'Au')

    # Generate the chemical ordering
    np.random.seed(12345)
    chemical_ordering = np.random.choice([0, 1], size=graph.num_atoms)

    # Calculate cohesive energy
    cohesive_energy = graph.getTotalCE(chemical_ordering)
    print('Cohesive energy = %.2e' % cohesive_energy)
