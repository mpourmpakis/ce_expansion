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
    unique_cns (np.array): array of unique coordination numbers of NP
    min_cn (int): minimum coordination number of NP
    max_cn (int): maximum coordination number of NP
    mono_ce0 (float): CE value for monometallic NP of "kind0"
    mono_ce1 (float): CE value for monometallic NP of "kind1"
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

        # precalculate some useful info on CNs
        self.unique_cns = np.unique(self.cns)
        self.min_cn = self.unique_cns.min()
        self.max_cn = self.unique_cns.max()

        # Set up the matrix of bond energies
        self._bond_energies = np.zeros((2, 2, 13), dtype=ctypes.c_double)
        self._p_bond_energies = None
        self.set_composition(kind0, kind1)

        # Create pointers
        self._long_num_atoms = ctypes.c_long(self.num_atoms)

        # Windows compatibility?
        self._p_cns = self.cns.ctypes.data_as(ctypes.POINTER(ctypes.c_long))

        self._long_num_bonds = ctypes.c_long(self._num_bonds)
        self._p_bond_list = self._bond_list.ctypes.data_as(
                                ctypes.POINTER(ctypes.c_long))

        # calculate monometallic CEs
        self.mono_ce0 = self.getTotalCE(np.zeros(self.num_atoms))
        self.mono_ce1 = self.getTotalCE(np.ones(self.num_atoms))

    def __len__(self):
        return self._num_bonds

    def __getitem__(self, i):
        """AtomGraph returns CN of given atom index"""
        return self.cns[i]

    def set_composition(self, kind0: "str", kind1: "str") -> "None":
        """
        Sets the bond energies to be passed to the C library. Energies come
        from the coeffs attribute.

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
        self._p_bond_energies = self._bond_energies.ctypes.data_as(
                                        ctypes.POINTER(ctypes.c_double))

    def calc_cn_dist(self, ordering: "np.array") -> "dict":
        """
        Gets CN distribution of each atom type
        - can be used to create bar plots of CN distribution

        Args:
        - ordering (np.array): chemical ordering of the NP

        Returns:
        - (dict): {cn_options (np.array): possible CN options,
                   m1_counts (np.array): metal1 CN counts,
                   m2_counts (np.array): metal2 CN counts,
                   tot_counts (np.array): total CN counts}
        """
        # create arrays of CNs occupied by each metal type
        metal1_cns = self.cns[np.where(ordering == 0)[0]]
        metal2_cns = self.cns[np.where(ordering == 1)[0]]

        # counts of CN type for each metal
        m1_counts = np.array([(metal1_cns == cn).sum()
                              for cn in self.unique_cns])
        m2_counts = np.array([(metal2_cns == cn).sum()
                              for cn in self.unique_cns])

        # total counts of CN type
        tot_counts = m1_counts + m2_counts

        return dict(cn_options=self.unique_cns,
                    m1_counts=m1_counts,
                    m2_counts=m2_counts,
                    tot_counts=tot_counts)

    def countMixing(self, ordering: "np.array",
                    holder_array: "np.array" = None) -> "np.array":
        """
        Determines the number of homo/hetero-atomic bonds in the system.

        Args:
        - ordering (np.array): Chemcial ordering of the NP
        - holder_array (np.array): A length-3 array to hold the result.
                                   Optional. If not supplied, will create
                                   the array on the spot. May be slower to do
                                   this. This array is OVER-WRITTEN by the
                                   C-library, and contains long ints.
        Returns:
            Bond counts(np.array): [A-A bonds, B-B bonds, A-B bonds]
        """
        ordering = ordering.astype(ctypes.c_long)
        p_ordering = ordering.ctypes.data_as(ctypes.POINTER(ctypes.c_long))

        if holder_array is None:
            holder_array = np.zeros(3, dtype=ctypes.c_long)

        p_holder_array = holder_array.ctypes.data_as(
                            ctypes.POINTER(ctypes.c_long))

        interface.pointerized_calculate_mixing(self._long_num_atoms,
                                               self._long_num_bonds,
                                               self._p_bond_list,
                                               p_ordering,
                                               p_holder_array)

        return holder_array

    def calcMixing(self, ordering: "np.array",
                   holder_array: "np.array" = None) -> "np.array":
        """
        Calculates the mixing parameter. From Batista et al. Adsorption of CO, NO, and H2 on the
        PdnAu55-n nanoclusters: A Density Functional Theory Investigation within the van der
        Waals D3 Corrections. J Phys Chem C 2019, 123 (12) 7431-7439.
        Link here: https://pubs.acs.org/doi/10.1021/acs.jpcc.8b12219
        
        For bimetallics, this mixing parameter is the percent of homo-atomic 
        bonds scaled between -1 and 1.

        Args:
        ordering(np.array): Chemcial ordering of the NP
        holder_array(np.array): A length-3 array to hold the result of countMixing. 
                                Optional. If not supplied, will create the array on the spot. 
                                May be slower to do this. This array is OVER-WRITTEN by the 
                                C-library, and contains long ints.

        Returns:
            The mixing parameter. +1 indicates complete segregation (e.g. only homo-atomic bonds).
            -1 indicates complete mixing (e.g. zero bonds between the same element).
        """
        if holder_array is None:
            holder_array = np.zeros(3, dtype=ctypes.c_long)
        self.countMixing(ordering, holder_array)
        
        numerator = (holder_array[0] + holder_array[1]) - holder_array[2]
        denominator = (holder_array[0] + holder_array[1]) + holder_array[2]
        
        mixing_parameter = numerator / denominator

        return mixing_parameter

    def getTotalCE(self, ordering: "np.array") -> "float":
        """
        Calculates the cohesive energy of the NP using the BC model,
        as implemented in interface.py and lib.c

        Args:
            ordering (np.array): Chemical ordering of the NP.
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

    def getEE(self, ordering: "np.array") -> "float":
        """
        Calculates the excess energy in eV / atom of a given ordering

        Args:
        ordering (np.array): 1D chemical ordering array

        Returns:
        (float): Excess Energy (EE) in eV / atom
        """
        # number and concentration of "kind1"
        n1 = ordering.sum()
        x1 = n1 / self.num_atoms

        # number and concentration of "kind0"
        n0 = self.num_atoms - n1
        x0 = n0 / self.num_atoms

        # excess energy
        ee = self.getTotalCE(ordering) - \
            (x0 * self.mono_ce0 + x1 * self.mono_ce1)
        return ee

    def get_adjacency_list(self):
        '''
        Calculates an adjacency list given the bonds list.

        Returns:
        The NxM adjacency list represented by the bonds list
        '''
        adjacency_list = [[]] * self.num_atoms
        for bond in self._bond_list:
            adjacency_list[bond[0]] = adjacency_list[bond[0]] + [bond[1]]
        return adjacency_list

    def metropolis(self, ordering,
                   num_steps=1000,
                   swap_any=False):
        '''
        Metropolis-Hastings-based exploration of similar NPs

        Args:
        atomgraph (atomgraph.AtomGraph) : An atomgraph representing the NP
        ordering (np.array) : 1D chemical ordering array
        num_steps (int) : How many steps to simulate for
        swap_any (bool) : Determines whether to restrict the algorithm's swaps
                          to only atoms directly bound to  the atom of interest.
                          If set to 'True', the algorithm chooses any two atoms
                          in the NP regardless of where they are. Selecting
                          'False' yields a slightly-more-physical case of
                          atomic diffusion.

        '''
        # Initialization
        # create new instance of ordering array
        ordering = ordering.copy()
        best_ordering = ordering.copy()
        best_energy = self.getTotalCE(ordering)
        prev_energy = best_energy
        energy_history = np.zeros(num_steps)
        energy_history[0] = best_energy
        if not swap_any:
            adj_list = self.get_adjacency_list()
        for step in range(1, num_steps):
            # Determine where the ones and zeroes are currently
            ones = np.where(ordering == 1)[0]
            zeros = np.where(ordering == 0)[0]

            # Choose a random step
            if swap_any:
                chosen_one = np.random.choice(ones)
                chosen_zero = np.random.choice(zeros)
            else:
                # Search the NP for a 1 with heteroatomic bonds
                for chosen_one in np.random.permutation(ones):
                    connected_atoms = adj_list[chosen_one]
                    connected_zeros = np.intersect1d(connected_atoms, zeros,
                                                     assume_unique=True)
                    if connected_zeros.size != 0:
                        # The atom has zeros connected to it
                        chosen_zero = np.random.choice(connected_zeros)
                        break

            # Evaluate the energy change
            prev_ordering = ordering.copy()
            ordering[chosen_one] = 0
            ordering[chosen_zero] = 1
            energy = self.getTotalCE(ordering)

            # Metropolis-related stuff
            ratio = energy / prev_energy
            if ratio > np.random.uniform():
                # Commit to the step
                energy_history[step] = energy
                if energy < best_energy:
                    best_energy = energy
                    best_ordering = ordering.copy()
            else:
                # Reject the step
                ordering = prev_ordering.copy()
                energy_history[step] = prev_energy

        return best_ordering, best_energy, energy_history


if __name__ == '__main__':
    import ase.cluster
    import adjacency
    import matplotlib.pyplot as plt

    # Create a nanoparticle and its graph object
    nanoparticle = ase.cluster.Icosahedron('Cu', 3)
    bond_list = adjacency.buildBondsList(nanoparticle)

    graph = AtomGraph(bond_list, 'Cu', 'Au')

    # Generate the chemical ordering
    np.random.seed(12345)
    chemical_ordering = np.random.choice([0, 1], size=graph.num_atoms)

    # Calculate cohesive energy
    cohesive_energy = graph.getTotalCE(chemical_ordering)
    print('Cohesive energy = %.2e' % cohesive_energy)

    mixing = graph.countMixing(chemical_ordering)
    print("[A-A   A-B   B-B] =", mixing)
    print("Sum of homo/hetero-atomic bonds: ", sum(mixing))
    num_bonds = len(bond_list) // 2
    print("True bondcount: ", num_bonds)
    mixing_parameter = graph.calcMixing(chemical_ordering)
    print("Mixing Parameter: ", mixing_parameter)

    # Enter global metropolis
    opt_order, opt_energy, energy_history = graph.metropolis(chemical_ordering, num_steps=1000, swap_any=True)
    print("Performed 1000 metropolis steps swapping anywhere, yielding CE = %.2e" % opt_energy)
    plt.plot(energy_history, label="Swap_Global")

    # Enter locally-swapped metropolis
    opt_order, opt_energy, energy_history = graph.metropolis(chemical_ordering, num_steps=1000, swap_any=False)
    print("Performed 1000 metropolis steps swapping across bonds, yielding CE = %.2e" % opt_energy)
    plt.plot(energy_history, label="Swap_Local")
    plt.legend()
    plt.xlabel("Step")
    plt.ylabel("Energy (eV)")
    plt.show()
    # Exeunt metropoles
