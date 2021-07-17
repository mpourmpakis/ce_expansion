import itertools
import collections.abc
from typing import Iterable

import numpy as np
import ase
import ase.units

from ce_expansion.atomgraph import adjacency
from ce_expansion.data.gamma import GammaValues


def recursive_update(d: dict, u: dict) -> dict:
    """
    recursively updates 'dict of dicts'
    Ex)
    d = {0: {1: 2}}
    u = {0: {3: 4}, 8: 9}

    recursive_update(d, u) == {0: {1: 2, 3: 4}, 8: 9}

    Args:
    d (dict): the nested dict object to update
    u (dict): the nested dict that contains new key-value pairs

    Returns:
    d (dict): the final updated dict
    """
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = recursive_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


class BCModel:
    def __init__(self, atoms: ase.Atoms, metal_types: Iterable = None,
                 bond_list: Iterable = None):
        """
        Based on metal_types, create ce_bulk and gamma dicts from data given

        Args:
        atoms: ASE atoms object which contains the data of the NP being tested
        bond_list: list of atom indices involved in each bond

        KArgs:
        metal_types: List of metals found within the nano-particle
                     If not passed, use elements provided by the atoms object
        """
        self.atoms = atoms.copy()
        self.atoms.pbc = False

        if metal_types is None:
            # get metal_types from atoms object
            self.metal_types = sorted(set(atoms.symbols))
        else:
            # ensure metal_types to unique, sorted list of metals
            self.metal_types = sorted(set(m.title() for m in metal_types))

        self.bond_list = bond_list
        if self.bond_list is None:
            self.bond_list = adjacency.build_bonds_arr(self.atoms)

        self.cn = np.bincount(self.bond_list[:, 0])

        # creating gamma list for every possible atom pairing
        self.gamma = None
        self.ce_bulk = None
        self._get_bcm_params()

        # get bonded atom columns
        self.a1 = self.bond_list[:, 0]
        self.a2 = self.bond_list[:, 1]

        # Calculate and set the precomps matrix
        self.precomps = None
        self._get_precomps()
        self.cn_precomps = np.sqrt(self.cn * 12)[self.a1]

    def __len__(self) -> int:
        return len(self.atoms)

    def _get_bcm_params(self):
        """
        Creates gamma and ce_bulk dictionaries which are then used
        to created precomputed values for the BCM calculation

        Sets:
        gamma: Weighting factors of the computed elements within the BCM
        ce_bulk: Bulk Cohesive energy values
        """
        gamma = {}
        ce_bulk = {}
        for item in itertools.combinations_with_replacement(self.metal_types, 2):
            # Casting metals and setting keys for dictionary
            metal_1, metal_2 = item

            gamma_obj = GammaValues(metal_1, metal_2)

            # using Update function to create clean Gamma an bulk dictionaries
            gamma = recursive_update(gamma, gamma_obj.gamma)
            # add ce_bulk vals
            ce_bulk[gamma_obj.element_a] = gamma_obj.ce_a
            ce_bulk[gamma_obj.element_b] = gamma_obj.ce_b

        self.ce_bulk = ce_bulk
        self.gammas = gamma

    def _get_precomps(self):
        """
        Uses the Gamma and ce_bulk dictionaries to create a precomputed
        BCM matrix of gammas and ce_bulk values

        [precomps] = [gamma of element 1] * [ce_bulk of element 1 to element 2]

        Sets:
        precomps: Precomp Matrix
        """
        # precompute values for BCM calc
        n_met = len(self.metal_types)

        precomps = np.ones((n_met, n_met))

        for i in range(n_met):
            for j in range(n_met):

                M1 = self.metal_types[i]
                M2 = self.metal_types[j]
                precomp_bulk = self.ce_bulk[M1]
                precomp_gamma = self.gammas[M1][M2]

                precomps[i, j] = precomp_gamma * precomp_bulk
        self.precomps = precomps

    def calc_ce(self, orderings: np.ndarray) -> float:
        """
        Calculates the Cohesive energy (in eV / atom) of the ordering given or of the default ordering of the NP

        [Cohesive Energy] = ( [precomp values of element A and B] / sqrt(12 * CN) ) / [num atoms]

        Args:
        orderings: The ordering of atoms within the NP; ordering key is based on Metals in alphabetical order

        Returns:
        Cohesive Energy (eV / atom)
        """
        return (self.precomps[orderings[self.a1], orderings[self.a2]] / self.cn_precomps).sum() / len(self.atoms)

    def calc_ee(self, orderings: np.ndarray) -> float:
        """
        Calculates the Excess energy (in eV / atom) of the ordering given or of the default ordering of the NP

        [Excess Energy] = [CE of NP] - sum([Pure Element NP] * [Comp of Element in NP])

        Args:
        orderings: The ordering of atoms within the NP; ordering key is based on Metals in alphabetical order

        Returns:
        Excess Energy (eV / atom)
        """

        metals = np.bincount(orderings)

        # obtain atom fractions of each tested element
        x_i = np.zeros(len(self.metal_types)).astype(float)
        x_i[:len(metals)] = metals / metals.sum()

        # calculate energy of tested NP first;
        ee = self.calc_ce(orderings)

        # Then, subtract calculated pure NP energies multiplied by respective
        # fractions to get Excess Energy
        for ele in range(len(self.metal_types)):
            x_ele = x_i[ele]
            o_mono_x = np.ones(len(self), int) * ele

            ee -= self.calc_ce(o_mono_x) * x_ele
        return ee

    def calc_smix(self, orderings: np.ndarray) -> float:
        """
        Uses boltzman constant, orderings, and element compositions to determine the smix of the nanoparticle

        Args:
        orderings: The ordering of atoms within the NP; ordering key is based on Metals in alphabetical order

        Returns:
        entropy of mixing (smix)

        """

        x_i = np.bincount(orderings) / len(orderings)

        # drop 0s to avoid errors
        x_i = x_i[x_i != 0]

        kb = ase.units.kB

        smix = -kb * sum(x_i * np.log(x_i))

        return smix

    def calc_gmix(self, orderings: np.ndarray, T: float = 298.15):
        """
        gmix (eV / atom) = self.ee - T * self.calc_smix(ordering)

        Args:
        T: Temperature of the system in Kelvin; Defaults at room temp of 25 C
        orderings: The ordering of atoms within the NP; ordering key is based on Metals in alphabetical order

        Returns:
        free energy of mixing (gmix)
        """
        return self.calc_ee(orderings) - T * self.calc_smix(orderings)

    def metropolis(self, ordering: np.ndarray, num_steps: int = 1000):
        """
        Metropolis-Hastings-based exploration of similar NPs

        Args:
        ordering: 1D chemical ordering array
        num_steps: How many steps to simulate for
        """
        # Initialization
        # create new instance of ordering array
        ordering = ordering.copy()
        best_ordering = ordering.copy()
        best_energy = self.calc_ce(ordering)
        prev_energy = best_energy
        energy_history = np.zeros(num_steps)
        energy_history[0] = best_energy

        ordering_indices = np.arange(len(ordering))
        for step in range(1, num_steps):
            prev_ordering = ordering.copy()
            i, j = np.random.choice(ordering_indices, 2, replace=False)
            ordering[i], ordering[j] = ordering[j], ordering[i]

            # Evaluate the energy change
            energy = self.calc_ce(ordering)

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
