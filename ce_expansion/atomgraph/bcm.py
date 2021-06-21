from ce_expansion.data.gamma import GammaValues
import numpy as np
import itertools
from ce_expansion.data.gamma import GammaValues
import collections.abc
import ase.units


def recursive_update(d, u):
    #Function found by team to update Gamma and Bulk Values
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = recursive_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


class BCModel:
    def __init__(self, atoms, bond_list, metal_types=None):
        """
        Based on metal_types, create ce_bulk and gamma dicts from the data given

        Args:
            atoms: ASE atoms object which contains the data of the NP being tested
            bond_list: a CSV containing the bond data
        KArgs:
            metal_types: List of metals found within the nano-particle
        """

        self.atoms = atoms
        self.bond_list = bond_list

        if metal_types is None:
            # get metal_types from atoms object
            self.metal_types = sorted(set(atoms.symbols))
        else:
            # ensure metal_types to unique, sorted list of metals
            self.metal_types = sorted(set(metal_types))

        self.metal_pairs = list(itertools.combinations_with_replacement(
                                    self.metal_types, 2))

        # creating gamma list for every possible atom pairing
        self.gamma = None
        self.ce_bulk = None
        self._get_bcm_params()

        # Calculate and set the precomps matrix
        self.precomps = None
        self.precomps = self._calc_precomps()

    def __len__(self):
        return len(self.atoms)

    def _get_bcm_params(self):
        """
            Creates gamma and ce_bulk dictionaries which are then used to created precomputed values for the BCM calculation

        Sets:   Gamma: Weighting factors of the computed elements within the BCM
                ce_bulk: Bulk Cohesive energy values

        """
        gamma = {}
        ce_bulk = {}
        for item in self.metal_pairs:
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

    def _calc_precomps(self):
        """
            Uses the Gamma and ce_bulk dictionaries to create a precomputed BCM matrix of gammas and ce_bulk values

            [precomps] = [gamma of element 1] * [ce_bulk of element 1 to element 2]

        Returns: Precomp Matrix

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
        return precomps

    def calc_ce(self, orderings):
        """
        Calculates the Cohesive energy of the ordering given or of the default ordering of the NP

        [Cohesive Energy] = ( [precomp values of element A and B] / sqrt(12 * CN) ) / [num atoms]

        Args:
            orderings: The ordering of atoms within the NP; ordering key is based on Metals in alphabetical order
                - Will use ardering defined by atom if not given an ordering

        Returns: Cohesive Energy

        """

        # determine CN Values
        (atom_CN, bond_CN) = np.hsplit(self.bond_list, 2)
        (CN_Vals, CN) = np.unique(bond_CN, return_counts=True)

        a1 = self.bond_list[:, 0]
        a2 = self.bond_list[:, 1]

        # creating bond orderings
        return (self.precomps[orderings[a1], orderings[a2]] / np.sqrt(12 * CN[a1])).sum() / len(self.atoms)

    def calc_ee(self, orderings=None):
        """
            Calculates the Excess energy of the ordering given or of the default ordering of the NP

            [Excess Energy] = [CE of NP] - sum([Pure Element NP] * [Comp of Element in NP])

        Args:
            orderings: The ordering of atoms within the NP; ordering key is based on Metals in alphabetical order
                - Will use ardering defined by atom if not given an ordering

        Returns: Excess Energy

        """

        # If orderings = none, ensure tested atoms object contains desired orderings to test
        if orderings is None:
            orderings = [test_of_bcm.metal_types.index(x) for x in atoms.symbols]

        metals = np.bincount(orderings)

        #Obtain atom fractions of each tested element
        x_i =  metals / len(orderings)

        #Calculate energy of tested NP first;
        ee = self.calc_ce(orderings)

       # Then, subtract calculated pure NP energies multiplied by respective fractions to get Excess Energy
        for ele in range(len(metals)):
            x_ele = x_i[ele]
            o_mono_x = np.ones(len(self), int) * ele

            ee -= self.calc_ce(o_mono_x) * x_ele
        return(ee)

    def calc_smix(self, orderings=None):
        """
        compositions = np.bincount(ordering) / len(ordering)

        kb = ase.units.Kb
        smix = -kb * sum(xi * ln(xi) for xi in compositions)
        sum(compositions) == 1

        Boltzmann constant

        smix units = eV / (K * atom)

        Returns: entropy of mixing (smix)

        """
        # If orderings = none, ensure tested atoms object contains desired orderings to test
        if orderings is None:
            orderings = [test_of_bcm.metal_types.index(x) for x in atoms.symbols]

        x_i = np.bincount(orderings) / len(orderings)

        # drop 0s to avoid errors
        x_i = x_i[x_i != 0]

        kb = ase.units.kB

        smix = -kb * sum(x_i * np.log(x_i))

        return smix

    def calc_gmix(self, orderings, T=298):
        """
        gmix = self.ee - T * self.calc_smix(ordering)
        Args:
            T: Temperature of the system in Kelvin; Defaults at room temp of 25 C

        Returns: free energy of mixing (gmix)

        """

        gmix = self.calc_ee(orderings) - T * self.calc_smix(orderings)

        return gmix

