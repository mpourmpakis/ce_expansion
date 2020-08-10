#!/usr/bin/env python
"""
Code dealing with the gamma coefficients
"""
import os

import numpy as np
import pandas as pd


class GammaValues(object):
    def __init__(self, element_a, element_b, bde_aa=None, bde_ab=None, bde_bb=None, ce_a=None, ce_b=None,
                 cnbulk_a=None, cnbulk_b=None, cn_max=None):
        """
        An object providing a convenient model for calculating and interacting with gamma values defined by the BCM.
        Args:
            element_a: Atomic symbol representing atom A
            element_b: Atomic symbol representing atom B
            bde_aa: Gas-phase homolytic bond dissociation energy of an A-A bond
            bde_ab: Gas-phase homolytic bond dissociation energy of an A-B bond
            bde_bb: Gas-phase homolytic bond dissociation energy of a B-B bond
            ce_a: Bulk cohesive energy of atom A
            ce_b: Bulk cohesive energy of atom B
            cnbulk_a: Bulk coordination number of atom A
            cnbulk_b: Bulk coordination number of atom B
        """
        # Default locations to search for values
        self.datafile_cn = os.path.join(os.path.dirname(__file__), "cndata.csv")
        self.datafile_ce = os.path.join(os.path.dirname(__file__), "bulkdata.csv")
        self.datafile_experimental_hbe = os.path.join(os.path.dirname(__file__), "experimental_hbe.csv")
        self.datafile_theoretical_hbe = os.path.join(os.path.dirname(__file__), "estimated_hbe.csv")

        self.element_a = element_a
        self.element_b = element_b

        # Look up CE if we have to
        if ce_a is None:
            self.ce_a = self.lookup_cohesive_energy(element_a)
        else:
            self.ce_a = ce_a
        if ce_b is None:
            self.ce_b = self.lookup_cohesive_energy(element_b)
        else:
            self.ce_b = ce_b

        # Look up BDE's if we have to
        bde_dict = {element_a: {element_a: bde_aa,
                                element_b: bde_ab},
                    element_b: {element_a: bde_ab,
                                element_b: bde_bb}
                    }
        for a, b in ((element_a, element_a), (element_a, element_b), (element_b, element_b)):
            if bde_dict[a][b] is None:
                bde_dict[a][b] = self.lookup_bde(a, b)
        self.bde_aa = bde_dict[element_a][element_a]
        self.bde_ab = bde_dict[element_a][element_b]
        self.bde_bb = bde_dict[element_b][element_b]

        # Look up bulk CNs if we have to
        if cnbulk_a is None:
            self.cnbulk_a = self.lookup_bulk_coordination(element_a)
        else:
            self.cnbulk_a = cnbulk_a
        if cnbulk_b is None:
            self.cnbulk_b = self.lookup_bulk_coordination(element_b)
        else:
            self.cnbulk_b = cnbulk_b

        # Figure out what CN_Max should be, if needed
        if cn_max is None:
            self.cn_max = max(self.cnbulk_a, self.cnbulk_b) + 1
        else:
            self.cn_max = cn_max

        # Calculate gammas
        ab_gamma, ba_gamma = self.calculate_gamma(element_a, element_b)
        self.gamma = {element_a: {element_a: 1,
                                  element_b: ab_gamma},
                      element_b: {element_a: ba_gamma,
                                  element_b: 1}
                      }

    def lookup_bde(self, a, b):
        """
        Looks up the heterolytic bond dissociation energy. Looks first in self.datafile_experimental_hbe. If the value
        is not found, looks in self.datafile_theoretical_hbe. If the value is still not found, raises a ValueError.
        Sign convention: Negative is more favorable. Values should be in eV.
        Args:
            a: Element A of the A-B bond
            b: Element B of the A-B bond

        Returns:

        """
        # First, try experimental data
        bde_csv = pd.read_csv(self.datafile_experimental_hbe, sep=",")
        bde = bde_csv[bde_csv.atom1 == a][b].iat[0]
        if np.isnan(bde):
            fallback_csv = pd.read_csv(self.datafile_theoretical_hbe, sep=",")
            fallback_bde = fallback_csv[bde_csv.atom1 == a][b].iat[0]
            if np.isnan(fallback_bde):
                raise ValueError(
                    f"Binding energy for {a} and {b} not found in" +
                    f"{self.datafile_experimental_hbe} or {self.datafile_theoretical_hbe}")
            else:
                result = fallback_bde
        else:
            result = bde
        return result

    def lookup_cohesive_energy(self, element):
        """
        Looks up the cohesive energy in the file specified by self.datafile_ce. File is a CSV. First column is the
        atomic symbol. Second column is the bulk cohesive energy, in eV. Sign convention: negative is favorable.
        Args:
            element: Atomic symbol of the element of interest

        Returns:
            Bulk cohesive energy of <element>
        """
        ce_df = pd.read_csv(self.datafile_ce, sep=",", index_col=False)
        ce = ce_df[ce_df.Atomic_Symbol == element].CE_eV.iat[0]
        return ce

    def lookup_bulk_coordination(self, element):
        """
        Looks up the bulk ccoordination number in the file specified by self.datafile_ce. File is a CSV. First column
        is the atomic symbol. Second column is the bulk coordination number.
        Args:
            element:

        Returns:

        """
        bulk_cn_csv = pd.read_csv(self.datafile_cn, sep=",", index_col=False)
        cn = bulk_cn_csv[bulk_cn_csv.Atomic_Symbol == element].CN.iat[0]
        return cn

    def calculate_gamma(self, element1, element2):
        """
        Calculates the gamma coefficient, between element1 and element2.
        Comes from solving the following system of equations:
            gamma_a * bde_aa + gamma_b * bde_bb = bde_ab
            gamma_a + gamma_b = 2

        Returns:
            Gamma coefficient set. First index is the half-bond coming from element1.
            Second index is the half-bond from element2.
        """
        # If A and B are the same, gamma's are always 1
        if element1 == element2:
            gamma_a = 1
            gamma_b = 1
        else:
            # gamma_1 * BDE(A-A) + gamma_2 * BDE(B-B) = 2 * BDE(A-B)
            # gamma1 + gamma2 = 2
            gammas = np.linalg.solve([[self.bde_aa, self.bde_bb], [1, 1]],
                                               [[2 * self.bde_ab], [2]])
            gamma_a = float(gammas[0])
            gamma_b = float(gammas[1])
        return gamma_a, gamma_b

    def calculate_total_gamma(self, cn, element1, element2):
        """
        Precalculates the total gamma value at the given CN for element_1 in a bond between element1 and element2
        Args:
            cn: Coordination number

        Returns:
            The total gamma value at the given coordination number
        """
        # Next, map over the CN range, building up the total gamma dictionary
        gamma = self.gamma[element1][element2]
        ce = None
        cnbulk = None
        if element1 == self.element_a:
            ce = self.ce_a
            cnbulk = self.cnbulk_a
        elif element1 == self.element_b:
            ce = self.ce_b
            cnbulk = self.cnbulk_b

        if cn == 0:
            total_gamma_a = None
        else:
            total_gamma_a = (gamma * ce / np.sqrt(cnbulk)) / np.sqrt(cn)
        return total_gamma_a

    def calc_coeffs_dict(self, max_cn=None):
        """
        Gives a total gamma coefficients dict in the format expected by the AtomsGraph class, precalculated for a
            range of CN values
        Args:
            max_cn: Maximum CN. Defaults to the GammaValues cn_max value if none is provided.
        Returns:
            Coefficients dict intended to be accessed by [element1][element2][cn]
        """
        if max_cn is None:
            max_cn = self.cn_max
        aa = [self.calculate_total_gamma(cn, self.element_a, self.element_a) for cn in range(0, max_cn)]
        bb = [self.calculate_total_gamma(cn, self.element_b, self.element_b) for cn in range(0, max_cn)]
        ab = [self.calculate_total_gamma(cn, self.element_a, self.element_b) for cn in range(0, max_cn)]
        ba = [self.calculate_total_gamma(cn, self.element_b, self.element_a) for cn in range(0, max_cn)]
        coefs = {self.element_a: {self.element_a: aa,
                                  self.element_b: ab},
                 self.element_b: {self.element_a: ba,
                                  self.element_b: bb}}
        return coefs


if __name__ == "__main__":
    gammas = GammaValues("Ag", "Au")
    # for a, b in itertools.combinations_with_replacement(["Cu", "Ag", "Au", "Pd", "Pt"], 2):
    # gammas = GammaValues(a, b).calc_coeffs_dict()
