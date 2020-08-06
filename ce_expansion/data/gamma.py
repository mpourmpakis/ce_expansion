#!/usr/bin/env python
"""
Code dealing with the gamma coefficients
"""
import numpy as np


class GammaValues(object):
    def __init__(self, element_a, element_b, bde_aa=None, bde_ab=None, bde_bb=None, ce_a=None, ce_b=None,
                 cnbulk_a=None, cnbulk_b=None, cn_max=None):
        """
        An object providing a convenient model for calculating and interacting with gamma values defined by the BCM.
        :param element_a: Atomic symbol representing atom A
        :param element_b: Atomic symbol representing atom B
        :param bde_aa: Gas-phase homolytic bond dissociation energy of an A-A bond
        :param bde_ab: Gas-phase homolytic bond dissociation energy of an A-B bond
        :param bde_bb: Gas-phase homolytic bond dissociation energy of a B-B bond
        :param ce_a: Bulk cohesive energy of atom A
        :param ce_b: Bulk cohesive energy of atom B
        :param cnbulk_a: Bulk coordination number of atom A
        :param cnbulk_b: Bulk coordination number of atom B
        """
        self.element_a = element_a
        self.element_b = element_b
        # Look up BDE's if we have to
        bde_dict = {element_a: {element_a: bde_aa,
                                element_b: bde_ab},
                    element_b: {element_a: bde_ab,
                                element_b: bde_bb}
                    }
        for a, b in ((element_a, element_a), (element_a, element_b), (element_b, element_a), (element_b, element_b)):
            if bde_dict[a][b] is None:
                bde_dict[a][b] = self.lookup_bde(a, b)
        self.bde_aa = bde_dict[element_a][element_a]
        self.bde_ab = bde_dict[element_a][element_b]
        self.bde_bb = bde_dict[element_b][element_b]
        # Look up CE if we have to
        self.ce_a = ce_a
        self.ce_b = ce_b

        self.cnbulk_a = cnbulk_a
        self.cnbulk_b = cnbulk_b
        if cn_max is None:
            self.cn_max = max(self.cnbulk_a, self.cnbulk_b) + 1
        else:
            self.cn_max = cn_max

        # Calculate gammas
        ab_gammas = self.calculate_gamma(element_a, element_b)[0]
        self.gamma = {element_a: {element_a: 1,
                                  element_b: ab_gammas[0]},
                      element_b: {element_a: ab_gammas[1],
                                  element_b: 1}
                      }

    @staticmethod
    def lookup_bde(a, b):
        # Todo: implement
        return None

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
            # Precalculated solution for the two gamma values after some linear algebra
            gamma_a = (self.bde_ab - 2 * self.bde_bb) / (self.bde_aa - self.bde_bb)
            gamma_b = (2 * self.bde_aa - self.bde_ab) / (self.bde_aa - self.bde_bb)
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
        aa = [self.calculate_total_gamma(cn, self.element_a, self.element_a) for cn in range(0, max_cn + 1)]
        bb = [self.calculate_total_gamma(cn, self.element_b, self.element_b) for cn in range(0, max_cn + 1)]
        ab = [self.calculate_total_gamma(cn, self.element_a, self.element_b) for cn in range(0, max_cn + 1)]
        ba = [self.calculate_total_gamma(cn, self.element_b, self.element_a) for cn in range(0, max_cn + 1)]
        coefs = {self.element_a: {self.element_a: aa,
                                  self.element_b: ab},
                 self.element_b: {self.element_a: ba,
                                  self.element_b: bb}}
        return coefs
