#!/usr/bin/env python
"""
Code dealing with the gamma coefficients
"""
import numpy as np


def calculate_gamma(bde_aa, bde_bb, bde_ab):
    """
    Calculates the gamma coefficient, if the values for the A-A, B-B, and A-B dimer are known.
    Comes from solving the following system of equations:
        gamma_a * bde_aa + gamma_b * bde_bb = bde_ab
        gamma_a + gamma_b = 2

    :param bde_aa: Energy of the A-A dimer.
    :param bde_bb: Energy of the B-B dimer.
    :param bde_ab: Energy of the A-B dimer.
    :return: Gamma coefficient for the given system
    """
    # If A and B are the same, gamma's are always 1
    if bde_aa == bde_bb == bde_ab:
        gamma_a = 1
        gamma_b = 1
    else:
        # Precalculated solution for the two gamma values after some linear algebra
        gamma_a = (bde_ab - 2 * bde_bb) / (bde_aa - bde_bb)
        gamma_b = (2 * bde_aa - bde_ab) / (bde_aa - bde_bb)
    return gamma_a, gamma_b


def calculate_total_gamma(ce_bulk_a, cn_bulk_a, gamma_a, cn_a_max=None):
    """
    Precalculates the total gamma for a half-bond coming from A in the A-B bond.
    :param ce_bulk_a: Bulk cohesive energy of atom A
    :param cn_bulk_a: Bulk coordination number of atom A
    :param gamma_a: Gamma coefficient of atom A for the A-B bond
    :param cn_a_max: Maximum CN for atom A. Defaults to cn_bulk_a if none provided.
    :return: The total gamma value for the given system
    """
    # Initialize
    if cn_a_max is None:
        cn_a_max = cn_bulk_a
    results_dict = {}
    # Next, map over the CN range, building up the total gamma dictionary
    for cn in range(0, cn_a_max + 1):
        total_gamma = (gamma_a * ce_bulk_a / np.sqrt(cn_bulk_a)) / np.sqrt(cn)
        results_dict[cn] = total_gamma
    return results_dict


