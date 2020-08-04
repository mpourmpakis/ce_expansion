#!/usr/bin/env python
"""
Code dealing with the gamma coefficients
"""


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
    if bde_aa == bde_bb == bde_ab:
        gamma_a = 1
        gamma_b = 1
    else:
        # Precalculated solution for the two gamma values after some linear algebra
        gamma_a = (bde_ab - 2 * bde_bb) / (bde_aa - bde_bb)
        gamma_b = (2 * bde_aa - bde_ab) / (bde_aa - bde_bb)
    return gamma_a, gamma_b



