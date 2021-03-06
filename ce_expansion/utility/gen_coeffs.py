#!/usr/bin/env python3

import re

import numpy as np

from ce_expansion.utility import gen_gamma


def read_data_table(filename: "str") -> "dict":
    """
    Reads in a CSV file containing columnwise data for various elements, and returns a dictionary containing the data.
    Lines beginning with "#" are ignored
    :param filename: A valid filename for a csv file
    :type filename: str

    :return: A dictionary containing the data of interest
    """
    values = {}
    with open(filename, "r") as data:
        for line in data:
            if re.match("^(\s+#|#|\s+$)", line):
                continue
            elif re.match("^(\s|\s+)$", line):
                continue
            key, value = line.strip().split(",")
            values[key] = float(value)

    return values


def calculate_total_gamma(element1: "str",
                          element2: "str",
                          ce_data: "str" = "../data/bulkcedata.csv",
                          cn_data: "str" = "../data/cndata.csv") -> "tuple":
    """
    Calculates the total gamma coefficient. The total gamma coefficient is defined as the product of the gamma
    coefficient, the bulk cohesive energy, and the inverse of the square root of the coordination number of the bulk.

    :param element1: The first element, A, in the bimetallic pair AB
    :type element1: str
    :param element2: The second element, B, in the bimetallic pair AB
    :type element2: str
    :param ce_data: Data source for bulk cohesive energies.
    :type ce_data: str
    :param  cn_data: Data source for the bulk coordination number.
    :type cn_data: str

    :return: A tuple containing the total gamma coefficients of A and B (in that order) in the bimetallic pair AB
    """
    # Look up CE_bulk
    ce_data = read_data_table(ce_data)
    ce_bulk1 = ce_data[element1]
    ce_bulk2 = ce_data[element2]

    # Look up bulk CN
    cn_data = read_data_table(cn_data)
    cn_bulk1 = cn_data[element1]
    cn_bulk2 = cn_data[element2]

    # Calculate gamma
    gamma1, gamma2 = gen_gamma.calculate_gamma(element1, element2)

    # Calculate the total gamma coefficients
    total_gamma = (gamma1 * ce_bulk1 / np.sqrt(cn_bulk1),
                   gamma2 * ce_bulk2 / np.sqrt(cn_bulk2))

    return total_gamma


def calculate_gamma_products(coordination: "iterable",
                             total_gamma: "float") -> "float":
    """
    Scales the total gamma by coordination number. This will break if you fail to give it a sorted list whose values
    increase by 1 and do not skip numbers.

    :param coordination: List of coordination numbers to calculate
    :type coordination: list
    :param total_gamma:  Total gamma coefficient.
    :type total_gamma: float

    :return: The calculated gamma product.
    """
    gamma_map = []
    for cn in coordination:
        if cn == 0:
            gamma_map.append(None)
        else:
            gamma_map.append(total_gamma / np.sqrt(cn))
    return gamma_map


def generate_coefficient_dictionary(element1: "str",
                                    element2: "str",
                                    min_coord: "int" = 0,
                                    max_coord: "int" = -1,
                                    cn_data: "str" = "../data/cndata.csv") -> "dict":
    """
    Generates the total Gamma coefficients for a bimetallic pair AB from min_coord to max_coord. Coordination
    number 0 is given the value None. The total Gamma coefficient is defined as the product of the gamma coefficient,
    the bulk cohesive energy, and the inverse of the square root of the coordination number of the bulk.

    :param element1: The first element, A, in the bimetallic pair AB
    :type element1: str
    :param element2: The second element, B, in the bimetallic pair AB
    :type element2: str
    :param min_coord: The minimum coordination number to investigate. Defaults to 0.
    :type min_coord: int
    :param max_coord: The maximum coordination number to investigate. The default value of -1 indicates the maximum
                      coordination number to investigate is that of the bulk.
    :param cn_data: The CSV file containing bulk coordination number information
    :type cn_data: str

    :return: A dictionary of form dict[element1][element2][CN] = float
    """
    # Deal with default CN
    if max_coord - 1:
        coord_dict = read_data_table(cn_data)
        max_coord = int(max(coord_dict[element2], coord_dict[element1]))

    # Calculate the list of gamma products by mapping our gamma-generator onto a range
    total_gammas = calculate_total_gamma(element1, element2, cn_data=cn_data)
    element1_coeffs = calculate_gamma_products(range(min_coord, max_coord + 1), total_gammas[0])
    element2_coeffs = calculate_gamma_products(range(min_coord, max_coord + 1), total_gammas[1])

    # Put these into a list. Element1 bound to element2 has a certain set of coefficients.
    coeff_dict = {element1: {element2: element1_coeffs},
                  element2: {element1: element2_coeffs}}

    return coeff_dict


def gen_coeffs_dict_from_raw(metal1, metal2, bulkce_m1, bulkce_m2,
                             homo_bde_m1, homo_bde_m2, hetero_bde, cnmax=12):
    """
        Generates the Bond-Centric half bond energy terms for a bimetallic
        pair AB. Coordination number 0 is given the value None. Dictionary is
        used with AtomGraph to calculate total CE of bimetallic nanoparticles.
        Relies on raw data arguments to create dictionary.

        Args:
        - metal1 (str): atomic symbol of metal 1
        - metal2 (str): atomic symbol of metal 2
        - bulkce_m1 (float): bulk cohesive energy (in eV / atom) of metal1
        - bulkce_m2 (float): bulk cohesive energy (in eV / atom) of metal2
        - homo_bde_m1 (float): m1-m1 (homoatomic) bond dissociation energy
        - homo_bde_m2 (float): m2-m2 (homoatomic) bond dissociation energy
        - hetero_bde (float): m1-m2 (heteroatomic) bond dissociation energy

        KArgs:
        - cnmax: maximum bulk coordination number (CN) of metals
                 (Default: 12)

        Returns:
        - (dict): form dict[m1][m2][CN] = half bond energy term
    """
    metals = [metal1, metal2]

    # calculate gammas
    gamma_m1 = (2 * (hetero_bde - homo_bde_m2)) / (homo_bde_m1 - homo_bde_m2)
    gamma_m2 = 2 - gamma_m1

    # create bulkce and gamma dictionaries
    bulkce = {metal1: bulkce_m1,
              metal2: bulkce_m2}
    gammas = {metal1: {metal1: 1, metal2: gamma_m1},
              metal2: {metal2: 1, metal1: gamma_m2}}

    # calculate "total gamma" params (part of BC model that is independent of
    # current atomic CN)
    totgamma = {}
    for m in metals:
        totgamma[m] = {}
        for m2 in metals:
            totgamma[m][m2] = gammas[m][m2] * bulkce[m] / np.sqrt(cnmax)

    # create coefficient dictionary
    coeffs = {}
    for m in metals:
        coeffs[m] = {}
        for m2 in metals:
            coeffs[m][m2] = [None if cn == 0 else totgamma[m][m2] / np.sqrt(cn)
                             for cn in range(cnmax + 1)]
    return coeffs
