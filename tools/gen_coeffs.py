#!/usr/bin/env python3

import numpy as np
import tools.gen_gamma
import re


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
                          element2: "str") -> "tuple":
    """
    Calculates the total gamma coefficient. The total gamma coefficient is defined as the product of the gamma
    coefficient, the bulk cohesive energy, and the inverse of the square root of the coordination number of the bulk.

    :param element1: The first element, A, in the bimetallic pair AB
    :type element1: str
    :param element2: The second element, B, in the bimetallic pair AB
    :type element2: str

    :return: A tuple containing the total gamma coefficients of A and B (in that order) in the bimetallic pair AB
    """
    # Look up CE_bulk
    CE_bulk1 = 0.0
    CE_bulk2 = 0.0

    # Look up bulk CN
    CN_bulk1 = 12
    CN_bulk2 = 12

    # Calculate gamma
    gamma1, gamma2 = tools.gen_gamma.calculate_gamma(element1, element2)

    # Calculate the total gamma coefficients
    total_gamma = (gamma1 * CE_bulk1 / np.sqrt(CN_bulk1),
                   gamma2 * CE_bulk2 / np.sqrt(CN_bulk2))

    return total_gamma


def generate_coefficient_dictionary(element1: "str",
                                    element2: "str",
                                    cn_bulk: "int",
                                    min_coord: "int" = 0,
                                    max_coord: "int" = -1) -> "dict":
    """
    Generates the total Gamma coefficients for a bimetallic pair AB from min_coord to max_coord. Coordination
    number 0 is given the value None. The total Gamma coefficient is defined as the product of the gamma coefficient,
    the bulk cohesive energy, and the inverse of the square root of the coordination number of the bulk.

    :param element1: The first element, A, in the bimetallic pair AB
    :type element1: str
    :param element2: The second element, B, in the bimetallic pair AB
    :type element2: str
    :param cn_bulk: The coordination number of the bulk.
    :type cn_bulk: int
    :param min_coord: The minimum coordination number to investigate. Defaults to 0.
    :type min_coord: int
    :param max_coord: The maximum coordination number to investigate. The default value of -1 indicates the maximum
                      coordination number to investigate is that of the bulk.

    :return: A dictionary of form dict[element1][element2][CN] = float
    """

    return {}
