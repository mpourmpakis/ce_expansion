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


a = read_data_table("../data/bulkdata.csv")
print(a)


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


# Gamma Coefficients (eV)
# Calling gamma_coeffs["Cu"]["Au"] returns the Cu-Au bond coefficient for Cu
# BDE data comes from:
#   Miedema, A. R.,
#   Model predictions of the dissociation energies of homonuclear and heteronuclear diatomic molecules of two transition metals.
#   Faraday Symposia of the Chemical Society 1980, 14 (0), 136-148.
# See Yan et. al. for further information on how gamma coefficients are calculated.
gamma_coeffs = {"Cu": {"Cu": 1.00,
                       "Ag": 0.61,
                       "Au": -0.36},
                "Ag": {"Cu": 1.39,
                       "Ag": 1.00,
                       "Au": 0.72},
                "Au": {"Cu": 2.36,
                       "Ag": 1.28,
                       "Au": 1.00}
                }
# Bulk Cohesive energies (eV/atom) from:
#   Charles Kittel. Introduction to Solid State Physics, 8th edition. Hoboken, NJ: John Wiley & Sons, Inc, 2005.

bulk_cohesive_energies = {"Cu": -3.49,
                          "Ag": -2.95,
                          "Au": -3.81}

# Bulk CN in an FCC cell (these are all FCC cells) is 12
bulk_CN = 12.0

# ======================================
# Calculate bulk bulk CE / sqrt(bulk_CN)
# ======================================
# Initialization
bulk_constants = {"Cu": 0,
                  "Ag": 0,
                  "Au": 0}

# Precompute Bulk CE / sqrt(bulk_CN)
for key, value in bulk_cohesive_energies.iteritems():
    bulk_constants[key] = value / np.sqrt(12)

# print bulk_constants

# ===============================================
# Calculate gamma_coeff * bulk_CE / sqrt(bulk_CN)
# ===============================================
# Initialization
precomputed_coeffs = {"Cu": {"Cu": 0,
                             "Ag": 0,
                             "Au": 0},
                      "Ag": {"Cu": 0,
                             "Ag": 0,
                             "Au": 0},
                      "Au": {"Cu": 0,
                             "Ag": 0,
                             "Au": 0}
                      }

# Precompute gamma coeff * bulk CE / sqrt(bulk_CN)
for element_1, inner_dict in gamma_coeffs.iteritems():
    for element_2 in gamma_coeffs:
        precomputed_coeffs[element_1][element_2] = bulk_constants[element_1] * inner_dict[element_2]

# =====================================================
# Construct a lookup table for each CN for each element
# =====================================================

# Initialization
CN_coeffs = {"Cu": {"Cu": 0,
                    "Ag": 0,
                    "Au": 0},
             "Ag": {"Cu": 0,
                    "Ag": 0,
                    "Au": 0},
             "Au": {"Cu": 0,
                    "Ag": 0,
                    "Au": 0}
             }

# CN cannot be 0 (divide by 0) and cannot be more than 12 (because the fully-coordinated bulk is 12)
CN_precomp = [None] * 13
for count, CN in enumerate(range(0, 13)):
    if CN == 0:
        continue
    else:
        CN_precomp[count] = 1 / np.sqrt(CN)

# Build up the lookup table
# Syntax for the dictionary is:
#   precomputed_coeffs[element_1][element_2][CN]
# So for example, to get the value for a Cu-Ag bond, with Cu @ CN6 and Ag @ CN12:
#   precomputed_coeffs["Cu"]["Ag"][6] + precomputed_coeffs["Ag"]["Cu"][12]
for element_1, inner_dict in precomputed_coeffs.iteritems():
    for element_2 in precomputed_coeffs:
        # Compute the CN table
        CN_precomp = [None] * 13
        for count, CN in enumerate(range(0, 13)):
            if CN == 0:
                continue
            else:
                CN_precomp[count] = inner_dict[element_2] / np.sqrt(CN)
        precomputed_coeffs[element_1][element_2] = CN_precomp

# Finally, print to console to be pasted in other code
# print(precomputed_coeffs)
