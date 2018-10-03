#!/usr/bin/env python3

import numpy as np

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
bulk_constants = {"Cu":0,
                  "Ag":0,
                  "Au":0}

# Precompute Bulk CE / sqrt(bulk_CN)
for key, value in bulk_cohesive_energies.iteritems():
    bulk_constants[key] = value/np.sqrt(12)

#print bulk_constants

# ===============================================
# Calculate gamma_coeff * bulk_CE / sqrt(bulk_CN)
# ===============================================
# Initialization
precomputed_coeffs = {"Cu": {"Cu": 0,
                             "Ag": 0,
                             "Au": 0},
                      "Ag": {"Cu": 0,
                             "Ag": 0 ,
                             "Au": 0},
                      "Au": {"Cu": 0,
                             "Ag": 0,
                             "Au": 0}
                      }

# Precompute gamma coeff * bulk CE / sqrt(bulk_CN)
for element_1, inner_dict in gamma_coeffs.iteritems():
    for element_2 in gamma_coeffs:
        precomputed_coeffs[element_1][element_2]= bulk_constants[element_1]*inner_dict[element_2]

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
CN_precomp = [None]*13
for count,CN in enumerate(range(0,13)):
    if CN == 0:
        continue
    else:
        CN_precomp[count]=1/np.sqrt(CN)

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
#print(precomputed_coeffs)

