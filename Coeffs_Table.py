#!/usr/bin/env python

import numpy as np

# Gamma Coefficients (eV)
# Calling gamma_coeffs["Cu"]["Au"] returns the Cu-Au bond coefficient
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

bulk_constants = {"Cu":0,
             "Ag":0,
             "Au":0}

for key, value in bulk_cohesive_energies.iteritems():
    bulk_constants[key] = value/np.sqrt(12)

print bulk_constants

precomputed_coeffs = {"Cu": {"Cu": 1.00*bulk_constants["Cu"],
                             "Ag": 0.61,
                             "Au": -0.36},
                      "Ag": {"Cu": 1.39,
                             "Ag": 1.00,
                             "Au": 0.72},
                      "Au": {"Cu": 2.36,
                             "Ag": 1.28,
                             "Au": 1.00}
                      }