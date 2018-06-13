# Gamma Coefficients Matrix
# Arranged as a dict. For example, calling gamma_coeffs["Cu"]["Au"] returns the Cu-Au bond coefficient
# BDE data comes from:
#   Miedema, A. R.,
#   Model predictions of the dissociation energies of homonuclear and heteronuclear diatomic molecules of two transition metals.
#   Faraday Symposia of the Chemical Society 1980, 14 (0), 136-148.
# Solved in mathematica to get the equations

gamma_coeffs = {"Cu": {"Cu":  1.00,
                       "Ag":  0.61,
                       "Au": -0.36},
                "Ag": {"Cu":  1.39,
                       "Ag":  1.00,
                       "Au":  0.72},
                "Au": {"Cu":  2.36,
                       "Ag":  1.28,
                       "Au":  1.00}
                }

# Bulk Cohesive energies from:
#   Charles Kittel. Introduction to Solid State Physics, 8th edition. Hoboken, NJ: John Wiley & Sons, Inc, 2005.
cohesive_energies = {"Cu": -3.49,
                     "Ag": -2.95,
                     "Au": -3.81}

import ase.cluster
import ase.neighborlist

x=ase.cluster.Icosahedron("Cu",2)
a=ase.neighborlist.neighbor_list("i")

ase.neighborlist

