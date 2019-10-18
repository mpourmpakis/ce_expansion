import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib import cm


use_experimental = False
startfrom = 7

ce_bulks = dict(
            Sc=-3.90,
            Ti=-4.85,
            V=-5.31,
            Cr=-4.10,
            Mn=-2.92,
            Fe=-4.28,
            Co=-4.39,
            Ni=-4.44,
            Cu=-3.49,
            Y=-4.37,
            Zr=-7.1914,
            Nb=-7.57,
            Mo=-6.82,
            Tc=-6.85,
            Ru=-6.74,
            Rh=-5.75,
            Pd=-3.89,
            Ag=-2.95,
            La=-4.47,
            Hf=-6.44,
            Ta=-8.10,
            W=-8.90,
            Re=-8.03,
            Os=-8.17,
            Ir=-6.94,
            Pt=-5.84,
            Au=-3.81,
            Th=-6.20)

n = 'experimental' if use_experimental else 'estimated'

df = pd.read_csv('ce_expansion\\data\\%s_hbe.csv' % n)
elements = df.columns.values.copy()
matrix = df.values.copy()

# drop Th
elements = elements[:-1]
matrix = matrix[:-1, :-1]

elements = elements.reshape((3, 9))[:, startfrom-1:].ravel()
matrix = matrix[startfrom-1:, startfrom-1:]

# bond dissociation energies of elements
hbes = {}

matrix[np.where(matrix == 'None')] = None
matrix = matrix.astype(float)

names = []
diffce = []
gamma1 = []

# iterate over every element pair combination
n_els = len(elements)
for m1 in range(n_els - 1):
    for m2 in range(m1 + 1, n_els):
        if m1 % (10 - startfrom) != m2 % (10 - startfrom):
            continue
        hetero_bde = matrix[m1, m2]
        homo_bde_m1 = matrix[m1, m1]
        homo_bde_m2 = matrix[m2, m2]

        # calculate gammas for element pairs with known BDEs
        if hetero_bde * homo_bde_m2 * homo_bde_m2 > 0:
            # get element symbols
            el1 = elements[m1]
            el2 = elements[m2]

            # calculate gammas
            gamma_m1 = ((2 * (hetero_bde - homo_bde_m2)) /
                        (homo_bde_m1 - homo_bde_m2))
            gamma_m2 = 2 - gamma_m1

            if el1 not in hbes:
                hbes[el1] = {el1: 1}
            if el2 not in hbes:
                hbes[el2] = {el2: 1}

            hbes[el1][el2] = gamma_m1
            hbes[el2][el1] = gamma_m2

            names.append(el1 + el2)
            diffce.append(ce_bulks[el1] - ce_bulks[el2])
            # gamma1.append(hbes[el1][el2])
            gamma1.append(homo_bde_m1 - hetero_bde)

cmap = cm.get_cmap('hsv')
colors = ['red', 'blue', 'green', 'gold', 'violet', 'black', 'gray',
          'brown', 'pink']
# markers = ['s', 'o', '*', '^']
x = np.linspace(0, 1, len(names))
print(len(names))
fig, ax = plt.subplots()
for i in range(len(names)):
    ax.scatter(diffce[i], gamma1[i], label=names[i], s=100,
               color=colors[i % (11 - startfrom)])  # , marker=markers[i % 4])
ax.legend(ncol=8)
ax.set_xlabel('$\\rm \\Delta CE$')
ax.set_ylabel('$\\rm \\gamma_1$')
plt.show()
