import matplotlib.pyplot as plt
import numpy as np
from ase.data import chemical_symbols
from ase.data.colors import jmol_colors

import atomgraph
from ce_expansion.atomgraph import atomgraph
from ce_expansion.ga import structure_gen
from ce_expansion.npdb import datatables as dt
from ce_expansion.npdb import db_inter

for rc in plt.rcParams:
    if plt.rcParams[rc] == 'bold':
        plt.rcParams[rc] = 'normal'


def cn_dist_plot(bimet_res: "dt.BimetallicResults",
                 pcty=False, show=False):
    """
    Creates CN distribution plot of bimetallic NP

    Args:
    - bimet_res (npdb.datatables.BimetallicResults): bimetallic result from
                                                     sql database

    KArgs:
    - pcty (bool): if True, y-axis is normalized to show percentage
                   of CN filled by each metal type
                   (Default: False)
    - show (bool): if True, plt.show() is called to show plot
                   (Default: False)

    Returns:
    - (plt.Figure), (plt.gca()): figure and axis object of plot
    """
    # create params from BiMetResult object (done to shorten names)
    m1 = bimet_res.metal1
    m2 = bimet_res.metal2
    shape = bimet_res.shape

    # create bonds list
    # try to load it from nanoparticle object
    if isinstance(bimet_res.nanoparticle.load_bonds_list(), np.ndarray):
        bonds = bimet_res.nanoparticle.bonds_list
    else:
        raise ValueError("Unable to load bonds list")

    # ordering array
    ordering = np.array([int(i) for i in bimet_res.ordering])

    # initialize atom graph
    ag = atomgraph.AtomGraph(bonds.copy(), m1, m2)

    cn_dist = ag.calc_cn_dist(ordering)

    # get metal colors
    m1_color = jmol_colors[chemical_symbols.index(m1)]
    m2_color = jmol_colors[chemical_symbols.index(m2)]

    # get x value for both plots
    x = range(1, len(cn_dist['cn_options']) + 1)

    fig, ax = plt.subplots()

    # plot params
    formula = bimet_res.build_chem_formula(True)
    ax.set_title(f'{formula} ({bimet_res.num_atoms}-atom {shape.title()})')
    ax.set_xlabel('CN')
    ax.set_xticks(x)
    ax.set_xticklabels(cn_dist['cn_options'])

    # normalize counts
    if pcty:
        cn_dist['m1_counts'] = cn_dist['m1_counts'] / cn_dist['tot_counts']
        cn_dist['m2_counts'] = cn_dist['m2_counts'] / cn_dist['tot_counts']
        ax.set_ylabel('Percentage of Atoms')
        ax.set_ylim(0, 1.2)
        ax.set_yticklabels(['{:,.0%}'.format(x)
                            for x in ax.get_yticks()[:-1]] + [''])

    else:
        ax.set_ylabel('Number of Atoms')
        ax.set_ylim(0, max(cn_dist['tot_counts']) * 1.1)

    ax.bar(x, cn_dist['m1_counts'], color=m1_color, edgecolor='k',
           label=m1)
    ax.bar(x, cn_dist['m2_counts'], bottom=cn_dist['m1_counts'],
           color=m2_color, edgecolor='k', label=m2)

    ax.legend()
    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax


if __name__ == '__main__':
    metals = 'agau'
    shape = 'icosahedron'
    num_shells = 9

    bonds = np.array(structure_gen.build_structure_sql(shape, num_shells)
                     .bonds_list)
    res = db_inter.get_bimet_result(metals, shape=shape, num_shells=num_shells,
                                    only_bimet=True)[4]
    fig, ax = cn_dist_plot(res, pcty=True)
    plt.show()
