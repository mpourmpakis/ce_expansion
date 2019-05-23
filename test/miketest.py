import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import atomgraph
import structure_gen
import numpy as np
import random
import sys

src = os.path.join(os.path.realpath(__file__), '..', 'src')
sys.path.append(src)

for p in plt.rcParams:
    if plt.rcParams[p] == 'bold':
        plt.rcParams[p] = 'normal'

# gamma values for Ag-Cu bonds
gamma = dict(Ag=50/36., Cu=2 - (50 / 36.))

# bulk CE
ce_bulk = dict(Ag=-2.95, Cu=-3.49)

# build CN (same for Ag and Cu)
cn_bulk = 12.


def calc_ce(atom, bonds):
    """
    Simple implementation of bond-centric CE method
    Yan et al. Nano Letters 2018, 18, 2696-2704
    """
    CE = 0

    # first atom (i)
    for i in range(len(atom)):
        sym1 = atom[i].symbol
        # iterate over bonds
        coords = np.where(bonds[:, 0] == i)[0]
        cni = len(coords)
        for bondindex in coords:
            # second atom (j)
            j = bonds[bondindex, 1]
            sym2 = atom[j].symbol

            # gamma is dependent on homo/heteroatomic bond
            gam = 1.0 if sym1 == sym2 else gamma[sym1]

            # calculate half bond energy
            CE += gam * (ce_bulk[sym1] / cni) * np.sqrt(cni / cn_bulk)

    CE /= len(atom)
    return CE


def compare(shape, nshell, ax_dict, color, plot=True):
    # build atoms object and its bonds list
    nanop = structure_gen.build_structure_sql(shape, nshell)
    atom = nanop.get_atoms_obj_skel()
    bonds = nanop.bonds_list

    # initialize AtomGraph object
    ag = atomgraph.AtomGraph(bonds, 'Ag', 'Cu')

    # TEST a number of random systems
    options = ['Ag', 'Cu']

    runs = 100

    # code: ce_expansion
    CE_code = np.zeros(runs)

    # mike: calc_ce - "actual"
    CE_mike = np.zeros(runs)

    for k in range(runs):
        # randomize atom
        for a in atom:
            a.symbol = random.choice(options)

        # AgCu atoms: Ag=0, Cu=1
        ordering = (atom.get_atomic_numbers() == 29).astype(int)

        # calc ce_expansion CE
        CE_code[k] = ag.getTotalCE(ordering)
        CE_mike[k] = calc_ce(atom, bonds)

    diffs = CE_mike - CE_code

    print('RESULTS'.center(40, '-'))
    print('%i-atom %s'.center(40) % (len(atom), shape))
    print('Mean Diff: %f' % diffs.mean())
    print(' STD Diff: %f' % diffs.std())
    print(' MAX Diff: %f' % diffs.max())
    print(' MIN Diff: %f' % diffs.min())

    if plot:
        ax_dict[shape].scatter(CE_mike, CE_code, s=75, edgecolor='k',
                               label='%i' % (len(atom)), color=color)

    return diffs, len(atom)

# ITERATE OVER ALL SHAPES
plt.close('all')

DPI = 150

basepath = os.path.join(
    os.path.expanduser('~'),
    'Desktop')

# shapes to test
shapes = ['icosahedron', 'cuboctahedron', 'fcc-cube']

max_shells = 13
totnshells = {'icosahedron': range(2, max_shells),
              'cuboctahedron': range(1, max_shells),
              'fcc-cube': range(1, max_shells)}

cmaps = [cm.Reds, cm.Blues, cm.Greens, cm.Purples]
color_dict = {}
fig_dict = {}
ax_dict = {}
for i, shape in enumerate(shapes):
    # parity plot
    fig, ax = plt.subplots()
    ax.set_xlabel('"actual" CE')
    ax.set_ylabel("ce_expansion CE")
    ax.set_title('$\\rm Ag_nCu_m$ %s' % shape)
    fig_dict[shape] = fig
    ax_dict[shape] = ax
    color_dict[shape] = cmaps[i]

# difference (b/n methods) plot
efig, eax = plt.subplots()
eax.set_xlabel('Number of atoms')
eax.set_ylabel('Average $\\rm CE_{diff}$')
eax.set_title('ce_expansion vs. paper model\n$\\rm Ag_nCu_m$')

# run the tests (iterate 100 times and compare methods)
for shape in shapes:
    nshells = totnshells[shape]
    totdiffs = np.empty((len(nshells), 2))
    numatoms = np.empty(len(nshells))
    for i, shell in enumerate(nshells):
        diffs, num = compare(shape, shell, ax_dict,
                             color=color_dict[shape](shell / max_shells))
        totdiffs[i, 0] = diffs.mean()
        totdiffs[i, 1] = diffs.std()
        numatoms[i] = num

    eax.errorbar(numatoms, totdiffs[:, 0], yerr=totdiffs[:, 1], fmt='o-',
                 markeredgecolor='k', label=shape, ms=10, capsize=5,
                 color=color_dict[shape](0.5))

eax.set_ylim(0, 0.002)
eax.legend()
efig.tight_layout()
efig.savefig(basepath + '\\natoms-vs-error-AgCu.png', dpi=DPI)

for k in ax_dict:
    fig = fig_dict[k]
    ax = ax_dict[k]
    ax.legend(loc='upper left', fontsize=13)
    parity = ax.dataLim.ymax, ax.dataLim.ymin
    ax.plot(parity, parity, color='k', zorder=-5)
    ax.set_xlim(-3.2, -2)
    ax.set_ylim(-3.2, -2)
    ax.set_aspect(1)

    fig.tight_layout()
    fig.savefig(basepath + '\\AgCu-parity-%s.png' % (k[:3]), dpi=DPI)
# plt.show()
