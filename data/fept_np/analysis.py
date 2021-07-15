import os

import numpy as np
import matplotlib.pyplot as plt
import ase.io
from ase.data import jmol_colors

import ce_expansion.atomgraph as atomgraph


def build_central_rdf(atoms, metals, nbins=5, pcty=False, ax=None):
    # center atoms at origin (COP)
    atoms.positions -= atoms.positions.mean(0)

    metal1, metal2 = metals

    # calculate distances from origin
    dists = np.linalg.norm(atoms.positions, axis=1)

    # calculate distances from COP for each metal type
    dist_m1 = dists[atoms.symbols == metal1]
    dist_m2 = dists[atoms.symbols == metal2]

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    m1_color = jmol_colors[atoms[atoms.symbols == metal1][0].number]
    m2_color = jmol_colors[atoms[atoms.symbols == metal2][0].number]

    binrange = (0, dists.max())

    counts_m1, bin_edges = np.histogram(dist_m1, bins=nbins, range=binrange)
    counts_m2, bin_edges = np.histogram(dist_m2, bins=nbins, range=binrange)

    # get bins for ax.hist
    bins = np.linspace(0, dists.max(), nbins + 1)

    if pcty:
        counts_tot, bin_edges = np.histogram(dists, bins=nbins, range=binrange)
        counts_m1 = counts_m1 / counts_tot
        counts_m2 = counts_m2 / counts_tot

    x = (bin_edges[1:] + bin_edges[:-1]) / 2
    width = x[1] - x[0]
    if not pcty:
        ax.bar(x, counts_m1, edgecolor='k', color=m1_color,
               width=width, label=metal1)
        ax.bar(x, counts_m2, edgecolor='k', color=m2_color,
               width=width, label=metal2, bottom=counts_m1)
    else:
        ax.bar(x, counts_m1, color=m1_color, width=width,
               label=metal1, zorder=2)
        ax.bar(x, counts_m2, bottom=counts_m1, color=m2_color,
               width=width, label=metal2, zorder=0)

    ax.set_xlabel('Distance from Core ($\\rm \\AA$)')
    return fig, ax


def vis_FePt_results(pcty=False):
    """
    Load in exp. FePt and GA opt FePt NPs
    - same as cn_dist_plot, but customized for FePt comparison

    KArgs:
    - pcty (bool): if True, y-axis is normalized to show percentage
                   of CN filled by each metal type
                   (Default: False)

    Returns:
    - (plt.Figure), (plt.gca()): figure and axis object of plot
    """
    # get path to FePt_np folder in data
    fept_path = os.path.dirname(os.path.realpath(__file__))

    # read in atoms object
    origpath = os.path.join(fept_path, 'FePt_cns.xyz')
    orig = ase.io.read(origpath)

    # get GA-optimized structure
    ga = ase.io.read(os.path.join(fept_path, 'ga.xyz'))

    # get random structure from benchmark plot in data\\fept_np
    rand = ase.io.read(os.path.join(fept_path, 'random.xyz'))

    metals = ('Fe', 'Pt')
    fig, axes = plt.subplots(1, 3, sharey=True)
    ga_ax, orig_ax, rand_ax = axes.flatten()
    # ga_ax.set_title('GA-Optimized FePt NP')
    # orig_ax.set_title('Experimental FePt NP')
    # rand_ax.set_title('Random FePt NP')
    nbins = 20

    build_central_rdf(ga, metals, nbins=nbins, pcty=pcty, ax=ga_ax)
    build_central_rdf(orig, metals, nbins=nbins, pcty=pcty, ax=orig_ax)
    build_central_rdf(rand, metals, nbins=nbins, pcty=pcty, ax=rand_ax)

    if pcty:
        ga_ax.set_ylim(0, 1.1)
        ga_ax.set_yticklabels(['{:,.0%}'.format(x) for x in ga_ax.get_yticks()])
        ga_ax.set_ylabel('Atom Type Composition')
    else:
        ga_ax.set_ylabel('Number of Atoms')
    ga_ax.legend(ncol=2, frameon=False)
    fig.tight_layout()
    plt.show()
    return

    # get bonds from .npy file (if it exists)
    bondpath = os.path.join(fept_path, 'fept_bonslist.npy')
    if os.path.isfile(bondpath):
        bonds = np.load(bondpath)
    # else get bonds list from xyz file
    else:
        bonds = []
        with open(origpath, 'r') as fid:
            for i, line in enumerate(fid):
                if i > 1:
                    for b in map(int, line.split('[')[-1]
                            .strip(']\n').split(', ')):
                        newbond = [i - 2, b]
                        bonds.append(newbond)
        bonds = np.array(bonds)

    # define GA properties
    # shape is required to interface with database
    metals = ('Fe', 'Pt')
    shape = 'fept'

    # initialize atom graph
    ag = bcmraph.bcmraph(bonds.copy(), 'Fe', 'Pt')

    # get ordering
    orig_order = (orig.symbols == 'Pt').astype(int)
    ga_order = (ga.symbols == 'Pt').astype(int)

    # get bond types
    orig_mixing = ag.countMixing(orig_order)
    ga_mixing = ag.countMixing(ga_order)

    orig_cn_dist = ag.calc_cn_dist(orig_order)
    ga_cn_dist = ag.calc_cn_dist(ga_order)

    # get metal colors
    m1_color = jmol_colors[26]
    m2_color = jmol_colors[78]

    # get x value for both plots
    x = range(1, len(ga_cn_dist['cn_options']) + 1)

    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(14, 7))
    ga_ax, orig_ax = axes

    # orig plot params
    orig_ax.set_title('Experimental FePt NP')
    orig_ax.set_xlabel('CN')

    # ga plot params
    ga_ax.set_title('GA-Optimized FePt NP')
    ga_ax.set_xlabel('CN')
    ga_ax.set_ylabel('Number of Atoms')

    # plot bar plots
    for ax, dist in zip([ga_ax, orig_ax], [ga_cn_dist, orig_cn_dist]):
        if pcty:
            # normalize counts
            dist['m1_counts'] = dist['m1_counts'] / dist['tot_counts']
            dist['m2_counts'] = dist['m2_counts'] / dist['tot_counts']
            ax.set_ylim(0, 1.2)
            ax.set_yticklabels(['{:,.0%}'.format(x) for x in ax.get_yticks()[:-1]] + [''])
        else:
            ax.set_ylim(0, max(dist['tot_counts']) * 1.1)

        # plot each metal type
        ax.bar(x, dist['m1_counts'], color=m1_color, edgecolor='k',
               label=metals[0])
        ax.bar(x, dist['m2_counts'], bottom=dist['m1_counts'], color=m2_color,
               edgecolor='k', label=metals[1])

        # format each axis limits and tick values
        ax.set_xticks(x)
        ax.set_xticklabels(dist['cn_options'])

    ga_ax.legend(ncol=2, loc='upper left')
    fig.tight_layout()
    plt.show()


def test_FePt_nanop():
    """
    Optimizes FePt structure from Nano Lett. paper
    - NOTE: not complemented
    """
    # get path to FePt_np folder in data
    fept_path = os.path.dirname(os.path.realpath(__file__))

    # read in atoms object
    atompath = os.path.join(fept_path, 'FePt_cns.xyz')
    atom = ase.io.read(atompath)

    # get bonds from .npy file (if it exists)
    bondpath = os.path.join(fept_path, 'fept_bonslist.npy')
    if os.path.isfile(bondpath):
        bonds = np.load(bondpath)
    # else get bonds list from xyz file
    else:
        bonds = []
        with open(atompath, 'r') as fid:
            for i, line in enumerate(fid):
                if i > 1:
                    for b in map(int, line.split('[')[-1]
                            .strip(']\n').split(', ')):
                        newbond = [i - 2, b]
                        bonds.append(newbond)
        bonds = np.array(bonds)

    # define GA properties
    # shape is required to interface with database
    metals = ('Fe', 'Pt')
    shape = 'fept'

    # initialize atom graph
    ag = atomgraph.atomgraph(bonds.copy(), 'Fe', 'Pt')

    # number of Platinum (metal 2) atoms
    n_metal2 = sum(atom.symbols == 'Pt')

    # cannot run if GA results already exist (ensures no overwriting)
    for n in ['ga.xyz', 'random.xyz', 'gainfo.txt',
              'results.png', 'results.svg']:
        if os.path.isfile(os.path.join(fept_path, n)):
            print('GA has already run (found ga.xyz).')
            print('To rerun, please move previous results to new folder.')
            return

    # initialize and run GA
    pop = ga.Pop(atom, bonds.copy(), metals, shape,
              n_metal2=n_metal2, bcm=ag)

    # GA simulation
    max_nochange = 500
    pop.run(max_nochange=max_nochange)
    # pop.save(os.path.join(fept_path, 'fept_gapop.pickle'))

    # save best structure from GA and random search
    make_file(pop.atom, pop[0], os.path.join(fept_path, 'ga.xyz'))

    with open(os.path.join(fept_path, 'gainfo.txt'), 'w') as fid:
        fid.write('MaxNoChange: %i\n' % max_nochange)
        fid.write('    Runtime: %.3f\n' % pop.runtime)
        fid.write('     N Gens: %i\n' % pop.max_gens)
        fid.write('         CE: %.5f\n' % pop[0].ce)
        fid.write('         EE: %.5f' % pop.bcm.getEE(pop[0].ordering))

    # simulate random pop for same gens as GA
    max_gens = pop.max_gens

    randpop = Pop(atom, bonds.copy(), metals, shape,
                  n_metal2=n_metal2, bcm=ag, random=True)
    randpop.run(max_gens=max_gens, max_nochange=-1)

    make_file(randpop.atom, randpop[0], os.path.join(fept_path, 'random.xyz'))

    # plot results
    fig, ax = pop.plot_results()
    randpop.plot_results(ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(fept_path, 'results.png'))
    fig.savefig(os.path.join(fept_path, 'results.svg'))
    return fig, ax
