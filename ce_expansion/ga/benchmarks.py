import os
import pathlib
from pprint import pprint

import numpy as np
import matplotlib.pyplot as plt

import ce_expansion.ga as ga
import ce_expansion.atomgraph as atomgraph
import ce_expansion.npdb.db_inter as db_inter


def check_db_values(update_db=False, metal_opts=None):
    """
    Checks CE values in database to ensure CE and EE match their ordering

    KArgs:
    - update_db (bool): if True and mismatch is found, the database will be
                        updated to the correct CE and EE
                        (Default: False)
    - metal_opts (list): can pass in list of metal combination options to check
                         (Default: all metal pairs from results in DB)

    Returns:
    - (np.ndarray): CE's and EE's of mismatches found
                    - each row contains: [CE-db, CE-actual, EE-db, EE-actual]

    """
    # if None, get all metal pairs found in BimetallicResults table in DB
    if metal_opts is None:
        metal_opts = db_inter.build_metal_pairs_list()

    # track systems that failed test
    fails = []

    # get all nanoparticle (shape, num_shell) pairs in database
    nanoparticles = set([(r.shape, r.num_shells)
                         for r in db_inter.get_nanoparticle()])

    # tracked number of results checked
    num_checked = 0
    print('Checking results...')
    for shape, num_shells in nanoparticles:
        nanop = ga.structure_gen.build_structure_sql(shape, num_shells)
        for metals in metal_opts:
            # ensure metal types are sorted
            metals = sorted(metals)

            # find all bimetallic results matching shape, size, and metals
            results = db_inter.get_bimet_result(metals, shape=shape,
                                                num_shells=num_shells)

            # if no results found, continue to next metal combination
            if not results:
                continue

            # else create bcmraph object
            bcm = atomgraph.AtomGraph(nanop.bonds_list,
                                        metals[0], metals[1])

            # iterate over results to compare CE in databse vs.
            # CE calculated with ordering
            for res in results:
                ordering = np.array(list(map(int, res.ordering)))
                n_metal2 = ordering.sum()
                actual_ce = bcm.getTotalCE(ordering)
                actual_ee = bcm.getEE(ordering)

                # increment number of results checkered
                num_checked += 1

                # create output string
                outp = (f'{res.shape[:3].upper():>4}',
                        f'{res.num_atoms:<7,}',
                        f'{res.build_chem_formula():<15}')

                # if deviation, add info to fails list
                if abs(actual_ce - res.CE) > 1E-10:
                    # print system with problem
                    print(*outp, f'WRONG VALUE! ({actual_ce - res.CE:.3e} eV)')

                    fails.append([metals, shape, num_shells, n_metal2, res.CE,
                                  actual_ce, res.EE, actual_ee])

                    # if update_db, correct the CE value
                    # NOTE: this most likely means that CE is not optimized
                    if update_db:
                        db_inter.update_bimet_result(
                            metals=metals,
                            shape=res.shape,
                            num_atoms=res.num_atoms,
                            diameter=res.diameter,
                            n_metal1=res.n_metal1,
                            CE=actual_ce,
                            ordering=res.ordering,
                            EE=actual_ee,
                            nanop=res.nanoparticle,
                            allow_insert=False,
                            ensure_ce_min=False)
                else:
                    print(*outp, end='\r')

    fails = np.array(fails)
    nfail = len(fails)
    issue_str = 'issue' if nfail == 1 else 'issues'
    print(' ' * 50, end='\r')
    print(f'{nfail:,} {issue_str} found.')
    print(f'{num_checked:,} results checked.')
    return fails


def benchmark_plot(max_nochange=500, metals=('Ag', 'Cu'), shape='icosahedron',
                   num_shells=9, x_metal2=0.57, spike=False, max_gens=-1,
                   min_gens=-1, path='', save=False, **kwargs):
    """
    Creates a plot comparing GA simulation vs. random search

    KArgs:
    - max_nochange (int): Convergence criteria for GA - GA stops when
                          <max_nochange> generations pass with no difference
                          in minimum CE
                          (Default: 50)
    - metals (tuple): tuple of metal types in nanoparticle
                      (default: {('Ag', 'Cu')})
    - shape (str): shape of nanoparticle
                   (Default: 'icosahedron')
    - num_shells (int): number of shells that make up nanoparticle
                        (Default: 10 (2869-atom nanoparticle))
    - x_metal2 (float): concentration of metal2 in nanoparticle
                        (Default: 0.57)
    - spike (bool): if True, following structures are added to generation 0
                    - if same structure and composition found in DB,
                      it is added to population
                    - minCN-filled or maxCN-filled nanoparticle
                      (depending on which is more fit) also added
                      to population - structure created using
                      fill_cn function
                    (Default: False)
    - max_gens (int): if not -1, use specified value as max
                      generations for each GA sim
                      (Default: -1)
    - min_gens (int): if not -1, use specified value as min
                      generations that run before max_nochange
                      criteria is applied
                      (Default: -1)
    - **kwargs (type): additional kwargs get passed in to Pop class
                       - examples: popsize, mute_pct, n_mute_atomswaps

    Returns:
    - plt.Figure, plt.axis: figure and axis object of plot
    """
    # create population to run GA and to run random search
    newp = ga.build_pop_obj(metals, shape, num_shells, x_metal2=x_metal2,
                         spike=spike, **kwargs)
    randp = ga.build_pop_obj(metals, shape, num_shells, x_metal2=x_metal2,
                          random=True)

    # run GA until it converges (no changes for <max_nochanges> generations)
    newp.run(max_gens=max_gens, min_gens=min_gens, max_nochange=max_nochange)
    print('RUNTIME: %.3f s' % newp.runtime)

    # random search for same number of generations
    randp.run(newp.max_gens, max_nochange=-1)

    # save best structure from GA and random search
    if save:
        ga.make_file(newp.atom, newp.pop[0], os.path.join(path, 'ga.xyz'))
        ga.make_file(randp.atom, randp.pop[0], os.path.join(path, 'random.xyz'))

    # plot results
    fig, ax = newp.plot_results()
    randp.plot_results(ax=ax)

    if save:
        if path and not os.path.isdir(path):
            pathlib.Path(path).mkdir(parents=True, exist_ok=True)

        # save figure as svg
        fig.savefig(os.path.join(path, 'results.svg'))
        fig.savefig(os.path.join(path, 'results.png'), dpi=66.6667)

        # save run info to txt file
        m1, m2 = metals
        n = newp.num_atoms
        n1 = newp.n_metal1
        n2 = newp.n_metal2
        x2 = int(round(x_metal2 * 100, 2))
        x1 = 100 - x2
        nscreened = newp.max_gens * newp.popsize
        screen_per_min = nscreened / (newp.runtime / 60)
        with open(os.path.join(path, 'benchmark_info.txt'), 'w') as fid:
            fid.write('%i-atom %s%s %s (%s shells)\n' % (newp.num_atoms,
                                                         m1, m2,
                                                         shape, num_shells))
            fid.write('%i%% %s (%i)\n' % (x1, m1, n1))
            fid.write('%i%% %s (%i)\n' % (x2, m2, n2))
            fid.write('\nGA PROPS\n')
            fid.write('popsize = %i\n' % newp.popsize)
            fid.write('max_nochange=%i\n' % max_nochange)
            fid.write('spike=%s\n' % str(spike))
            fid.write('runtime=%.2f seconds (%.2f minutes)\n'
                      % (newp.runtime, newp.runtime / 60))
            fid.write('total NPs screened = %i (%.2f NPs per min)'
                      % (nscreened, screen_per_min))
    return fig, ax


def scaling_plot(metals=('Ag', 'Cu'), shape='icosahedron',
                 num_shells_range=range(2, 11), x_metal2=0.5):
    """Creates a scaling plot to test GA
       - number of atoms vs. runtime for 500 generations (in seconds)
       Reveals that GA simulations scale by O(n) where n = num_atoms

    Keyword Arguments:
        metals (tuple): metal types in nanoparticle
                        (Default: ('Ag', 'Cu'))
        shape (str): shape of nanoparticle
                     (Default: 'icosahedron')
        num_shells_range (list): range of num_shells (nanoparticle sizes)
                                 to test
                                 (Default: (range(2, 11))
        x_metal2 (float): concentration of metal2 in nanoparticle
                          (Default: 0.57)

    Returns:
    - plt.Figure, plt.axis: figure and axis object of plot
    """
    natoms = []
    times = []

    for num_shells in num_shells_range:
        newp = ga.build_pop_obj(metals, shape, num_shells,
                                x_metal2=x_metal2, use_metropolis=False)
        newp.run(max_gens=500, max_nochange=-1)
        natoms.append(len(newp.atom))
        times.append(newp.runtime)

    return natoms, times

    fig, ax = plt.subplots()
    ax.plot(natoms, times, 'o-', color='green',
            ms=10, markeredgecolor='k')

    ax.set_xlabel('Number of Atoms')
    ax.set_ylabel('Runtime (seconds)\nfor 500 Generations')
    fig.tight_layout()
    return fig, ax


def compare_AtomGraph_and_BCModel():
    # number of runs per size
    n_runs = 2
    
    # max shell size
    max_n_shells = 3
    
    fig, ax = plt.subplots()
    
    all_runs = {}
    for bcm_object, color in zip(['BCModel', 'AtomGraph'], ['dodgerblue', 'orange']):
        print(f'Runnning {bcm_object}.')
        runs = []
        np.random.seed(15213)
        for i in range(n_runs):
            natoms, times = scaling_plot(['Ag', 'Cu'], num_shells_range=range(1, max_n_shells), x_metal2=0.5)
            runs.append(times)
        # average the runs
        runs = np.array(runs)
        runs_avg = runs.mean(0)
        runs_std = runs.std(0)
        all_runs[bcm_object] = (runs_avg, runs_std)
        ax.plot(natoms, runs_avg, '--', color=color, zorder=-300, alpha=0.5)
        ax.scatter(natoms, runs_avg, s=50, edgecolor='k', label=bcm_object, color=color)
        ax.errorbar(natoms, runs_avg, linestyle='', yerr=runs_std, zorder=-100, capsize=5, color=color)
    pprint(all_runs)
    ax.legend()
    ax.set_ylabel('Time (s / 500 Generations)')
    ax.set_xlabel('$\\rm N_{atoms}$')
    plt.show()

compare_AtomGraph_and_BCModel()
