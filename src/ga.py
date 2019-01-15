import operator as op
import pandas as pd
import os
import sys
import time
import random
import pathlib
import pickle
from functools import reduce
import itertools as it
from atomgraph import AtomGraph
from adjacency import buildAdjacencyList
import ase.cluster
import ase.io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D

# random.seed(9876)


class Chromo(object):
    def __init__(self, atomg, n_dope=0, arr=None, x_dope=None):
        self.n_atoms = atomg.n_atoms
        if x_dope is not None:
            self.n_dope = int(atomg.n_atoms * x_dope)
        else:
            self.n_dope = n_dope
        self.atomg = atomg
        self.arr = np.zeros(atomg.n_atoms).astype(int)

        if n_dope > self.n_atoms:
            raise ValueError("Can't dope more atoms than there are atoms...")

        # if an array is given, use it - else random generate array
        if isinstance(arr, np.ndarray):
            self.arr = arr
            self.n_dope = arr.sum()
        else:
            self.arr[:n_dope] = 1
            np.random.shuffle(self.arr)

        # calculate initial CE
        self.calc_score()

    def get_atomic_ce(self, i):
        return self.atomg.getAtomicCE(i, self.arr)

    def mutate(self, nps=1):
        if not self.n_dope or self.n_dope == self.n_atoms:
            print('Warning: attempting to mutate, but system is monometallic')
            return

        assert self.arr.sum() == self.n_dope
        # shift a "1" <nps> times
        for i in range(nps):
            ones = list(np.where(self.arr == 1)[0])
            s = random.sample(ones, 1)[0]
            self.arr[s] = 0

            # shift '1' over to the left
            shift = s - 1
            while 1:
                if self.arr[shift] == 0:
                    self.arr[shift] = 1
                    break
                shift -= 1

        assert self.arr.sum() == self.n_dope
        self.calc_score()

    def cross(self, chrom2):
        x = 0

        child1 = self.arr.copy()
        child2 = chrom2.arr.copy()
        assert child1.sum() == child2.sum()
        s1 = s2 = 2
        ones = np.where(child1 == 1)[0]
        diff = np.where(child1 != child2)[0][::-1]
        for i in diff:
            if i in ones:
                if s1:
                    child1[i] = 0
                    child2[i] = 1
                    s1 -= 1
            elif s2:
                child1[i] = 1
                child2[i] = 0
                s2 -= 1
            if s1 == s2 == 0:
                break

        if not (child1.sum() == child2.sum() == self.n_dope):
            print()
            print(child1.sum())
            print(child2.sum())
            print(self.n_dope)
            input()
        assert child1.sum() == child2.sum() == self.n_dope
        return [Chromo(self.atomg, n_dope=self.n_dope, arr=child1),
                Chromo(self.atomg, n_dope=self.n_dope, arr=child2)]

    def calc_score(self):
        self.score = self.atomg.getTotalCE(self.arr)


class Pop(object):
    def __init__(self, atomg, n_dope=1, popsize=100, kill_rate=0.2,
                 mate_rate=0.25, mute_rate=0.1, mute_num=1, x_dope=None):
        self.atomg = atomg
        self.popsize = popsize
        if x_dope:
            self.n_dope = int(atomg.n_atoms * x_dope)
        else:
            self.n_dope = n_dope

        self.x_dope = self.n_dope / atomg.n_atoms

        self.nkill = int(popsize * kill_rate)
        self.nmut = int((popsize - self.nkill) * mute_rate)
        self.n_mate = int(popsize * kill_rate * mate_rate)
        self.mute_num = mute_num
        self.mute_rate = mute_rate
        self.kill_rate = kill_rate

        self.initialize_new_run()

    def initialize_new_run(self):
        self.x_dope = self.n_dope / self.atomg.n_atoms
        self.build_pop()
        self.sort_pop()

        self.info = []
        self.stats()

        # track runtime
        self.runtime = 0

    def build_pop(self, n_fill_cn=0):
        self.pop = [Chromo(self.atomg, n_dope=self.n_dope)
                    for i in range(self.popsize - n_fill_cn)]
        self.pop += [Chromo(self.atomg, n_dope=self.n_dope, arr=z)
                     for z in fill_cn(self.atomg, self.n_dope,
                                      return_n=n_fill_cn)]

    def get_min(self):
        return self.info[:, 0].min()

    def step(self, rand=False):
        if rand:
            self.build_pop(n_fill_cn=0)
        else:
            self.pop = self.pop[:self.nkill]

            mated = 0

            # start mating from the 1st 2nd down
            tomate = 1
            while len(self.pop) < self.popsize:
                if mated < self.n_mate:
                    n1, n2 = 0, tomate
                    # n1, n2 = random.sample(range(len(self.pop)), 2)
                    self.pop += self.pop[n1].cross(self.pop[n2])
                    mated += 2
                    tomate += 1
                else:
                    self.pop.append(Chromo(self.pop[0].atomg,
                                           n_dope=self.n_dope))

            self.pop = self.pop[:self.popsize]

            for j in range(self.nmut):
                self.pop[random.randrange(1,
                                          self.popsize)].mutate(self.mute_num)

        self.sort_pop()
        self.stats()

    def run(self, nsteps=50, std_cut=0, rand=False):
        # no GA required for monometallic systems
        if self.n_dope not in [0, self.atomg.n_atoms]:
            start = time.time()
            for i in range(int(nsteps)):
                print('\tdopeX = %.2f\tMin: %.5f eV \t %i' % (self.x_dope,
                                                              self.info[-1][0],
                                                              i), end='\r')
                self.step(rand)
                # if STD less than std_cut end the GA
                if self.info[-1][-1] < std_cut:
                    break
            print('\tdopeX = %.2f\tMin: %.5f eV \t %i' % (self.x_dope,
                                                          self.info[-1][0],
                                                          i + 1), end='\r')
            self.runtime = time.time() - start
        self.info = np.array(self.info)

    def sort_pop(self):
        self.pop = sorted(self.pop,
                          key=lambda j: j.score)

    def stats(self):
        s = np.array([i.score for i in self.pop])
        self.info.append([s[0],
                          s.mean(),
                          s.std()])


def results_str(p, disp=True):
    res = ' Min: %.5f\nMean: %.3f\n STD: %.3f\n' % tuple(p.info[-1, :])
    res += 'Mute: %.2f\nKill: %.2f\n' % (p.mute_rate, p.kill_rate)
    res += ' Pop: %i\n' % p.popsize
    res += 'nRun: %i\n' % max_runs
    res += 'Form: %s%i_%s%i\n' % (metal1, len(atom) - p.n_dope,
                                  metal2, p.n_dope)
    res += 'Done: %i\n' % (len(p.info) - 1) + '\n\n\n'
    if disp:
        print(res.strip('\n'))
    return res


def log_results(p):
    results = results_str(p, disp=False)

    with open('results.txt', 'a') as fid:
        fid.write(results)

    # see if results are new max
    with open('best.txt', 'r') as rfid:
        best = float(rfid.readline().strip('\n').split()[-1])

    # if new max, write it to best.txt
    if best > p.info[-1, 0]:
        print('NEW MIN!'.center(50, '-'))
        with open('best.txt', 'w') as bestfid:
            bestfid.write(results)


def make_plot(p):
    fig, ax = plt.subplots(figsize=(7, 7))

    ax.fill_between(range(len(p.info)), p.info[:, 1] + p.info[:, 2],
                    p.info[:, 1] - p.info[:, 2], color='lightblue',
                    label='STD')

    ax.plot(p.info[:, 1], color='k', label='MEAN')
    ax.plot(p.info[:, 0], ':', color='k', label='MIN')
    ax.legend()
    ax.set_ylabel('Score')
    ax.set_xlabel('Step')
    ax.set_title('Min Val: %.5f' % (p.pop[0].score))
    fig.tight_layout()
    return fig, ax


def make_xyz(atom, chrom, path, verbose=False):
    metal1, metal2 = chrom.atomg.symbols
    atom.info['CE'] = chrom.score
    for i, dope in enumerate(chrom.arr):
        atom[i].symbol = metal2 if dope else metal1

    n_dope = sum(chrom.arr)
    fname = '%s%i_%s%i.xyz' % (metal1, len(atom) - n_dope,
                               metal2, n_dope)
    path = os.path.join(path, fname)
    ase.io.write(path, atom)
    if verbose:
        print('Saved as %s' % path)
    return atom


def gen_random(atomg, n_dope, n=500):
    if n_dope == 0 or n_dope == atomg.n_atoms:
        n = 1
    scores = np.zeros(n)
    for i in range(n):
        scores[i] = Chromo(atomg, n_dope=n_dope).score
        print(i, end='\r')
    return scores.min(), scores.mean(), scores.std()


def build_icosahedron(nshell, return_adj=True):
    # ensure necessary directories exist within local repository
    pathlib.Path('../data/atom_objects/icosahedron/').mkdir(parents=True,
                                                            exist_ok=True)
    apath = '../data/atom_objects/icosahedron/%i.pickle' % nshell
    if os.path.isfile(apath):
        with open(apath, 'rb') as fidr:
            a = pickle.load(fidr)
    else:
        a = ase.cluster.Icosahedron('Cu', nshell)
        with open(apath, 'wb') as fidw:
            pickle.dump(a, fidw)
    if return_adj:
        return a, buildAdjacencyList(a, 'icosahedron_%i' % nshell)
    else:
        return a


def build_fcc_cube(nlayers, return_adj=True):
    pathlib.Path('../data/atom_objects/fcc-cube').mkdir(parents=True,
                                                        exist_ok=True)
    apath = '../data/atom_objects/fcc-cube/%i.pickle' % nlayers
    if os.path.isfile(apath):
        with open(apath, 'rb') as fidr:
            cube = pickle.load(fidr)
    else:
        cube = ase.cluster.FaceCenteredCubic('Cu', [(1, 0, 0),
                                                    (0, 1, 0),
                                                    (0, 0, 1)], [nlayers] * 3)
        with open(apath, 'wb') as fidw:
            pickle.dump(cube, fidw)
    if return_adj:
        return cube, buildAdjacencyList(cube, 'fcc-cube_%i' % nlayers)
    else:
        return cube


def ncr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer//denom


def fill_cn(atomg, n_dope, max_search=50, low_first=True, return_n=None,
            verbose=False):
    formula = 'Cu(%i)Au(%i)' % (atomg.n_atoms - n_dope, n_dope)

    # handle monometallic cases efficiently
    if not n_dope or n_dope == atomg.n_atoms:
        struct_min = np.zeros(atomg.n_atoms) if \
            not n_dope else np.ones(atomg.n_atoms)
        struct_min = struct_min.astype(int)
        ce = atomg.getTotalCE(struct_min)
        checkall = True
    else:
        cn_list = atomg.getAllCNs()
        cnset = sorted(set(cn_list))
        if not low_first:
            cnset = cnset[::-1]
        struct_min = np.zeros(atomg.n_atoms).astype(int)
        ce = None
        for cn in cnset:
            spots = np.where(cn_list == cn)[0]
            if len(spots) == n_dope:
                struct_min[spots] = 1
                checkall = True
                break
            elif len(spots) > n_dope:
                low = 0
                low_struct = None

                # check to see how many combinations exist
                options = ncr(len(spots), n_dope)
                # print('%.2e' % options)

                # return sample of 'return_n' options
                if return_n:
                    if return_n > options:
                        raise ValueError('not enough options to '
                                         'produce desired sample size')
                    sample = []
                    while len(sample) < return_n:
                        base = struct_min.copy()
                        pick = random.sample(list(spots), n_dope)
                        base[pick] = 1
                        sample.append(base)
                    return sample

                # if n combs < max_search, check them all
                if options <= max_search:
                    if verbose:
                        print('Checking all options')
                    searchopts = it.combinations(spots, n_dope)
                    checkall = True
                else:
                    if verbose:
                        print("Checking {0:.2%}".format(max_search / options))
                    searchopts = range(max_search)
                    checkall = False

                # stop looking after 'max_search' counts
                for c in searchopts:  # it.combinations(spots, n_dope)
                    base = struct_min.copy()
                    if checkall:
                        pick = list(c)
                    else:
                        pick = random.sample(list(spots), n_dope)
                    base[pick] = 1
                    checkce = atomg.getTotalCE(base)
                    if checkce < low:
                        low = checkce
                        low_struct = base.copy()
                struct_min = low_struct
                ce = low
                break
            else:
                struct_min[spots] = 1
                n_dope -= len(spots)
    if not ce:
        ce = atomg.getTotalCE(struct_min)
    return struct_min, ce


def make_3d_plot(path, metals=None):
    shape, metals = os.path.basename(path).split('_')[:2]
    metal1, metal2 = metals[:2], metals[2:]
    if isinstance(path, str):
        df = pd.read_excel(path)
    else:
        df = path
    size = df.diameter.values
    comps = df['composition_%s' % metal2].values
    ees = df.EE.values

    colormap = plt.get_cmap('coolwarm')
    normalize = matplotlib.colors.Normalize(vmin=ees.min(), vmax=ees.max())

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    try:
        ax.plot_trisurf(comps, size, ees,
                        cmap=colormap, norm=normalize)
    except RuntimeError:
        ax.scatter3D(comps, size, ees,
                     cmap=colormap, norm=normalize)
    ax.set_xlabel('$X_{%s}$' % metal2)
    ax.set_ylabel('Size (nm)')
    ax.set_zlabel('EE (eV)')
    ax.set_title('%s %s %s' % (metal1, metal2, shape.title()))
    return fig


def run_ga(metals, shape='fcc-cube', plotit=True,
           save_data=True, save_structs=True, max_shells=None):
    # clear previous plots and define desktop and Box paths
    plt.close('all')
    desk = os.path.join(os.path.expanduser('~'), 'desktop')
    box = os.path.join(os.path.expanduser('~'), 'Box Sync',
                       'Michael_Cowan_PhD_research', 'data', 'np_ce')

    # ensure metals is a list of two elements
    assert len(metals) == 2
    assert sum(map(lambda i: isinstance(i, str) and len(i) == 2, metals)) == 2

    # always sort metals by alphabetical order for consistency
    metal1, metal2 = metals

    # print run info
    print('\n___________RUN INFO___________\n')
    print('         Metals: %s, %s' % (metal1, metal2))
    print('          Shape: %s' % shape)
    print('Save Structures: %s' % bool(save_structs))
    print('Save GA Results: %s' % bool(save_data))
    print(' Create 3D Plot: %s' % bool(plotit))
    if max_shells:
        print('     Max Shells: %i' % max_shells)
    print('______________________________')

    # GA properties
    max_runs = 50
    popsize = 100
    kill_rate = 0.2
    mate_rate = 0.8
    mute_rate = 0.2
    mute_num = 1

    # dict giving number of atoms in icosahedron NP based on nshells
    with open('../data/ico_shell2numatoms.pickle', 'rb') as fidr:
        shell2atoms = pickle.load(fidr)

    # CEs for monometallic NPs
    if os.path.isfile('../data/monomet_CE/%s.pickle' % shape):
        with open('../data/monomet_CE/%s.pickle' % shape, 'rb') as fidr:
            monos = pickle.load(fidr)
    else:
        monos = {}

    # data for 3D plot
    eedata = []
    cedata = []
    comps = []
    tot_natoms = []
    tot_size = []

    # initialize 3D plot
    if plotit:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        colormap = plt.get_cmap('coolwarm')

    # 24 shells = about 10 nm
    # 13 shells = about 5 nm
    # range of number of shells to test
    shape2shell = {'icosahedron': [2, 14],
                   'fcc-cube': [1, 15]
                   }
    nshell_range = shape2shell[shape]
    if max_shells:
        nshell_range[1] = max_shells
    for nshells in range(*nshell_range):
        # build atom, adjacency list, and atomgraph
        atom, adj = build_fcc_cube(nshells)  # build_icosahedron(nshells)
        ag = AtomGraph(adj, metal1, metal2)

        natoms = len(atom)
        if natoms not in monos:
            monos[natoms] = {}

        if save_structs:
            path = '%s%s/%s/%i/' % (metal1, metal2, shape, natoms)
            struct_path = os.path.join(box, path, 'structures')
            pathlib.Path(struct_path).mkdir(parents=True,
                                            exist_ok=True)

        # x = metal2 concentration [0, 1]
        x = np.linspace(0, 1, 11)
        n = (x * natoms).astype(int)

        # USE THIS TO TEST EVERY CONCENTRATION
        if natoms < 15:  # 0:
            n = np.arange(0, natoms + 1)
            x = n / n.max()

        rands = np.zeros((len(x), 3))
        ces = np.zeros(len(x))

        # INITIALIZE POP object
        pop = Pop(ag, n_dope=n[0], popsize=popsize,
                  kill_rate=kill_rate, mate_rate=mate_rate,
                  mute_rate=mute_rate, mute_num=mute_num)

        print('%s %s in %i atom %s' % (metal1, metal2, natoms, shape))

        for i, dope in enumerate(n):
            if i:
                pop.n_dope = dope
                pop.initialize_new_run()
            pop.run(max_runs)
            ces[i] = pop.get_min()
            if dope == 0 and metal1 not in monos[natoms]:
                monos[natoms][metal1] = ces[i]
                print('Adding %s for %i atom %s' % (metal1, natoms, shape))
            if dope == natoms and metal2 not in monos[natoms]:
                monos[natoms][metal2] = ces[i]
                print('Adding %s for %i atom %s' % (metal2, natoms, shape))

            # save min structure
            if save_structs:
                make_xyz(atom.copy(), pop.pop[0], struct_path)

        # calculate excess energy (ees)
        ees = ces - (x * monos[len(atom)][metal2]) - \
            ((1 - x) * monos[len(atom)][metal1])

        tot_natoms += [natoms] * len(x)
        comps += list(x)
        tot_size += [atom.cell[0][0] / 10] * len(x)
        cedata += list(ces)
        eedata += list(ees)

    # save data from each GA run
    if save_data:
        excel = '../data/bimetallic_results/' \
                '{0}/{0}_{1}{2}_data_{3}-{4}shells.xlsx'.format(shape,
                                                                metal1,
                                                                metal2,
                                                                *nshell_range)
        df = pd.DataFrame({'n_atoms': tot_natoms, 'diameter': tot_size,
                           'composition_%s' % metal2: comps, 'CE': cedata,
                           'EE': eedata})
        writer = pd.ExcelWriter(excel, engine='xlsxwriter')
        df.to_excel(writer, index=False)
        writer.save()
        writer.close()

    # save new monometallic NP data
    with open('../data/monomet_CE/%s.pickle' % shape, 'wb') as fidw:
        pickle.dump(monos, fidw)

    # create 3D plot of size, comp, EE
    if plotit:
        normalize = matplotlib.colors.Normalize(vmin=min(eedata),
                                                vmax=max(eedata))
        ax.plot_trisurf(comps, tot_size, eedata,
                        cmap=colormap, norm=normalize)
        ax.set_xlabel('$X_{%s}$' % metal2)
        ax.set_ylabel('Size (nm)')
        ax.set_zlabel('EE (eV)')
        ax.set_title('%s %s %s' % (metal1, metal2, shape.title()))
        fig.show()

if __name__ == '__main__':
    # metals = 'Ag', 'Cu'
    # metals = 'Ag', 'Au'
    metals = 'Cu', 'Au'

    metals = 'Au', 'Cu'

    # shape = 'fcc-cube'
    shape = 'icosahedron'

    run_ga(metals, shape, save_data=False, save_structs=False,
           max_shells=3, plotit=False)
