import operator as op
import pandas as pd
import os
import time
import random
from datetime import datetime as dt
import pathlib
import pickle
import functools
import itertools as it
import atomgraph
import structure_gen
import ase.cluster
import ase.io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# random.seed(9876)

# center output strings around <CENTER> characters
CENTER = 40


class Chromo(object):
    def __init__(self, atomg, n_dope=0, arr=None, x_dope=None):
        """
        Chromosome object for GA simulations
        Represents a single structure with a given chemical ordering (arr)

        Args:
        atomg:

        Kargs:
        n_dope (int): (default: 0)
        arr (np.ndarray || list || None): (default: None)
        x_dope (float || None): (default: None)

        Raises:
                ValueError: n_dope is greater than total atoms
        """
        self.num_atoms = atomg.num_atoms
        if x_dope is not None:
            self.n_dope = int(atomg.num_atoms * x_dope)
        else:
            self.n_dope = n_dope
        self.atomg = atomg
        self.arr = np.zeros(atomg.num_atoms).astype(int)

        if n_dope > self.num_atoms:
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

    def mutate(self, nps=1):
        """
        Algorithm to randomly swith a '1' & a '0' within ordering arr
        - mutates the ordering array
        NOTE: slow algorithm - can probably be improved

        Returns: None
        """
        if not self.n_dope or self.n_dope == self.num_atoms:
            print('Warning: attempting to mutate, but system is monometallic')
            return

        # shift a "1"
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

        # update CE 'score' of Chrom
        self.calc_score()

    def cross(self, chrom2):
        """
        Crossover algorithm to mix two parent chromosomes into
        two new child chromosomes, taking traits from each parent
        - conserves doping concentration

        Args:
        chrom2 (Chrom): second parent Chrom obj

        Returns:
                (list): two children Chrom objs with new ordering <arr>s
        """
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
        """
        Returns CE of structure based on Bond-Centric Model
        - Yan, Z. et al., Nano Lett. 2018, 18 (4), 2696-2704.
        """
        self.score = self.atomg.getTotalCE(self.arr)


class Pop(object):
    def __init__(self, atomg, n_dope=1, popsize=100, kill_rate=0.2,
                 mate_rate=0.8, mute_rate=0.2, mute_num=1, x_dope=None):
        """

        :param atomg:
        :param n_dope:
        :param popsize:
        :param kill_rate:
        :param mate_rate:
        :param mute_rate:
        :param mute_num:
        :param x_dope:
        """
        self.atomg = atomg
        self.popsize = popsize
        if x_dope:
            self.n_dope = int(atomg.num_atoms * x_dope)
        else:
            self.n_dope = n_dope

        self.x_dope = self.n_dope / atomg.num_atoms

        self.nkill = int(popsize * kill_rate)
        self.nmut = int((popsize - self.nkill) * mute_rate)
        self.n_mate = int(popsize * kill_rate * mate_rate)
        self.mute_num = mute_num
        self.mute_rate = mute_rate
        self.kill_rate = kill_rate

        self.initialize_new_run()

    def initialize_new_run(self):
        """

        :return:
        """
        self.x_dope = self.n_dope / self.atomg.num_atoms
        self.build_pop()
        self.sort_pop()

        self.stats = []
        self.min_struct_ls = []
        self.update_stats()

        # track runtime
        self.runtime = 0

        # keep track of whether a sim has been run
        self.has_run = False

    def build_pop(self):
        """

        :return:
        """
        # create <popsize> - 2 random structures
        self.pop = [Chromo(self.atomg, n_dope=self.n_dope)
                    for i in range(self.popsize - 2)]

        # add max and min CN filled structs
        self.pop += [Chromo(self.atomg, n_dope=self.n_dope,
                            arr=fill_cn(self.atomg, self.n_dope,
                                        low_first=True, return_n=1)[0])]

        self.pop += [Chromo(self.atomg, n_dope=self.n_dope,
                            arr=fill_cn(self.atomg, self.n_dope,
                                        low_first=False, return_n=1)[0])]

    def get_min(self):
        """

        :return:
        """
        return self.stats[:, 0].min()

    def step(self, rand=False):
        """

        :param rand:
        :return:
        """
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
        self.update_stats()

    def run(self, nsteps=50, std_cut=0, rand=False):
        """

        :param nsteps:
        :param std_cut:
        :param rand:
        :return:
        """

        # format of string to be written to console during sim
        update_str = 'dopeX = %.2f\tMin: %.5f eV/atom\t%05i'

        # no GA required for monometallic systems
        if self.n_dope not in [0, self.atomg.num_atoms]:
            start = time.time()
            for i in range(int(nsteps)):
                val = update_str % (self.x_dope, self.stats[-1][0], i)
                print(val.center(CENTER), end='\r')
                self.step(rand)
                # if STD less than std_cut end the GA
                if self.stats[-1][2] < std_cut:
                    break
            val = update_str % (self.x_dope, self.stats[-1][0], i + 1)
            print(val.center(CENTER), end='\r')
            self.runtime = time.time() - start
        self.stats = np.array(self.stats)
        self.has_run = True

    def sort_pop(self):
        """

        :return:
        """
        self.pop = sorted(self.pop,
                          key=lambda j: j.score)

    def update_stats(self):
        """
        - Adds min, mean, and std dev of current generation to self.stats
        - Adds min structure Chromo objf of current generation

        Returns: None
        """
        s = np.array([i.score for i in self.pop])
        self.stats.append([s[0],
                          s.mean(),
                          s.std()])
        self.min_struct_ls.append(Chromo(self.atomg, self.n_dope,
                                         arr=self.pop[0].arr.copy()))

    def summ_results(self, disp=False):
        """
            Creates string listing GA simulation stats and results info
            str includes:
                - minimum, mean, and std dev. of CEs for last population
                - mute and mate (crossover) rates of GA
                - total steps taken by GA
                - structure formula

        Kargs:
        disp (bool): if True, results string is written to console
                     (default: False)

        Returns:
                (str): result string

        Raises:
                Exception: if sim has not been run
                           i.e. self.has_run == False
        """
        if not self.has_run:
            raise Exception('No simulation has been run')

        res = ' Min: %.5f\nMean: %.3f\n STD: %.3f\n' % tuple(self.stats[-1, :])
        res += 'Mute: %.2f\nKill: %.2f\n' % (self.mute_rate, self.kill_rate)
        res += ' Pop: %i\n' % self.popsize
        res += 'nRun: %i\n' % (len(self.stats) - 1)
        res += 'Form: %s%i_%s%i\n' % (metal1, len(atom) - p.n_dope,
                                      metal2, p.n_dope)
        if self.has_run:
            res += 'Time: %.3e\n' % (self.runtime) + '\n\n\n'
        if disp:
            print(res)
        return res

    def plot_results(self):
        """
        Method to create a plot of GA simulation
        - plots average, std deviation, and minimum score
          of the population at each step

        Returns:
                (matplotlib.figure.Figure),
                (matplotlib.axes._subplots.AxesSubplot): fig and ax objs
        """
        fig, ax = plt.subplots(figsize=(9, 9))

        # number of steps GA took
        steps = range(len(self.stats))

        # minimum, average, and std deviation scores of population at each step
        # NOTE: GA's goal is to minimize score
        low = self.stats[:, 0]
        mean = self.stats[:, 1]
        std = self.stats[:, 2]

        # plot minimum CE as a dotted line
        ax.plot(low, ':', color='k', label='MIN')
        # light blue fill of one std deviation
        ax.fill_between(range(len(self.stats)), mean + std, mean - std,
                        color='lightblue', label='STD')

        # plot mean as a solid line and minimum as a dotted line
        ax.plot(mean, color='k', label='MEAN')
        ax.legend()
        ax.set_ylabel('Cohesive Energy (eV / atom)')
        ax.set_xlabel('Generation')
        # ax.set_title('Min CE: %.5f' % (self.get_min()))
        fig.tight_layout()
        return fig, ax


def make_xyz(atom, chrom, path, verbose=False):
    """
    Creates an XYZ file given Atoms obj skeleton and
    GA Chrom obj for metals and ordering

    Args:
    atom (ase.Atoms): Atoms obj skeleton
    chrom (Chrom): Chrom obj from GA containing ordering and metals
    path (str): path to save XYZ file

    Kargs:
    verbose (bool): if True, print save path on success

    Returns: None
    """
    atom = atom.copy()
    metal1, metal2 = chrom.atomg.symbols
    atom.info['CE'] = chrom.score
    for i, dope in enumerate(chrom.arr):
        atom[i].symbol = metal2 if dope else metal1

    # create file name if not included in path
    if not path.endswith('.xyz'):
        n_dope = sum(chrom.arr)
        fname = '%s%i_%s%i.xyz' % (metal1, len(atom) - n_dope,
                                   metal2, n_dope)
        path = os.path.join(path, fname)

    # save xyz file to path
    ase.io.write(path, atom)
    if verbose:
        print('Saved as %s' % path)
    return atom


def gen_random(atomg, n_dope, n=500):
    """
    Generates random structures (constrained by size, shape, and concentration)
    and returns minimum structure and stats on CEs

    Args:
    atomg (atomgraph.AtomGraph): AtomGraph object
    n_dope (int): number of doped atoms
    n (int): sample size

    Returns:
            (Chrom), (np.ndarray): Chrom object of minimum structure found
                                   1D array of all CEs calculated in sample
    """
    if n_dope == 0 or n_dope == atomg.num_atoms:
        n = 1
    scores = np.zeros(n)

    # keep track of minimum structure and minimum CE
    min_struct = None
    min_ce = 10
    for i in range(n):
        c = Chromo(atomg, n_dope=n_dope)
        scores[i] = c.score
        if c.score < min_ce:
            min_struct = Chromo(atomg, n_dope=n_dope, arr=c.arr.copy())
            min_ce = min_struct.score
    return min_struct, scores


def ncr(n, r):
    """
    N choose r function (combinatorics)

    Args:
    n (int): from 'n' choices
    r (int): choose r without replacement

    Returns:
        (int): total combinations
    """
    r = min(r, n - r)
    numer = functools.reduce(op.mul, range(n, n - r, -1), 1)
    denom = functools.reduce(op.mul, range(1, r + 1), 1)
    return numer // denom


def fill_cn(atomg, n_dope, max_search=50, low_first=True, return_n=None,
            verbose=False):
    """
    Algorithm to fill the lowest (or highest) coordination sites with dopants

    Args:
    atomg (atomgraph.AtomGraph): AtomGraph object
    n_dope (int): number of dopants

    Kargs:
    max_search (int): if there are a number of possible structures with
                      partially-filled sites, the function will search
                      max_search options and return lowest CE structure
                      (default: 50)
    low_first (bool): if True, fills low CNs, else fills high CNs
                      (default: True)
    return_n (bool): if > 0, function will return a list of possible
                     structures
                     (default: None)
    verbose (bool): if True, function will print info to console
                    (default: False)
    Returns:
            if return_n > 0:
                (list) -- list of chemical ordering np.ndarrays
            else:
                (np.ndarray), (float) -- chemical ordering np.ndarray with its
                                         calculated CE

    Raises:
           ValueError: not enough options to produce <return_n> sample size
    """
    # handle monometallic cases efficiently
    if not n_dope or n_dope == atomg.num_atoms:
        struct_min = np.zeros(atomg.num_atoms) if \
            not n_dope else np.ones(atomg.num_atoms)
        struct_min = struct_min.astype(int)
        ce = atomg.getTotalCE(struct_min)
        checkall = True
    else:
        cn_list = atomg.cns
        cnset = sorted(set(cn_list))
        if not low_first:
            cnset = cnset[::-1]
        struct_min = np.zeros(atomg.num_atoms).astype(int)
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

                # return sample of 'return_n' options
                if return_n:
                    # check to see how many combinations exist
                    options = ncr(len(spots), n_dope)
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
    if return_n:
        return [struct_min]
    return struct_min, ce


def make_3d_plot(path, metals=None):
    """

    :param path:
    :param metals:
    :return:
    """
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


def run_ga(metals, shape, datapath=None, plotit=False,
           save_data=True, log_results=True,
           batch_runinfo=None, max_shells=None):
    """
    Batch submission function to run GAs of a given metal combination and
    shape, sweeping over different sizes (measured in number of shells)
    - capable of saving minimum structures as XYZs, logging GA stats, saving
      all run info as excel, and creating a 3D surface plot of results

    Args:
    metals (list) || (tuple): list of two metals used in the bimetallic NP
                              NOTE: currently supports Cu, Ag, and Au
    shape (str): shape of NP that is being studied
                 NOTE: currently supports
                        - icosahedron
                        - cuboctahedron
                        - fcc-cube
                        - elongated-trigonal-pyramic

    Kargs:
    datapath (str || None): path to save sims.log info and structures
                            (default: None)
    plotit (bool): if true, a 3D surface plot is made of GA sims
                   dope concentration vs. size vs. excess energy
                   (default: False)
    save_data (bool): - if true, GA sim data is saved to
                        '../data/bimetallic_results/<metal & shape based name>'
                      - new minimum structures are also saved as XYZs
                      (default: True)
    log_results (bool): if true, GA sim stats are logged to
                        sims.log
                        (default: True)
    batch_runinfo (str || None): if str, add to log under
                                 'Completed Run: <batch_runinfo>'
                                 (default: None)
    max_shells (int || None): if int, limits size of NPs studied in GA runs
                              (default: None)

    Returns: None
    """
    # clear previous plots and define desktop and data paths
    plt.close('all')
    desk = os.path.join(os.path.expanduser('~'), 'desktop')

    # if datapatha not given, create one on desktop
    if not datapath:
        datapath = os.path.join(desk, 'ga_data')
        pathlib.Path(datapath).mkdir(parents=True, exist_ok=True)

    # ensure metals is a list of two elements
    assert len(metals) == 2
    assert sum(map(lambda i: isinstance(i, str) and len(i) == 2, metals)) == 2

    # always sort metals by alphabetical order for consistency
    metal1, metal2 = sorted(metals)

    # number of shells range to sim ga for each shape
    shape2shell = {'icosahedron': [2, 14],
                   'fcc-cube': [1, 15],
                   'cuboctahedron': [1, 15],
                   'elongated-pentagonal-bipyramid': [2, 12]
                   }
    nshell_range = shape2shell[shape]

    if max_shells:
        nshell_range[1] = max_shells
    nstructs = len(range(*nshell_range))
    if not nstructs:
        print(nshell_range)
        return

    # print run info
    print('\n----------------RUN INFO----------------')
    print('             Metals: %s, %s' % (metal1, metal2))
    print('              Shape: %s' % shape)
    print('    Save GA Results: %s' % bool(save_data))
    print('     Create 3D Plot: %s' % bool(plotit))
    print('        Shell Range: %s' % str(nshell_range))
    print('----------------------------------------')

    # attempt to read in previous results
    excel_columns = ['num_atoms', 'diameter', 'composition_%s' % metal2,
                     'n_%s' % metal1, 'n_%s' % metal2, 'CE', 'EE']

    shapepath = '../data/bimetallic_results/%s/' % shape
    pathlib.Path(shapepath).mkdir(parents=True, exist_ok=True)
    excel = '{0}/{1}_{2}{3}_data.xlsx'.format(shapepath, shape,
                                              metal1, metal2)
    if os.path.isfile(excel):
        df = pd.read_excel(excel)
    else:
        df = pd.DataFrame([], columns=excel_columns)

    # GA properties
    max_runs = 1000
    popsize = 50
    kill_rate = 0.2
    mate_rate = 0.8
    mute_rate = 0.2
    mute_num = 1

    # CEs for monometallic NPs
    mono_path = '../data/monometallic_CE/'
    mono_pickle = mono_path + '%s.pickle' % shape
    pathlib.Path(mono_path).mkdir(parents=True, exist_ok=True)
    if os.path.isfile(mono_pickle):
        with open(mono_pickle, 'rb') as fidr:
            monos = pickle.load(fidr)
    else:
        monos = {}

    # track if any new mono calcs are added
    new_mono_calcs = False

    # keep track of total new minimum CE structs (based on saved structs)
    tot_new_structs = 0

    # count total structs
    tot_st = 0

    # data for 3D plot
    eedata = []
    cedata = []
    comps = []
    tot_natoms = []
    nmetal1 = []
    nmetal2 = []
    tot_size = []

    # log runtime
    starttime = time.time()

    for struct_i, nshells in enumerate(range(*nshell_range)):
        # build atom, adjacency list, and atomgraph
        atom, bond_list = structure_gen.build_structure(shape, nshells,
                                                        return_bond_list=True)

        ag = atomgraph.AtomGraph(bond_list, metal1, metal2)

        natoms = len(atom)
        if natoms not in monos:
            monos[natoms] = {}

        # build structure path
        path = '%s%s/%s/%i/' % (metal1, metal2, shape, natoms)
        struct_path = os.path.join(datapath, path, 'structures')
        pathlib.Path(struct_path).mkdir(parents=True,
                                        exist_ok=True)

        # USE THIS TO TEST EVERY CONCENTRATION
        if natoms < 150:
            n = np.arange(0, natoms + 1)
            x = n / float(natoms)
        else:
            # x = metal2 concentration [0, 1]
            x = np.linspace(0, 1, 11)
            n = (x * natoms).astype(int)

            # recalc concentration to match n
            x = n / float(natoms)

        # total structures checked ( - 2 to exclude monometallics)
        tot_st += float(len(n) - 2)

        rands = np.zeros((len(x), 3))
        ces = np.zeros(len(x))

        # INITIALIZE POP object
        pop = Pop(ag, n_dope=n[0], popsize=popsize,
                  kill_rate=kill_rate, mate_rate=mate_rate,
                  mute_rate=mute_rate, mute_num=mute_num)

        starting_outp = '%s%s in %i atom %s' % (metal1, metal2, natoms, shape)
        print(starting_outp.center(CENTER))

        # keep track of new minimum CE structs (based on saved structs)
        new_min_structs = 0
        for i, dope in enumerate(n):
            if i:
                pop.n_dope = dope
                pop.initialize_new_run()
            pop.run(max_runs)
            ces[i] = pop.get_min()
            if dope == 0 and metal1 not in monos[natoms]:
                monos[natoms][metal1] = ces[i]
                new_mono_calcs = True
            if dope == natoms and metal2 not in monos[natoms]:
                monos[natoms][metal2] = ces[i]
                new_mono_calcs = True

            fname = '%s%i_%s%i.xyz' % (metal1, len(atom) - dope,
                                       metal2, dope)

            # check to see if older ga run founded lower CE structure
            if save_data:
                # if older data exists, see if new struct has lower CE
                if not df.loc[(df['num_atoms'] == natoms) &
                              (df['n_%s' % metal2] == dope),
                              'CE'].empty:

                    oldrunmin = df.loc[(df['num_atoms'] == natoms) &
                                       (df['n_%s' % metal2] == dope),
                                       'CE'].values[0]

                    # update df if new struct has lower CE
                    if (pop.get_min() - oldrunmin) < -1E-5:
                        df.loc[(df['num_atoms'] == natoms) &
                               (df['n_%s' % metal2] == dope),
                               'CE'] = pop.get_min()
                        make_xyz(atom.copy(), pop.pop[0], struct_path)
                        new_min_structs += 1
                        tot_new_structs += 1
                    else:
                        ces[i] = oldrunmin

                # if no older data, save new struct
                else:
                    make_xyz(atom.copy(), pop.pop[0], struct_path)

        print(' ' * 100, end='\r')
        print('-' * CENTER)
        outp = 'Completed Size %i of %i' % (struct_i + 1, nstructs)
        if new_min_structs:
            outp += ' (%i new mins)' % new_min_structs
        print(outp.center(CENTER))
        print('-' * CENTER)

        # calculate excess energy (ees)
        ees = ces - (x * monos[natoms][metal2]) - \
              ((1 - x) * monos[natoms][metal1])

        tot_natoms += [natoms] * len(x)
        comps += list(x)
        tot_size += [atom.cell.max() / 10] * len(x)
        nmetal1 += list(natoms - n)
        nmetal2 += list(n)
        cedata += list(ces)
        eedata += list(ees)

    complete = time.time() - starttime

    # save data from each GA run
    if save_data:
        newdf = pd.DataFrame({'num_atoms': tot_natoms,
                              'diameter': tot_size,
                              'composition_%s' % metal2: comps,
                              'n_%s' % metal1: nmetal1,
                              'n_%s' % metal2: nmetal2,
                              'CE': cedata,
                              'EE': eedata})

        data = df.append(newdf, ignore_index=True)
        data.drop_duplicates(['num_atoms', 'n_%s' % metal2], inplace=True)
        data.sort_values(by=['num_atoms', 'n_%s' % metal2], inplace=True)

        writer = pd.ExcelWriter(excel, engine='xlsxwriter')
        data.to_excel(writer, index=False)
        writer.save()
        writer.close()

    # save new monometallic NP data if any new calcs have been added
    if new_mono_calcs:
        print('New monometallic calcs have been saved'.center(CENTER))
        print('-' * CENTER)
        with open(mono_pickle, 'wb') as fidw:
            pickle.dump(monos, fidw)

    # create 3D plot of size, comp, EE
    if plotit:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        colormap = plt.get_cmap('coolwarm')
        normalize = matplotlib.colors.Normalize(vmin=min(eedata),
                                                vmax=max(eedata))
        ax.plot_trisurf(comps, tot_size, eedata,
                        cmap=colormap, norm=normalize)
        ax.set_xlabel('$X_{%s}$' % metal2)
        ax.set_ylabel('Size (nm)')
        ax.set_zlabel('EE (eV)')
        ax.set_title('%s %s %s' % (metal1, metal2, shape.title()))
        fig.show()

    # log sim results
    today = dt.now().strftime("%m/%d/%Y")
    timeofnow = dt.now().strftime("%I:%M %p")

    logtxt = '----------------RUN INFO----------------\n'

    # today's date
    logtxt += '            Date: %s\n' % today

    # time of completion
    logtxt += '            Time: %s\n' % timeofnow

    # total runtime (days : hours: minutes : seconds)
    logtxt += '         Runtime: %02i:%02i:%02i:%02i\n'
    logtxt = logtxt % (complete // 86400, complete % 86400 // 3600,
                       complete % 3600 // 60, complete % 60)

    # two metals studied (always in alphabetical order)
    logtxt += '          Metals: %s, %s\n' % (metal1, metal2)

    # shape studied (icosahedron, etc.)
    logtxt += '           Shape: %s\n' % shape

    # shapes are built shell-by-shell; number of shells indicates size range
    logtxt += '     Shell Range: %i - %i\n' % tuple(nshell_range)

    # number of new minimum CE structures found (compared to previous runs)
    logtxt += ' New Min Structs: %i\n' % tot_new_structs

    # percentage of new structures (relative to total structures analyzed)
    logtxt += '   %% New Structs: %.2f%%\n' % (100 * tot_new_structs / tot_st)

    # if run is part of a batch submission,
    # indicate how far along the batch job is
    if batch_runinfo:
        logtxt += '   Completed Run: %s\n' % batch_runinfo
    logtxt += '----------------------------------------\n'

    # write to sims.log to datapath
    logpath = os.path.join(datapath, 'sims.log')
    with open(logpath, 'a') as flog:
        flog.write(logtxt)


if __name__ == '__main__':
    plt.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]
    plt.rcParams['figure.figsize'] = (9, 9)
    plt.rcParams['savefig.dpi'] = 600
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.labelsize'] = 20
    plt.rcParams['axes.titlesize'] = 20
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['xtick.labelsize'] = 20
    plt.rcParams['ytick.labelsize'] = 20
    plt.rcParams['lines.linewidth'] = 2

    metal1 = 'Ag'
    metal2 = 'Cu'

    atom, bond_list = structure_gen.build_structure('icosahedron', 10)
    ag = atomgraph.AtomGraph(bond_list, metal1, metal2)

    p = Pop(ag, x_dope=0.2, popsize=50)
    p.run(5000)
    # make_xyz(atom, p.pop[0], path='c:/users/yla/desktop/')
    f, a = p.plot_results()
    f.savefig('c:/users/yla/desktop/ga_run_SAVINGCHROMOS.png')
    f.show()

    """
    metal_opts = [('Ag', 'Cu'),
                  ('Ag', 'Au'),
                  ('Au', 'Cu')
                  ]

    shape_opts = ['icosahedron', 'fcc-cube', 'cuboctahedron',
                  'elongated-pentagonal-bipyramid']
    batch_tot = len(metal_opts) * len(shape_opts)
    batch_i = 1
    for metals in metal_opts:
        for shape in shape_opts:
            run_ga(metals, shape, save_data=True, plotit=False,
                   log_results=True,
                   batch_runinfo='%i of %i' % (batch_i, batch_tot))
            batch_i += 1
    """
