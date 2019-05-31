import operator as op
import pandas as pd
import os
import time
import random
from datetime import datetime as dt
import pathlib
import re
import pickle
import functools
import itertools as it
import atomgraph
import structure_gen
import ase.io
import numpy as np
import plot_defaults
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from npdb import db_inter

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
        self.atomg = atomg
        self.num_atoms = atomg.num_atoms
        if x_dope is not None:
            self.n_dope = int(self.num_atoms * x_dope)
        else:
            self.n_dope = n_dope
        # self.atomg = atomg
        self.arr = np.zeros(self.num_atoms).astype(int)

        if n_dope > self.num_atoms:
            raise ValueError("Can't dope more atoms than there are atoms...")

        # if an array is given, use it - else random generate array
        if isinstance(arr, np.ndarray) or isinstance(arr, list):
            self.arr = np.array(arr)
            self.n_dope = self.arr.sum()
        else:
            self.arr[:n_dope] = 1
            np.random.shuffle(self.arr)

        # calculate initial CE
        self.calc_score()

    def mutate(self, nps=1):
        """
        Algorithm to randomly switch a '1' & a '0' within ordering arr
        - mutates the ordering array
        NOTE: slow algorithm - can probably be improved

        Args:
            nps (int): number of swaps to make
                       (default: 1)

        Returns: None

        Raises:
                ValueError: if not AtomGraph, Chromo can not and
                            should not be mutated
        """
        if not self.atomg:
            raise ValueError("Mutating Chromo should only be done through"
                             "Pop simulations")

        if not self.n_dope or self.n_dope == self.num_atoms:
            print('Warning: attempting to mutate, but system is monometallic')
            return

        # keep track of indices used so there are no repeats
        used = []

        # complete <nps> swaps
        for i in range(nps):

            # keep track of 0 and 1 changes
            # (don't have to make the swap at the same time)
            changed = [False, False]

            # loop until a 0 and 1 have been changed
            while sum(changed) != 2:

                # 1) pick a random index
                i = random.randint(0, self.num_atoms - 1)

                # don't use the same index during mutation!
                if i not in used:

                    # 2a) if the index leads to a '0' and no '0'
                    # has been changed, switch it to a '1'
                    if self.arr[i] == 0 and not changed[0]:
                        self.arr[i] = 1
                        changed[0] = True
                        used.append(i)

                    # 2b) if the index leads to a '1' and no '1'
                    # has been changed, switch it to a '0'
                    elif self.arr[i] == 1 and not changed[1]:
                        self.arr[i] = 0
                        changed[1] = True
                        used.append(i)

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

        assert child1.sum() == child2.sum() == self.n_dope
        return [Chromo(self.atomg, n_dope=self.n_dope, arr=child1),
                Chromo(self.atomg, n_dope=self.n_dope, arr=child2)]

    def calc_score(self):
        """
        Returns CE of structure based on Bond-Centric Model
        - Yan, Z. et al., Nano Lett. 2018, 18 (4), 2696-2704.
        """
        self.ce = self.atomg.getTotalCE(self.arr)


class Pop(object):
    def __init__(self, atom, bond_list, metals, shape, n_dope=1,
                 popsize=50, kill_rate=0.2, mate_rate=0.8,
                 mute_rate=0.2, mute_num=1, x_dope=None):
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
        self.atom = atom
        # organize AtomGraph info
        self.metals = sorted(metals)
        metal1, metal2 = sorted(metals)

        self.shape = shape

        self.atomg_props = dict(bond_list=bond_list,
                                metal1=metal1,
                                metal2=metal2)

        self.atomg = self.__make_atomg__()
        self.num_atoms = len(atom)

        self.popsize = popsize
        if x_dope is not None:
            self.n_dope = np.array([self.num_atoms * x_dope]).astype(int)[0]
        else:
            self.n_dope = n_dope

        self.x_dope = self.n_dope / self.num_atoms

        self.nkill = int(popsize * kill_rate)
        self.nmut = int((popsize - self.nkill) * mute_rate)
        self.n_mate = int(popsize * kill_rate * mate_rate)
        self.mute_num = mute_num
        self.mute_rate = mute_rate
        self.kill_rate = kill_rate

        # keep track of how many times the sim has been continued
        self.continued = 0

        # previous GA sim results (npdb.datatables.BimetallicResults)
        self.prev_results = None

        self.initialize_new_run()

    def __make_atomg__(self):
        """
        Initializes AtomGraph obj (used to calc CE)

        Returns:
                AtomGraph obj
        """
        return atomgraph.AtomGraph(self.atomg_props['bond_list'],
                                   self.atomg_props['metal1'],
                                   self.atomg_props['metal2'])

    def initialize_new_run(self):
        """

        :return:
        """
        # search for previous bimetallic result
        self.prev_results = db_inter.get_bimet_result(
            metals=self.metals,
            shape=self.shape,
            num_atoms=self.num_atoms,
            n_metal1=int(self.num_atoms - self.n_dope),
            lim=1)

        self.x_dope = self.n_dope / self.num_atoms
        self.formula = '%s%i_%s%i' % (self.metals[0],
                                      self.num_atoms - self.n_dope,
                                      self.metals[1],
                                      self.n_dope)

        self.build_pop()
        self.sort_pop()

        self.stats = []
        self.min_struct_ls = []
        self.update_stats()

        # track runtime
        self.runtime = 0

        # keep track of whether a sim has been run
        self.has_run = False

    def build_pop(self, random=False):
        """

        :return:
        """
        self.pop = []

        # if not random pop add max and min CN-fileed structures
        # also check to see if current min xyz path was given
        if not random:
            # min CN
            self.pop += [Chromo(self.atomg, n_dope=self.n_dope,
                                arr=fill_cn(self.atomg, self.n_dope,
                                            low_first=True, return_n=1)[0])]
            # max CN
            self.pop += [Chromo(self.atomg, n_dope=self.n_dope,
                                arr=fill_cn(self.atomg, self.n_dope,
                                            low_first=False, return_n=1)[0])]

            # add current min CE structure if path exists not monometallic
            if self.prev_results and self.n_dope not in [0, self.num_atoms]:
                self.pop += [Chromo(self.atomg, self.n_dope, arr=np.array(
                    [int(i) for i in self.prev_results.ordering]))]

        # create random structures for remaining popsize
        self.pop += [Chromo(self.atomg, n_dope=self.n_dope)
                     for i in range(self.popsize - len(self.pop))]

    def get_min(self):
        """

        :return:
        """
        self.sort_pop()
        return self.pop[0].ce

    def step(self, rand=False):
        """

        :param rand:
        :return:
        """
        if rand:
            self.build_pop(n_fill_cn=0)
        else:
            # kill off <kill_rate>% of pop
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
                    # add random chromosomes to fill remaining pop
                    self.pop.append(Chromo(self.pop[0].atomg,
                                           n_dope=self.n_dope))

            self.pop = self.pop[:self.popsize]

            for j in range(self.nmut):
                self.pop[random.randrange(1,
                                          self.popsize)].mutate(self.mute_num)

        self.sort_pop()
        self.update_stats()

    def run(self, nsteps=100, std_cut=0.0, rand=False):
        """
        Runs a GA simulation

        Kargs:
        nsteps (int): (default: 100)
        std_cut (float): (default: 0.0)
        rand (bool): (default: False)

        Returns: None

        Raises:
                ValueError: can only call run for first GA sim
        """
        if self.has_run:
            raise ValueError("Simulation has already run. "
                             "Please use continue method.")

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
            self.runtime += time.time() - start

            # run James' metropolis algorithm to search for
            # minimum struct near current min
            low_struct = self.pop[0]
            opt_order, opt_ce, en_hist = self.atomg.metropolis(low_struct.arr,
                                                               num_steps=5000,
                                                               swap_any=False)

            # if metropolis alg finds a new minimum, drop bottom one from pop
            if opt_ce < low_struct.ce:
                print('Found new min with Metropolis!')
                self.pop = [Chromo(self.atomg,
                                   n_dope=self.n_dope,
                                   arr=opt_order)] + self.pop[:-1]
                self.sort_pop()
                self.update_stats()

        self.stats = np.array(self.stats)

        self.has_run = True

    def continue_run(self, nsteps=100, std_cut=0.0, rand=False):
        """
        Used to continue GA sim from where it left off

        Kargs:
            NOTE: see self.run for args
        """
        # remake AtomGraph objects
        if not self.atomg:
            self.reload_atomg()

        self.has_run = False
        self.stats = list(self.stats)
        self.run(nsteps, std_cut, rand)
        self.continued += 1

    def sort_pop(self):
        """

        :return:
        """
        self.pop = sorted(self.pop,
                          key=lambda j: j.ce)

    def update_stats(self):
        """
        - Adds min, mean, and std dev of current generation to self.stats
        - Adds min structure Chromo objf of current generation

        Returns: None
        """
        s = np.array([i.ce for i in self.pop])
        self.stats.append([s[0],
                          s.mean(),
                          s.std()])
        self.min_struct_ls.append(Chromo(self.atomg, self.n_dope,
                                         arr=self.pop[0].arr.copy()))

    def is_new_min(self):
        """
        Returns True if GA sim found new minimum CE
        (compares to SQL DB)
        """
        if not self.prev_results:
            return True
        else:
            return True if self.get_min() < self.prev_results.CE else False

    def reload_atomg(self):
        """
            Used to reinitialize AtomGraph objects
            in self, and two lists of Chromos
        """

        self.atomg = self.__make_atomg__()
        for c in self.pop:
            c.atomg = self.__make_atomg__()
        for c2 in self.min_struct_ls:
            c2.atomg = self.__make_atomg__()

    def save(self, path):
        """
        Saves the Pop instance as a pickle

        Args:
        path (str): path to save pickle file
                    - can include filename
        """
        # Pop cannot be pickled with an AtomGraph instance
        self.atomg = None

        # remove AtomGraph from Chromos
        for c in self.pop:
            c.atomg = None
        for c2 in self.min_struct_ls:
            c2.atomg = None

        # if path doesn't include a filename, make one
        if not path.endswith('.pickle'):
            fname = '%s%s_GA_sim.pickle'
            path = os.path.join(path, fname)

        # pickle self
        with open(path, 'wb') as fidw:
            pickle.dump(self, fidw, protocol=pickle.HIGHEST_PROTOCOL)

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

        m1, m2 = self.metals

        res = ' Min: %.5f\nMean: %.3f\n STD: %.3f\n' % tuple(self.stats[-1, :])
        res += 'Mute: %.2f\nKill: %.2f\n' % (self.mute_rate, self.kill_rate)
        res += ' Pop: %i\n' % self.popsize
        res += 'nRun: %i\n' % (len(self.stats) - 1)
        res += 'Form: %s%i_%s%i\n' % (m1, self.num_atoms - self.n_dope,
                                      m2, self.n_dope)
        if self.has_run:
            res += 'Time: %.3e\n' % (self.runtime) + '\n\n\n'
        if disp:
            print(res)
        return res

    def plot_results(self, savepath=None):
        """
        Method to create a plot of GA simulation
        - plots average, std deviation, and minimum score
          of the population at each step

        Kargs:
        savepath (str): path and file name to save the figure
                        - if None, figure is not saved
                        (default: None)

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

        # format latex formula for plot title
        tex_form = re.sub('([A-Z][a-z])([0-9]+)_([A-Z][a-z])([0-9]+)',
                          '\\1_{\\2}\\3_{\\4}$',
                          self.formula)
        ax.set_title('$\\rm %s: %.3f eV' % (tex_form, low[-1]),
                     fontdict=dict(weight='normal'))
        # ax.set_title('Min CE: %.5f' % (self.get_min()))
        fig.tight_layout()

        # save figure if <savepath> was specified
        if savepath:
            fig.savefig(savepath)

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
    atom.info['CE'] = chrom.ce
    atom.set_tags(None)
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


def xyz_to_chrom(atomg, path):
    """
    Method used to create Chromo of *.xyz
    using AtomGraph and *.xyz path

    Args:
    atomg (atomgraph.AtomGraph): AtomGraph obj
    path (str): path to *.xyz file

    Returns:
        (Chromo): Chromo representing *.xyz structure
        (None): if path is not a file
    """
    if not os.path.isfile(path):
        return

    atom_obj = ase.io.read(path)
    metal1, metal2 = sorted(set(atom_obj.get_chemical_symbols()))

    # 1-0 ordering of structure
    arr = [1 if i.symbol == metal2 else 0 for i in atom_obj]

    c = Chromo(atomg, n_dope=sum(arr), arr=arr)
    return c


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
        scores[i] = c.ce
        if c.ce < min_ce:
            min_struct = Chromo(atomg, n_dope=n_dope, arr=c.arr.copy())
            min_ce = min_struct.ce
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


def build_pop_obj(metals, shape, num_shells, **kwargs):
    """
    Creates an initialized Pop for the specified shape,
    metals, and number of shells
    - e.g. 55-atom (3 shell) AgCu Icosahedron Pop
    - **kwargs gets passed directly into Pop.__init__

    Args:
    metals (str | iterator): list of two metals used in the bimetallic NP
    shape (str): shape of NP that is being studied
                 NOTE: currently supports
                        - icosahedron
                        - cuboctahedron
                        - fcc-cube
                        - elongated-trigonal-pyramic
    num_shells (int): number of shells used to generate atom size
                      e.g. icosahedron with 3 shells makes a 55-atom object
                      ( 1 in core + 12 in shell_1 + 42 in shell_2)

    **Kwargs:
        valid arguments to initialize Pop object
        - e.g. popsize=50, x_dope=0.5

    Returns:
        (Pop): initialized Pop object
    """
    nanop = structure_gen.build_structure_sql(shape, num_shells,
                                              build_bonds_list=True)

    p = Pop(nanop.get_atoms_obj_skel(), nanop.bonds_list,
            metals, shape, **kwargs)
    return p


def run_ga(metals, shape, save_data=True,
           batch_runinfo=None, shells=None,
           max_generations=None):
    """
    Submission function to run GAs of a given metal combination and
    shape, sweeping over different sizes (measured in number of shells)
    - capable of saving minimum structures as XYZs, logging GA stats, saving
      all run info as excel, and creating a 3D surface plot of results

    Args:
    metals (iterator): list of two metals used in the bimetallic NP
    shape (str): shape of NP that is being studied
                 NOTE: currently supports
                        - icosahedron
                        - cuboctahedron
                        - fcc-cube
                        - elongated-trigonal-pyramic

    Kargs:
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
    batch_runinfo (str): if str, add to log under
                         'Completed Run: <batch_runinfo>'
                         (default: None)
    shells (int || list): if int, only that shell size is simulated
                          elif list of ints, nshell_range = shells
                          (default: None)
    max_generations (int): if not None, use specified value as max generations
                           for each GA sim
                           (default: None)

    Returns: None
    """

    # track start of run
    start_time = dt.now()

    # clear previous plots and define desktop and data paths
    plt.close('all')
    desk = os.path.join(os.path.expanduser('~'), 'desktop')

    # always sort metals by alphabetical order for consistency
    if isinstance(metals, str):
        metal1, metal2 = sorted([metals[:2], metals[2:]])
    else:
        metal1, metal2 = sorted(metals)
    metal1, metal2 = metal1.title(), metal2.title()

    metals = (metal1, metal2)

    # number of shells range to sim ga for each shape
    shape2shell = {'icosahedron': [3, 14],
                   'fcc-cube': [2, 15],
                   'cuboctahedron': [2, 15],
                   'elongated-pentagonal-bipyramid': [3, 12]
                   }
    nshell_range = shape2shell[shape]

    if shells:
        if isinstance(shells, int):
            nshell_range = [shells, shells + 1]
        else:
            nshell_range = shells
    nstructs = len(range(*nshell_range))
    if not nstructs:
        print(nshell_range)
        return

    # print run info
    print('\n----------------RUN INFO----------------')
    print('             Metals: %s, %s' % (metal1, metal2))
    print('              Shape: %s' % shape)
    print('    Save GA Results: %s' % bool(save_data))
    print('        Shell Range: %s' % str(nshell_range))
    print('----------------------------------------')

    # GA PROPERTIES

    # number of generations for each GA sim
    if max_generations:
        max_runs = max_generations
    else:
        max_runs = 5000

    # population size
    popsize = 50

    # % of pop to kill each generation
    kill_rate = 0.2

    # % of remaining pop to mate (Chromo.cross)
    mate_rate = 0.8

    # % of pop to mutate (Chromo.mutate)
    mute_rate = 0.2

    # number of "atom switches" to make during mutation
    # (see <nps> karg in Chromo.mutate)
    mute_num = 1

    # keep track of total new minimum CE structs (based on saved structs)
    tot_new_structs = 0

    # count total structs
    tot_structs = 0

    for struct_i, nshells in enumerate(range(*nshell_range)):
        # build atom, adjacency list, and atomgraph
        nanop = structure_gen.build_structure_sql(shape, nshells,
                                                  build_bonds_list=True)
        num_atoms = len(nanop)

        diameter = nanop.get_diameter()

        ag = atomgraph.AtomGraph(nanop.bonds_list, metal1, metal2)

        # check to see if monometallic results exist
        # if not, calculate them
        mono1 = db_inter.get_bimet_result(metals=metals,
                                          shape=shape,
                                          num_atoms=num_atoms,
                                          n_metal1=num_atoms,
                                          lim=1)
        if not mono1:
            mono_ord = np.zeros(num_atoms)
            mono_ce1 = ag.getTotalCE(mono_ord)
            mono1 = db_inter.update_bimet_result(
                metals=metals,
                shape=shape,
                num_atoms=num_atoms,
                diameter=diameter,
                n_metal1=num_atoms,
                CE=mono_ce1,
                ordering=''.join(str(int(i)) for i in mono_ord),
                EE=0,
                nanop=nanop,
                allow_insert=True)

        mono2 = db_inter.get_bimet_result(metals=metals,
                                          shape=shape,
                                          num_atoms=num_atoms,
                                          n_metal1=0,
                                          lim=1)
        if not mono2:
            mono_ord = np.ones(num_atoms)
            mono_ce2 = ag.getTotalCE(mono_ord)
            mono2 = db_inter.update_bimet_result(
                metals=metals,
                shape=shape,
                num_atoms=num_atoms,
                diameter=diameter,
                n_metal1=0,
                CE=mono_ce2,
                ordering=''.join(str(int(i)) for i in mono_ord),
                EE=0,
                nanop=nanop,
                allow_insert=True)

        # USE THIS TO TEST EVERY CONCENTRATION
        if nanop.num_atoms < 150:
            n = np.arange(0, num_atoms + 1)
            x = n / float(num_atoms)
        else:
            # x = metal2 concentration [0, 1]
            x = np.linspace(0, 1, 11)
            n = (x * num_atoms).astype(int)

            # recalc concentration to match n
            x = n / float(num_atoms)

        # total structures checked ( - 2 to exclude monometallics)
        tot_structs += float(len(n) - 2)

        # INITIALIZE POP object
        # pop = Pop(nanop.get_atoms_obj_skel().copy(), nanop.bonds_list,
        #           metals, shape=shape, n_dope=n[0], popsize=popsize,
        #           kill_rate=kill_rate, mate_rate=mate_rate,
        #           mute_rate=mute_rate, mute_num=mute_num)

        starting_outp = '%s%s in %i atom %s' % (metal1, metal2,
                                                num_atoms, shape)
        print(starting_outp.center(CENTER))

        # track min structures for each size
        new_min_structs = 0

        # sweep over different compositions
        for i, dope in enumerate(n):
            # INITIALIZE POP object
            pop = Pop(nanop.get_atoms_obj_skel().copy(), nanop.bonds_list,
                      metals, shape=shape, n_dope=dope, popsize=popsize,
                      kill_rate=kill_rate, mate_rate=mate_rate,
                      mute_rate=mute_rate, mute_num=mute_num)
            # if i:
            #    pop.n_dope = dope
            #    pop.initialize_new_run()
            pop.run(max_runs)

            # if new minimum CE found and <save_data>
            # store result in DB
            if pop.is_new_min() and save_data:
                new_min_structs += 1
                tot_new_structs += 1
                n_metal1 = int(num_atoms - dope)
                ordering = ''.join([str(i) for i in pop.pop[0].arr])
                ce = pop.pop[0].ce

                # calculate excess energy
                ee = ag.getEE(pop.pop[0].arr)

                if abs(ee) < 1E-10:
                    ee = 0

                # update DB
                db_inter.update_bimet_result(
                    metals=metals,
                    shape=shape,
                    num_atoms=num_atoms,
                    diameter=diameter,
                    n_metal1=n_metal1,
                    CE=ce,
                    ordering=ordering,
                    EE=ee,
                    nanop=nanop)

        print(' ' * 100, end='\r')
        print('-' * CENTER)
        outp = 'Completed Size %i of %i (%i new mins)' % (struct_i + 1,
                                                          nstructs,
                                                          new_min_structs)
        print(outp.center(CENTER))
        print('-' * CENTER)

    # insert log results into DB
    if save_data:
        db_inter.insert_bimetallic_log(
            start_time=start_time,
            metal1=metal1,
            metal2=metal2,
            shape=shape,
            ga_generations=max_runs,
            shell_range='%i - %i' % tuple(nshell_range),
            new_min_structs=tot_new_structs,
            tot_structs=tot_structs,
            batch_run_num=batch_runinfo)


def check_db_values(update_db=False):
    """
    Checks CE values in database to ensure they match their ordering

    KArgs:
    update_db (bool): if True and mismatch is found, the database will be
                      updated to the correct CE and EE
                      (Default: False)
    """
    metal_opts = [('Ag', 'Au'),
                  ('Ag', 'Cu'),
                  ('Au', 'Cu')]

    shape_opts = ['icosahedron', 'cuboctahedron', 'fcc-cube',
                  'elongated-pentagonal-bipyramid']

    fails = []
    for shape in shape_opts:
        for shell in range(2, 15):
            for metals in metal_opts:
                for x_dope in np.linspace(0, 1, 11):
                    newp = build_pop_obj(metals, shape, shell,
                                         x_dope=x_dope, popsize=5)
                    res = newp.prev_results
                    if not res:
                        continue
                    ordering = np.array(list(map(int, res.ordering)))
                    actual_ce = newp.atomg.getTotalCE(ordering)
                    actual_ee = newp.atomg.getEE(ordering)
                    if abs(actual_ce - res.CE) > 1E-10:
                        fails.append([res.CE, actual_ce, res.EE, actual_ee])
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

                        print(res.build_chem_formula(False))

    fails = np.array(fails)
    print('%i issue%s found' % (len(fails), ['s', ''][bool(len(fails) == 1)]))
    return fails

if __name__ == '__main__':
    # run_ga(metals, shape, save_data=True, batch_runinfo='testing...',
    #       shells=6, max_generations=300)
    check_db_values()
