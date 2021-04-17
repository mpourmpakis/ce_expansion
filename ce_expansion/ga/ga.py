import functools
import itertools as it
import operator as op
import os
import pathlib
import pickle
import random
import re
import time
from datetime import datetime as dt

import ase.io
import matplotlib.pyplot as plt
import numpy as np
from ase.data.colors import jmol_colors

from ce_expansion.atomgraph import atomgraph
from ce_expansion.ga import structure_gen
from ce_expansion.npdb import db_inter

# random.seed(9876)

# center output strings around <CENTER> characters
CENTER = 50


class Chromo(object):
    def __init__(self, atomg, n_metal2=0, ordering=None, x_metal2=None):
        """
        Chromosome object for GA simulations
        Represents a single structure with a given chemical ordering (arr)

        Args:
        - atomg:

        Kargs:
        - n_metal2 (int): number of metal2 (atom depicted by 1 in ordering)
                          (Default: 0)
        - ordering (np.ndarray || list): binary array representing positions in
                                           nanoparticle and whether they're
                                         occupied by metal1 (0) or metal2 (1)
                                         (Default: None)
        - x_metal2 (float): concentration of metal2 - not required, but will be
                            used over n_metal2
                            (Default: None)

        Raises:
                ValueError: n_metal2 is greater than total atoms
        """
        self.atomg = atomg
        self.num_atoms = atomg.num_atoms
        if x_metal2 is not None:
            assert 0 <= x_metal2 <= 1, "x_metal2: [0, 1]"
            self.n_metal2 = np.array([self.num_atoms * x_metal2]
                                     ).astype(int)[0]
        else:
            self.n_metal2 = n_metal2

        self.x_metal2 = self.n_metal2 / self.num_atoms

        if self.n_metal2 > self.num_atoms:
            raise ValueError("Can't dope more atoms than there are atoms...")

        # if an array is given, use it - else random generate ordering
        if isinstance(ordering, np.ndarray) or isinstance(ordering, list):
            self.ordering = np.array(ordering)
            self.n_metal2 = self.ordering.sum()
        else:
            self.ordering = np.zeros(self.num_atoms).astype(int)
            self.ordering[:self.n_metal2] = 1
            np.random.shuffle(self.ordering)

        # calculate initial CE
        self.calc_score()

    def __len__(self):
        return self.num_atoms

    def __getitem__(self, i):
        return self.ordering[i]

    def copy(self):
        """
        Creates a copy of the Chromo object

        Returns:
        - (Chromo): exact copy of the Chromo object
        """
        return Chromo(self.atomg, self.n_metal2, ordering=self.ordering.copy())

    def mutate(self, n_swaps=1):
        """
        Algorithm to randomly switch a '1' & a '0' within ordering array
        - about O(n) scaling

        Args:
        - n_swaps (int): number of swaps to make
                         (Default: 1)

        Raises:
        - ValueError: if not AtomGraph, Chromo can not and
                      should not be mutated
        """
        if not self.atomg:
            raise ValueError("Mutating Chromo should only be done through"
                             "Pop simulations")

        if not self.n_metal2 or self.n_metal2 == self.num_atoms:
            print('Warning: attempting to mutate, but system is monometallic')
            return

        # keep track of indices used so there are no repeats
        used = [None, None] * n_swaps

        # complete <nps> swaps
        # track what swap we're on
        swap = 0
        for s in range(n_swaps):
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
                    if self.ordering[i] == 0 and not changed[0]:
                        self.ordering[i] = 1
                        changed[0] = True
                        used[swap] = i
                        swap += 1

                    # 2b) if the index leads to a '1' and no '1'
                    # has been changed, switch it to a '0'
                    elif self.ordering[i] == 1 and not changed[1]:
                        self.ordering[i] = 0
                        changed[1] = True
                        used[swap] = i
                        swap += 1

        # update CE 'score' of Chrom
        self.calc_score()

    def mate(self, chromo2):
        """
        Pairwise crossover algorithm to mix two parent chromosomes into
        two new child chromosomes, taking traits from each parent
        - conserves doping concentration
        - about O(N^2) scaling

        Args:
        - chromo2 (Chromo): second parent Chromo obj

        Returns:
        - (list): two children Chromo objs with new ordering <arr>s
        """
        child1 = self.ordering.copy()
        child2 = chromo2.ordering.copy()

        assert (child1.sum() == child2.sum() ==
                self.n_metal2 == chromo2.n_metal2)

        # if only one '1' or one '0' then mating is not possible
        if self.n_metal2 in [1, self.num_atoms - 1]:
            return [self.copy(), chromo2.copy()]

        # indices where a '1' is found in child1
        ones = np.where(child1 == 1)[0]

        # indices where child1 and child2 do not match
        diff = np.where(child1 != child2)[0]

        # must have at least two different sites
        if len(diff) < 2:
            return [self.copy(), chromo2.copy()]

        # swap ~half of the different sites
        # s1: number of '1's from child1 switched to child2
        # s2: number of '1's from child2 switched to child1
        s1 = s2 = int(len(diff)) // 2

        # even number of swaps must be made to conserve composition
        if s1 % 2:
            s1 = s2 = s1 - 1

        # number of swaps must be at least 2
        if s1 < 2:
            s1 = s2 = 2

        # shuffle diff for random selection of traits to crossover
        np.random.shuffle(diff)
        for i in diff:
            # swap '1' from child1 to child2
            if i in ones:
                if s1:
                    child1[i] = 0
                    child2[i] = 1
                    s1 -= 1
            # swap '1' from child2 to child1
            elif s2:
                child1[i] = 1
                child2[i] = 0
                s2 -= 1
            if s1 == s2 == 0:
                break

        assert (child1.sum() == child2.sum() ==
                self.n_metal2 == chromo2.n_metal2)

        children = [Chromo(self.atomg, n_metal2=self.n_metal2,
                           ordering=child1.copy()),
                    Chromo(self.atomg, n_metal2=self.n_metal2,
                           ordering=child2.copy())]
        return children

    def calc_score(self):
        """
        Returns CE of structure based on Bond-Centric Model
        - Yan, Z. et al., Nano Lett. 2018, 18 (4), 2696-2704.
        """
        self.ce = self.atomg.getTotalCE(self.ordering)


class Pop(object):
    def __init__(self, atom, bond_list, metals, shape, n_metal2=1,
                 popsize=50, mute_pct=0.8, n_mute_atomswaps=None, spike=False,
                 x_metal2=None, random=False, num_shells=None, atomg=None,
                 e=1, save_every=100, use_metropolis=True):
        """
        Bimetallic nanoparticle genetic algorithm population
        - initialize a population of nanoparticles
        - call self.run to run a GA simulation
        - if random=True, self.run will conduct a random search

        Args:
        - atom (ase.Atoms): nanoparticle atoms object
        - bond_list (list): list of atom indices connected by a bond
        - metals (tuple): metal types in nanoparticle
        - shape (str): nanoparticle shape

        KArgs:
        - n_metal2 (int): number of metal2 in nanoparticle
                          (Default: 1)
        - popsize (int): size of population
                         (Default: 50)
        - mute_pct (float): percentage of population to mutate each generation
                            (Default: 0.8)
        - n_mute_atomswaps (int): number of atom swaps to make
                                  during a mutation
                                  (Default: None: half min(n_metal1, n_metal2))
        - spike (bool): if True, following structures are added to generation 0
                        - if same structure and composition found in DB,
                          it is added to population
                        - minCN-filled or maxCN-filled nanoparticle
                          (depending on which is more fit) also added
                          to population - structure created using
                          fill_cn function
        - x_metal2 (float): concentration of metal2 - not required, but will be
                            used over n_metal2
                            (Default: None)
        - random (bool): if True, self.run does a random search and
                         does not perform a GA optimization
        - num_shells (int): number of shells in nanoparticle
                            - used when saving nanoparticle to database
        - e (float): exploration - exploitation factor used to bias parent
                                   selection probabilities
                                   (Default: 1 = No effect)
        - save_every (int): choose how often pop CEs are stored in all_data
                            (Default: every 100 generations)
        - use_metropolis (bool): use metropolis algorithm at end of GA sim
                                 (Default: True)
        """
        # atoms object
        self.atom = atom

        # if num_shells not given, set it to -1
        if num_shells:
            self.num_shells = num_shells
        elif 'num_shells' in self.atom.info:
            self.num_shells = self.atom.info['num_shells']
        else:
            self.num_shells = -1

        # np shape and number of atoms
        self.shape = shape
        self.num_atoms = len(atom)

        # bond_list
        self.bond_list = bond_list

        # metal1 is always selected by alphabetical order
        self.metals = sorted(metals)
        self.metal1 = self.metals[0]
        self.metal2 = self.metals[1]

        # create AtomGraph instance
        self.atomg = atomg if atomg is not None else self.__make_atomg__()

        # determine number of metal2
        # use x_metal2 if passed in
        if x_metal2 is not None:
            self.n_metal2 = np.array([self.num_atoms * x_metal2]
                                     ).astype(int)[0]
        else:
            self.n_metal2 = n_metal2

        self.n_metal2 = int(self.n_metal2)

        # calculate exact x_metal2
        self.x_metal2 = self.n_metal2 / self.num_atoms

        # set n_metal1
        self.n_metal1 = int(self.num_atoms - self.n_metal2)

        self.popsize = popsize

        # determine number of chromos that will be mutated each generation
        # popsize - 1 is max number of chromo objects that can be mutated
        self.n_mute = min(int(popsize * mute_pct), popsize - 1)
        self.mute_pct = self.n_mute / self.popsize

        # determine number of atom swaps to apply to a chromo
        # when it's getting mutated
        self.n_mute_atomswaps = n_mute_atomswaps

        # if no specific atomswap number given, choose half of the
        # minimum atom type (min(n_mute_atomswaps) is 1)
        if self.n_mute_atomswaps is None:
            self.n_mute_atomswaps = max(min(self.n_metal1, self.n_metal2) // 50,
                                        1)

        # spike ga with previous run and structures from fill_cn
        self.spike = spike

        # keep track of how many times the sim has been continued
        self.continued = 0

        # previous GA sim results (npdb.datatables.BimetallicResults)
        self.prev_results = None

        self.random = random

        # exploration-exploitation factor
        self.e = e

        # store CEs of pops during simulation
        self.all_data = {}

        # how often pop data should be saved to all_data
        self.save_every = save_every

        # track whether metropolis should be used at end of GA sim
        self.use_metropolis = use_metropolis

        # population - list of chromosomes
        self.pop = []
        self.initialize_new_run()

    def __len__(self):
        return self.popsize

    def __getitem__(self, i):
        return self.pop[i]

    def __make_atomg__(self):
        """
        Initializes AtomGraph object (used to calc fitness)

        Returns:
        - (AtomGraph)
        """
        return atomgraph.AtomGraph(self.bond_list,
                                   self.metal1,
                                   self.metal2)

    def reload_atomg(self):
        """
        Used to reinitialize AtomGraph objects
        in self and two lists of Chromos
        """

        self.atomg = self.__make_atomg__()
        for c in self.pop:
            c.atomg = self.__make_atomg__()
        # for c2 in self.min_struct_ls:
        #     c2.atomg = self.__make_atomg__()

    def initialize_new_run(self):
        """
        Sets up Pop for a new GA simulation (generation 0)
        - can simulate different composition by changing n_metal2
          then running this method
        """
        # search for previous bimetallic result
        self.prev_results = db_inter.get_bimet_result(
            metals=self.metals,
            shape=self.shape,
            num_atoms=self.num_atoms,
            n_metal1=int(self.num_atoms - self.n_metal2),
            lim=1)

        self.x_metal2 = self.n_metal2 / self.num_atoms
        self.formula = '%s%i-%s%i' % (self.metal1,
                                      self.num_atoms - self.n_metal2,
                                      self.metal2,
                                      self.n_metal2)

        self.initialize_pop()
        self.sort_pop()
        self.orig_min = self[0].ce

        self.stats = []
        # self.min_struct_ls = []
        self.update_stats()

        # track runtime
        self.runtime = 0

        # track generations
        self.generation = 0

        # minimum CE of last generation
        self.last_min = 100

        # keep track of whether a sim has been run
        self.has_run = False

    def initialize_pop(self):
        """
        Initialize population of nanoparticle Chromosome objects
        - if self.random, population filled completely with random structures
        - if self.spike, following structures are added
            - if same structure and composition found in DB, it is added to
              population
            - minCN-filled or maxCN-filled nanoparticle (depending on which is
              more fit) also added to population - structure created using
              fill_cn function
        """
        # if random and stepping to next generation, keep current best
        if self.random and self.pop:
            self.pop = [self[0]]

        # if not random pop add max or min CN-fileed structures
        # also check to see if current min xyz path was given
        elif not self.random and self.spike:
            # min CN
            mincn = Chromo(self.atomg, n_metal2=self.n_metal2,
                           ordering=fill_cn(self.atomg, self.n_metal2,
                                            low_first=True, return_n=1)[0])
            # max CN
            maxcn = Chromo(self.atomg, n_metal2=self.n_metal2,
                           ordering=fill_cn(self.atomg, self.n_metal2,
                                            low_first=False, return_n=1)[0])

            self.pop.append(min([mincn, maxcn], key=lambda i: i.ce))

            # add current min CE structure if path exists not monometallic
            if self.prev_results and self.n_metal2 not in [0, self.num_atoms]:
                self.pop += [Chromo(
                    self.atomg,
                    self.n_metal2,
                    ordering=np.array(
                        [int(i) for i
                         in self.prev_results.ordering]))]

        # create random structures for remaining popsize
        self.pop += [Chromo(self.atomg, n_metal2=self.n_metal2)
                     for i in range(self.popsize - len(self.pop))]

    def get_min(self):
        """
        Returns nanoparticle with lowest CE of current generation
        """
        self.sort_pop()
        return self[0].ce

    def roulette_mate(self):
        """
        Roulette Wheel selection algorithm that chooses mates
        based on their fitness
        - probability of chromosome <i> being selected:
          P_i = (fitness_i / sum of all fitnesses)
        - fitter chromosomes more likely to be selected, but not guaranteed
          to help mitigate lack of diversity in population
        """
        ces = np.array([abs(p.ce) for p in self])
        fitness = (ces - ces.min())**self.e
        totfit = fitness.sum()
        probs = np.zeros(self.popsize)
        for i in range(self.popsize):
            probs[i:] += abs(fitness[i] / totfit)

        mates = []
        for n in range(self.popsize // 2):
            m = [self[np.where(probs > random.random())[0][0]]
                 for z in range(2)]
            mates += m[0].mate(m[1])
        # keep the previous minimum
        self.pop = [self[0]] + mates
        self.pop += [Chromo(self[0].atomg, n_metal2=self.n_metal2)
                     for z in range(self.popsize - len(self.pop))]
        self.pop = self[:self.popsize]

    def step(self):
        """
        Wrapper method that takes GA to next generation
        - mates the population
        - mutates the population
        - calculates statistics of new population
        - resorts population and increments self.generation
        """
        if self.random:
            self.initialize_pop()
        else:
            # MATE
            self.roulette_mate()

            # MUTATE - does not mutate most fit nanoparticle
            for j in range(self.n_mute):
                self[random.randrange(1, self.popsize)
                ].mutate(self.n_mute_atomswaps)

        self.sort_pop()
        self.update_stats()
        self.print_status()

        # increment generation
        self.generation += 1

    def print_status(self, end='\r'):
        """
        Prints info on current generation of GA

        KArgs:
        - end (str): what to end the print statement with
                     - allows generations to overwrite on same line
                       during simulation
                     (Default: '\r')
        """
        # format of string to be written to console during sim
        update_str = ' %s %s...Min: %.5f eV/atom   %05i'
        val = update_str % (self.formula,
                            self.shape.upper()[:3],
                            self.stats[-1][0],
                            self.generation)
        assert self.stats[-1][0] <= self.last_min
        self.last_min = self.stats[-1][0]
        print(val.center(CENTER), end=end)

    def run(self, max_gens=-1, max_nochange=50, min_gens=-1):
        """
        Runs a GA simulation

        Kargs:
        - max_gens (int): maximum generations the GA will run
                          -1: the GA only stops based on <max_nochange>
                          (Default: -1)
        - max_nochange (int): specifies max number of generations GA will run
                              without finding a better NP
                              -1: GA will run to <max_gens>
                              (default: 50)
        - min_gens (int): minimum generations that the GA runs before checking
                          the max_nochange criteria
                          -1: no minimum
                          (Default: -1)

        Raises:
        - TypeError: can only call run for first GA sim
        - ValueError: max_gens and max_nochange cannot both equal -1
        """
        if self.has_run:
            raise TypeError("Simulation has already run. "
                            "Please use continue method.")
        elif max_gens == max_nochange == -1:
            raise ValueError("max_gens and max_nochange cannot both be "
                             "turned off (equal: -1)")

        # atomg might be None from loading a pickled Pop object
        if self.atomg is None:
            self.reload_atomg()

        self.max_gens = max_gens

        # GA will not continue if <max_nochange> generations are
        # taken without a change in minimum CE
        nochange = 0

        # no GA required for monometallic systems
        if self.n_metal2 not in [0, self.atomg.num_atoms]:
            """ALL DATA ADDED"""
            self.all_data[self.generation] = [i.ce for i in self]
            start = time.time()

            while self.generation != self.max_gens:
                # step to next generation
                self.step()
                if self.generation % self.save_every == 0:
                    self.all_data[self.generation] = [i.ce for i in self]

                # track if there was a change (if min_gens has been passed)
                if (max_nochange and min_gens and
                        self.generation > min_gens):
                    if self.stats[-1][0] == self.stats[-2][0]:
                        nochange += 1
                    else:
                        nochange = 0

                    # if no change has been made after <maxnochange>, stop GA
                    if nochange == max_nochange:
                        break

            """Get info for last generation"""
            self.all_data[self.generation] = [i.ce for i in self]

            # print status of final generation
            self.print_status(end='\n')

            # set max_gens to actual generations simulated
            self.max_gens = self.generation

            # capture runtime in seconds
            self.runtime += time.time() - start

            # run James' metropolis algorithm function to search for
            # minimum struct near current min
            if not self.random and self.use_metropolis:
                best = self[0]
                best_ordering = best.ordering.copy()
                opt_order, opt_ce, en_hist = self.atomg.metropolis(
                    best_ordering,
                    num_steps=5000,
                    swap_any=False)

                # if metropolis alg finds a new minimum,
                # drop bottom one from pop
                if opt_ce < best.ce:
                    print('Found new min with Metropolis!')
                    self.pop = [Chromo(self.atomg,
                                       n_metal2=self.n_metal2,
                                       ordering=opt_order)] + self[:-1]
                    assert self.pop == sorted(self.pop, key=lambda i: i.ce)
                    self.update_stats()
            else:
                print('NOT running metropolis')

        # convert stats to an array
        self.stats = np.array(self.stats)

        self.has_run = True

    def continue_run(self, max_gens=-1, max_nochange=50, min_gens=-1):
        """
        Used to continue GA sim from where it left off

        Kargs:
        max_gens (int): maximum generations the GA will run
                        -1: the GA only stops based on <max_nochange>
                        (Default: -1)
        max_nochange (int): specifies max number of generations GA will run
                            without finding a better NP
                            -1: GA will run to <max_gens>
                            (default: 50)
        - min_gens (int): minimum generations that the GA runs before checking
                          the max_nochange criteria
                          -1: no minimum
                          (Default: -1)
        """
        # remake AtomGraph objects
        if not self.atomg:
            self.reload_atomg()

        self.has_run = False
        self.stats = list(self.stats)
        self.run(max_gens=max_gens, max_nochange=max_nochange,
                 min_gens=min_gens)
        self.continued += 1

    def sort_pop(self):
        """
        Sorts population based on cohesive energy
        - lowest cohesive energy = most fit = first in list
        """
        self.pop = sorted(self.pop,
                          key=lambda j: j.ce)

    def update_stats(self):
        """
        - Adds statistics of current generation to self.stats
        - Adds best Chromo of current generation to self.min_struct_ls
        """
        self.sort_pop()
        s = np.array([i.ce for i in self.pop])
        self.stats.append([s.min(),  # minimum CE
                           s.mean(),  # mean CE
                           s.std()])  # STD CE
        # self.min_struct_ls.append(Chromo(self.atomg, self.n_metal2,
        #                                  ordering=self[0].ordering.copy()))

    def is_new_min(self, check_db=True):
        """
        Returns True if GA sim found new minimum CE
        (compares to SQL DB if <check_db>)

        KArgs:
        check_db (bool): if True, only compares to database min
                         else it'll compare to generation 0
        """
        cur_min = self.get_min()
        if check_db:
            if not self.prev_results:
                return True
            else:
                return cur_min < self.prev_results.CE
        else:
            return cur_min < self.orig_min

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
        # for c2 in self.min_struct_ls:
        #     c2.atomg = None

        # if path doesn't include a filename, make one
        if not path.endswith('.pickle'):
            fname = '%s_%s_GA_sim.pickle' % (self.formula, self.shape[:3])
            path = os.path.join(path, fname)

        # pickle self
        with open(path, 'wb') as fidw:
            pickle.dump(self, fidw, protocol=pickle.HIGHEST_PROTOCOL)

        # reload AtomGraph after saving pickle
        self.reload_atomg()

    def summ_results(self, disp=False):
        """
        Creates string listing GA simulation stats and results info
        str includes:
            - minimum, mean, and std dev. of CEs for last population
            - mute and mate (crossover) rates of GA
            - total steps taken by GA
            - structure formula

        Kargs:
        - disp (bool): if True, results string is written to console
                       (default: False)

        Returns:
        - (str): result string

        Raises:
        - Exception: if sim has not been run
                     i.e. self.has_run == False
        """
        if not self.has_run:
            raise Exception('No simulation has been run')

        res = '  Min: %.5f\n' % self.stats[-1, 0]
        res += ' Mean: %.3f\n' % self.stats[-1, 1]
        res += '  STD: %.3f\n' % self.stats[-1, 2]
        res += ' Mute: %.2f\n' % self.mute_pct
        res += '  Pop: %i\n' % self.popsize
        res += 'nGens: %i\n' % self.max_gens
        res += ' Form: %s\n' % self.formula
        if self.has_run:
            res += ' Time: %.3e\n' % self.runtime
        if disp:
            print(res)
        return res

    def plot_results(self, savepath=None, ax=None):
        """
        Method to create a plot of GA simulation
        - plots average, std deviation, and minimum score
          of the population at each step

        Kargs:
        - savepath (str): path and file name to save the figure
                          - if None, figure is not saved
                          (default: None)
        - ax (matplotlib.axis): if given, results will be plotted on axis
                                (Default: None)

        Returns:
        - (matplotlib.figure.Figure),
        - (matplotlib.axes._subplots.AxesSubplot): fig and ax objs
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(9, 9))
            color = 'navy'
            fillcolor = 'lightblue'
            secondary = False
        else:
            fig = ax.figure
            color = 'red'
            fillcolor = 'pink'
            secondary = True

        # number of steps GA took
        steps = range(len(self.stats))

        # minimum, average, and std deviation scores of population at each step
        low = self.stats[:, 0]
        mean = self.stats[:, 1]
        std = self.stats[:, 2]

        # plot minimum CE as a dotted line
        ax.plot(low, ':', color=color, label='_nolabel_')
        # light blue fill of one std deviation
        ax.fill_between(range(len(self.stats)), mean + std, mean - std,
                        color=fillcolor, label='_nolabel_')

        # plot mean as a solid line and minimum as a dotted line
        ax.plot(mean, color=color, label='_nolabel_')
        if not secondary:
            # create legend
            ax.plot([], [], ':', color='gray', label='MIN')
            ax.plot([], [], color='k', label='MEAN')
            ax.fill_between([], 0, 0, color='k', label='STD')
            ax.legend(ncol=3, loc='upper center')
            ax.set_ylabel('Cohesive Energy (eV / atom)')
            ax.set_xlabel('Generation')

            # format latex formula for plot title
            tex_form = re.sub('([A-Z][a-z])([0-9]+)-([A-Z][a-z])([0-9]+)',
                              '\\1_{\\2}\\3_{\\4}$',
                              self.formula)
            ax.set_title('$\\rm %s %s\n%.3f eV'
                         % (tex_form, self.shape.title(), low[-1]),
                         fontdict=dict(weight='normal'))
            # ax.set_title('Min CE: %.5f' % (self.get_min()))
            fig.tight_layout()

        # save figure if <savepath> was specified
        if savepath:
            fig.savefig(savepath)

        return fig, ax

    def save_to_db(self):
        """Save minimum CE nanoparticle to database
           - connection is made through npdb.db_inter
           - only updates database if new min is found
        """
        if not self.is_new_min():
            print('Database contains NP with as good/better fitness.')
            print('No action taken.')
            return

        # get best chrom
        self.sort_pop()
        best = self[0]

        # try to find nanoparticle in DB
        nanop = db_inter.get_nanoparticle(
            self.shape,
            num_atoms=self.num_atoms,
            lim=1)

        # if not found, add a new nanoparticle to DB
        if not nanop:
            nanop = db_inter.insert_nanoparticle(self.atom, self.shape,
                                                 self.num_shells)

        ce = best.ce
        assert ce == best.atomg.getTotalCE(best.ordering)

        ee = best.atomg.getEE(best.ordering)

        # update DB
        db_inter.update_bimet_result(
            metals=self.metals,
            shape=self.shape,
            num_atoms=self.num_atoms,
            diameter=nanop.get_diameter(),
            n_metal1=self.n_metal1,
            CE=ce,
            ordering=''.join(map(str, best.ordering)),
            EE=ee,
            nanop=nanop,
            allow_insert=False)
        print('New min NP added to database.')


def load_pop(path):
    """
    Load a pickled Pop object

    Args:
    path (str): path to Pop object (*.pickle file)

    Returns:
    (Pop): pop object
    """
    with open(path, 'rb') as fid:
        unpickler = pickle.Unpickler(fid)
        pop = unpickler.load()

    # reload AtomGraph object
    pop.reload_atomg()
    return pop


def make_file(atom, chrom, path, verbose=False, filetype="xyz"):
    """
    Creates an XYZ file given Atoms obj skeleton and
    GA Chrom obj for metals and ordering

    Args:
    - atom (ase.Atoms): Atoms obj skeleton
    - chrom (Chrom): Chrom obj from GA containing ordering and metals
    - path (str): path to save file
    - filetype (str): Type of file to save. Default XYZ

    Kargs:
    - verbose (bool): if True, print save path on success

    Returns: None
    """
    atom = atom.copy()
    metal1, metal2 = chrom.atomg.symbols
    atom.info['CE'] = chrom.ce
    atom.info['EE'] = chrom.atomg.getEE(chrom.ordering)
    atom.set_tags(None)
    for i, dope in enumerate(chrom.ordering):
        atom[i].symbol = metal2 if dope else metal1
    print(atom)
    print(chrom.ordering)

    # create file name if not included in path
    if not path.endswith(filetype):
        n_metal2 = sum(chrom.ordering)
        n_metal1 = len(atom) - n_metal2
        fname = '%s%i_%s%i.%s' % (metal1, n_metal1,
                                  metal2, n_metal2,
                                  filetype)
        path = os.path.join(path, fname)

    # save file to path
    ase.io.write(path, atom)
    if verbose:
        print('Saved as %s' % path)
    return atom


def xyz_to_chrom(atomg, path):
    """
    Method used to create Chromo of *.xyz
    using AtomGraph and *.xyz path

    Args:
    - atomg (atomgraph.AtomGraph): AtomGraph obj
    - path (str): path to *.xyz file

    Returns:
    - (Chromo): Chromo representing *.xyz structure
    - (None): if path is not a file
    """
    if not os.path.isfile(path):
        return

    atom_obj = ase.io.read(path)
    metal1, metal2 = sorted(set(atom_obj.get_chemical_symbols()))

    # 1-0 ordering of structure
    ordering = np.array([int(i.symbol == metal2) for i in atom_obj])

    return Chromo(atomg, n_metal2=sum(arr), ordering=ordering)


def gen_random(atomg, n_metal2, n=500):
    """
    Generates random structures (constrained by size, shape, and concentration)
    and returns minimum structure and stats on CEs

    Args:
    - atomg (atomgraph.AtomGraph): AtomGraph object
    - n_metal2 (int): number of metal2 in nanoparticle
    - n (int): sample size

    Returns:
    - (Chrom), (np.ndarray): Chrom object of minimum structure found
                             1D array of all CEs calculated in sample
    """
    if n_metal2 in [0, atomg.num_atoms]:
        n = 1
    scores = np.zeros(n)

    # keep track of minimum structure and minimum CE
    min_struct = None
    min_ce = 10
    for i in range(n):
        c = Chromo(atomg, n_metal2=n_metal2)
        scores[i] = c.ce
        if c.ce < min_ce:
            min_struct = Chromo(atomg, n_metal2=n_metal2,
                                ordering=c.ordering.copy())
            min_ce = min_struct.ce
    return min_struct, scores


def ncr(n, r):
    """
    N choose r function (combinatorics)

    Args:
    - n (int): from 'n' choices
    - r (int): choose r without replacement

    Returns:
    - (int): total combinations
    """
    r = min(r, n - r)
    numer = functools.reduce(op.mul, range(n, n - r, -1), 1)
    denom = functools.reduce(op.mul, range(1, r + 1), 1)
    return numer // denom


def fill_cn(atomg, n_metal2, max_search=50, low_first=True, return_n=None,
            verbose=False):
    """
    Algorithm to fill the lowest (or highest) coordination sites with 'metal2'

    Args:
    - atomg (atomgraph.AtomGraph): AtomGraph object
    - n_metal2 (int): number of dopants

    Kargs:
    - max_search (int): if there are a number of possible structures with
                        partially-filled sites, the function will search
                        max_search options and return lowest CE structure
                        (Default: 50)
    - low_first (bool): if True, fills low CNs, else fills high CNs
                        (Default: True)
    - return_n (bool): if > 0, function will return a list of possible
                       structures
                       (Default: None)
    - verbose (bool): if True, function will print info to console
                      (Default: False)

    Returns:
    if return_n > 0:
    - (list): list of chemical ordering np.ndarrays
    else:
    - (np.ndarray), (float): chemical ordering np.ndarray with its
                             calculated CE

    Raises:
    - ValueError: not enough options to produce <return_n> sample size
    """
    # handle monometallic cases efficiently
    if n_metal2 in [0, atomg.num_atoms]:
        # all '0's if no metal2, else all '1' since all metal2
        struct_min = [np.zeros, np.ones][bool(n_metal2)](atomg.num_atoms)
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
            if len(spots) == n_metal2:
                struct_min[spots] = 1
                checkall = True
                break
            elif len(spots) > n_metal2:
                low = 0
                low_struct = None

                # check to see how many combinations exist
                options = ncr(len(spots), n_metal2)

                # return sample of 'return_n' options
                if return_n:
                    if return_n > options:
                        raise ValueError('not enough options to '
                                         'produce desired sample size')
                    sample = []
                    while len(sample) < return_n:
                        base = struct_min.copy()
                        pick = random.sample(list(spots), n_metal2)
                        base[pick] = 1
                        sample.append(base)
                    return sample

                # if n combs < max_search, check them all
                if options <= max_search:
                    if verbose:
                        print('Checking all options')
                    searchopts = it.combinations(spots, n_metal2)
                    checkall = True
                else:
                    if verbose:
                        print("Checking {0:.2%}".format(max_search / options))
                    searchopts = range(max_search)
                    checkall = False

                # stop looking after 'max_search' counts
                for c in searchopts:
                    base = struct_min.copy()
                    if checkall:
                        pick = list(c)
                    else:
                        pick = random.sample(list(spots), n_metal2)
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
                n_metal2 -= len(spots)
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
    - metals (str | iterator): list of two metals used in the bimetallic NP
    - shape (str): shape of NP
                   NOTE: currently supported methods (found in NPBuilder)
                   - cuboctahedron
                   - elongated-trigonal-pyramid
                   - fcc-cube
                   - icosahedron
    - num_shells (int): number of shells used to generate atom size
                        e.g. icosahedron with 3 shells makes a 55-atom object
                        (1 in core (shell_0) + 12 in shell_1 + 42 in shell_2)

    **Kwargs:
    - valid arguments to initialize Pop object
      - e.g. popsize=50, x_metal2=0.5

    Returns:
    - (Pop): Pop instance
    """
    assert num_shells > 0, "NP must have at least one shell"
    nanop = structure_gen.build_structure_sql(shape, num_shells)

    p = Pop(nanop.atoms_obj, nanop.bonds_list,
            metals, shape, num_shells=num_shells, **kwargs)
    return p


def run_ga(metals, shape, save_data=True,
           batch_runinfo=None, shells=None,
           max_generations=5000,
           min_generations=-1,
           max_nochange=2000,
           add_coreshell=True,
           **kwargs):
    """
    Submission function to run GAs of a given metal combination and
    shape, sweeping over different sizes (measured in number of shells)
    - capable of saving minimum structures as XYZs, logging GA stats, saving
      all run info as excel, and creating a 3D surface plot of results

    Args:
    - metals (iterator): list of two metals used in the bimetallic NP
    - shape (str): shape of NP that is being studied
                   NOTE: currently supports
                          - icosahedron
                          - cuboctahedron
                          - fcc-cube
                          - elongated-trigonal-pyramic

    Kargs:
    - plotit (bool): if true, a 3D surface plot is made of GA sims
                     dope concentration vs. size vs. excess energy
                     (Default: False)
    - save_data (bool): - if true, GA sim data is saved to BimetallicResults
                          table in database
                        (Default: True)
    - batch_runinfo (str): if str, add to BimetallicLog entry
                           (Default: None)
    - shells (int || list): if int, only that shell size is simulated
                            elif list of ints, nshell_range = shells
                            (Default: None)
    - max_generations (int): if not -1, use specified value as max
                             generations for each GA sim
                             (Default: 5000)
    - min_generations (int): if not -1, use specified value as min
                             generations that run before max_nochange
                             criteria is applied
                             (Default: -1)
    - max_nochange (int): maximum generations GA will go without a change in
                          minimum CE
                          (Default: 2000)
    - add_coreshell (bool): if True, core shell structures will be included in
                            GA simulations
                            (Default: True)

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
    default_shell_range = [1, 11]
    shape2shell = {'cuboctahedron': default_shell_range,
                   'elongated-pentagonal-bipyramid': default_shell_range,
                   'fcc-cube': default_shell_range,
                   'icosahedron': default_shell_range
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
    print('')
    print('RUN INFO'.center(CENTER, '-'))
    print('             Metals: %s, %s' % (metal1, metal2))
    print('              Shape: %s' % shape)
    print('    Save GA Results: %s' % bool(save_data))
    print('        Shell Range: %s' % str(nshell_range))
    print('-' * CENTER)

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
        if nanop.num_atoms < 366:
            n = np.arange(0, num_atoms + 1)
            x = n / float(num_atoms)
        else:
            # x = metal2 concentration [0, 1]
            x = np.linspace(0, 1, 11)
            n = (x * num_atoms).astype(int)

            # recalc concentration to match n
            x = n / float(num_atoms)

        # add core-shell structures list of comps to run
        if add_coreshell:
            srfatoms = db_inter.build_atoms_in_shell_dict(shape, nshells)
            nsrf = len(srfatoms)
            ncore = num_atoms - nsrf
            n = np.unique(n.tolist() + [ncore, nsrf])
            x = n / float(num_atoms)

        # total structures checked ( - 2 to exclude monometallics)
        tot_structs += float(len(n) - 2)

        starting_outp = '%s%s in %i atom %s' % (metal1, metal2,
                                                num_atoms, shape)
        print(starting_outp.center(CENTER))

        # track min structures for each size
        new_min_structs = 0

        # sweep over different compositions
        for i, nmet2 in enumerate(n):
            # INITIALIZE POP object
            pop = Pop(nanop.get_atoms_obj_skel().copy(), nanop.bonds_list,
                      metals, shape=shape, n_metal2=nmet2, **kwargs)

            # run GA simulation
            pop.run(max_generations, min_generations, max_nochange)

            # if new minimum CE found and <save_data>
            # store result in DB
            if pop.is_new_min() and save_data:
                new_min_structs += 1
                tot_new_structs += 1
                pop.save_to_db()

        outp = 'Completed Size %i of %i (%i new mins)' % (struct_i + 1,
                                                          nstructs,
                                                          new_min_structs)
        print('-' * CENTER)
        print(outp.center(CENTER))
        print('-' * CENTER)

    # insert log results into DB
    if save_data:
        db_inter.insert_bimetallic_log(
            start_time=start_time,
            metal1=metal1,
            metal2=metal2,
            shape=shape,
            ga_generations=max_generations,
            shell_range='%i - %i' % tuple(nshell_range),
            new_min_structs=tot_new_structs,
            tot_structs=tot_structs,
            batch_run_num=batch_runinfo)


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
        nanop = structure_gen.build_structure_sql(shape, num_shells)
        for metals in metal_opts:
            # ensure metal types are sorted
            metals = sorted(metals)

            # find all bimetallic results matching shape, size, and metals
            results = db_inter.get_bimet_result(metals, shape=shape,
                                                num_shells=num_shells)

            # if no results found, continue to next metal combination
            if not results:
                continue

            # else create AtomGraph object
            atomg = atomgraph.AtomGraph(nanop.bonds_list,
                                        metals[0], metals[1])

            # iterate over results to compare CE in databse vs.
            # CE calculated with ordering
            for res in results:
                ordering = np.array(list(map(int, res.ordering)))
                n_metal2 = ordering.sum()
                actual_ce = atomg.getTotalCE(ordering)
                actual_ee = atomg.getEE(ordering)

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
    newp = build_pop_obj(metals, shape, num_shells, x_metal2=x_metal2,
                         spike=spike, **kwargs)
    randp = build_pop_obj(metals, shape, num_shells, x_metal2=x_metal2,
                          random=True)

    # run GA until it converges (no changes for <max_nochanges> generations)
    newp.run(max_gens=max_gens, min_gens=min_gens, max_nochange=max_nochange)
    print('RUNTIME: %.3f s' % newp.runtime)

    # random search for same number of generations
    randp.run(newp.max_gens, max_nochange=-1)

    # save best structure from GA and random search
    if save:
        make_file(newp.atom, newp.pop[0], os.path.join(path, 'ga.xyz'))
        make_file(randp.atom, randp.pop[0], os.path.join(path, 'random.xyz'))

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
        newp = build_pop_obj(metals, shape, num_shells,
                             x_metal2=x_metal2)
        newp.run(max_gens=500, max_nochange=-1)
        natoms.append(len(newp.atom))
        times.append(newp.runtime)
        if newp.is_new_min():
            print('New min.')
        # newp.save_to_db()

    fig, ax = plt.subplots()
    ax.plot(natoms, times, 'o-', color='green',
            ms=10, markeredgecolor='k')

    ax.set_xlabel('Number of Atoms')
    ax.set_ylabel('Runtime (seconds)\nfor 500 Generations')
    fig.tight_layout()
    return fig, ax


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
    filedir = os.path.dirname(os.path.realpath(__file__))
    fept_path = os.path.join(filedir, '..', 'data', 'fept_np')

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
    ag = atomgraph.AtomGraph(bonds.copy(), 'Fe', 'Pt')

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
    filedir = os.path.dirname(os.path.realpath(__file__))
    fept_path = os.path.join(filedir, '..', '..', 'data', 'fept_np')

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
    ag = atomgraph.AtomGraph(bonds.copy(), 'Fe', 'Pt')

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
    pop = Pop(atom, bonds.copy(), metals, shape,
              n_metal2=n_metal2, atomg=ag)

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
        fid.write('         EE: %.5f' % pop.atomg.getEE(pop[0].ordering))

    # simulate random pop for same gens as GA
    max_gens = pop.max_gens

    randpop = Pop(atom, bonds.copy(), metals, shape,
                  n_metal2=n_metal2, atomg=ag, random=True)
    randpop.run(max_gens=max_gens, max_nochange=-1)

    make_file(randpop.atom, randpop[0], os.path.join(fept_path, 'random.xyz'))

    # plot results
    fig, ax = pop.plot_results()
    randpop.plot_results(ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(fept_path, 'results.png'))
    fig.savefig(os.path.join(fept_path, 'results.svg'))
    return fig, ax


if __name__ == '__main__':
    check_db_values()
    for r in plt.rcParams:
        if plt.rcParams[r] == 'bold':
            plt.rcParams[r] = 'normal'
    # vis_FePt_results(True)
