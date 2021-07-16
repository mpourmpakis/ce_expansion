from __future__ import annotations
from ce_expansion.atomgraph import adjacency
import functools
import itertools as it
import operator as op
import pickle
import re
import time
import warnings
from datetime import datetime as dt
from datetime import timedelta
from collections import Counter, defaultdict
from typing import Iterable, List, Union

import ase
import ase.io
import matplotlib.pyplot as plt
import numpy as np
import ase.visualize

from ce_expansion.atomgraph.bcm import BCModel
from ce_expansion.atomgraph.atomgraph import AtomGraph
from ce_expansion.ga import structure_gen
from ce_expansion.npdb import db_inter


class GAError(Exception):
    """Custom error for GA simulations"""


@functools.total_ordering
class Nanoparticle:
    def __init__(self, bcm: BCModel, composition: Iterable[int],
                 ordering: Iterable[int] = None):
        """
        Nanoparticle object for GA simulations
        Represents a single structure with a given chemical ordering (arr)

        Args:
        bcm: instantiated BCModel to compute score as function of ordering
        composition: array of metal compositions:
                     [# metal1, # metal2, ..., # metaln]

        KArgs:
        ordering: array representing positions in
                  nanoparticle and whether they're
                  occupied by metal1 (0),
                  metal2 (1), ..., or metaln (n-1)
                  (Default: None - random generated)

        Raises:
        GAError: ordering kwarg does not match composition arg
        """
        self.bcm : BCModel
        self.bcm = bcm
        self.composition = np.array(composition).astype(int)
        self.num_atoms = len(self.bcm)

        # if an array is given, use it - else random generate ordering
        if ordering is not None:
            self.ordering = np.array(ordering).astype(int)

            # make sure ordering has correct composition
            comp = np.zeros(len(self.bcm.metal_types))
            counts = np.bincount(self.ordering)
            comp[:len(counts)] = counts
            if (counts != self.composition).any():
                raise GAError("Nanoparticle ordering has incorrect composition"
                              f" (should be {self.composition}).")
        else:
            # else generate random ordering
            self.ordering = np.repeat(range(len(self.bcm.metal_types)),
                                      self.composition)
            np.random.shuffle(self.ordering)

        # calculate initial CE
        self._calc_score()

    def __len__(self) -> int:
        return self.num_atoms

    def __getitem__(self, i: int) -> int:
        return self.ordering[i]

    def __lt__(self, nanop2: Nanoparticle) -> bool:
        """NP comparisons (< > == !=) use self.ce"""
        return self.ce < nanop2.ce

    def __eq__(self, nanop2: Nanoparticle) -> bool:
        """NP comparisons (< > == !=) use self.ce"""
        return self.ce == nanop2.ce

    def copy(self) -> Nanoparticle:
        """
        Creates a copy of the Nanoparticle as a new instance

        Returns:
        exact copy of the Nanoparticle instance
        """
        return Nanoparticle(self.bcm, self.composition, ordering=self.ordering.copy())

    def mate(self, nanop2: Nanoparticle) -> List[Nanoparticle]:
        """
        Pairwise crossover algorithm to mix two parent NPs
        into two new child NPs, taking traits from each parent
        - conserves metal composition
        - about O(N^2) scaling
        - NOTE: only swaps matching pairs
                (a-b at pos i can only swap with b-a at other pos j!=i)

        Args:
        nanop2: second parent Nanoparticle

        Returns:
        two children Nanoparticles with new orderings
        """
        # parents = parent ordering arrays
        parent1 = self.ordering
        parent2 = nanop2.ordering

        # create children ordering arrays
        child1, child2 = [p.copy() for p in [parent1, parent2]]

        # if parents are identical, just mutate to make children
        if (child1 == child2).all():
            children = [Nanoparticle(self.bcm, self.composition, c)
                        for c in [child1, child2]]
            n_mut = len(self) // 2
            children[0].mutate(n_mut)
            children[1].mutate(n_mut)
            return children

        # 1) make hash tables of all differences
        # O(n)
        diffmap = defaultdict(lambda: defaultdict(set))
        diffset = defaultdict(set)
        diffs = []
        indices = np.arange(len(parent1))
        np.random.shuffle(indices)
        for i in indices:
            a, b = parent1[i], parent2[i]
            if a != b:
                diffmap[b][a].add(i)
                diffset[b].add(a)
                diffs.append(i)
        
        # shuffle diff indices
        np.random.shuffle(diffs)
        diffs = set(diffs)

        # 2) find matchine pairs to swap
        # O(n^2)
        to_swap = []
        max_swaps = len(diffs) // 3
        while len(to_swap) < max_swaps:
            # break out when out of diff positions
            if not diffs:
                break
            i = diffs.pop()
            a, b = parent1[i], parent2[i]
            if b in diffset[a]:
                avail = diffmap[a][b] & diffs
                if avail:
                    j = avail.pop()
                    diffs.remove(j)
                    to_swap.extend([i, j])

        # make the swaps
        child1[to_swap], child2[to_swap] = child2[to_swap], child1[to_swap]

        # return the children
        return [Nanoparticle(self.bcm, self.composition, c)
                for c in [child1, child2]]

    def mutate(self, n_swaps: int = 1):
        """
        Algorithm to randomly swap positions within ordering array
        - O(n) scaling [n == number of atoms]

        Args:
        n_swaps: number of swaps to make
                 (Default: 1)
        """
        # get all indices of ordering positions
        indices = np.arange(self.num_atoms)

        # randomly select <n_swaps> pairs of positions to swap
        pos = np.random.choice(indices, size=(n_swaps, 2), replace=False)

        # we will swap each position <i> with position <j>
        i, j = np.hsplit(pos, 2)

        # apply all swaps simultaneously
        self.ordering[i], self.ordering[j] = self.ordering[j], self.ordering[i]

        # recalculate CE
        self._calc_score()

    def _bimetallic_mate(self, nanop2: Nanoparticle) -> List[Nanoparticle]:
        """
        Pairwise crossover algorithm to mix two parent chromosomes into
        two new child chromosomes, taking traits from each parent
        - conserves doping concentration
        - about O(N^2) scaling

        Args:
        nanop2: second parent Chromo obj

        Returns:
        two children Chromo objs with new ordering <arr>s
        """
        child1 = self.ordering.copy()
        child2 = nanop2.ordering.copy()

        assert (child1.sum() == child2.sum() ==
                self.n_metal2 == nanop2.n_metal2)

        # if only one '1' or one '0' then mating is not possible
        if self.n_metal2 in [1, self.num_atoms - 1]:
            return [self.copy(), nanop2.copy()]

        # indices where a '1' is found in child1
        ones = np.where(child1 == 1)[0]

        # indices where child1 and child2 do not match
        diff = np.where(child1 != child2)[0]

        # must have at least two different sites
        if len(diff) < 2:
            return [self.copy(), nanop2.copy()]

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
                self.n_metal2 == nanop2.n_metal2)

        children = [Nanoparticle(self.bcm, n_metal2=self.n_metal2,
                                 ordering=child1.copy()),
                    Nanoparticle(self.bcm, n_metal2=self.n_metal2,
                                 ordering=child2.copy())]
        return children

    def _bimetallic_mutate(self, n_swaps: int = 1):
        """
        Algorithm to randomly switch a '1' & a '0' within ordering array
        - about O(n) scaling

        Args:
        n_swaps: number of swaps to make
                 (Default: 1)

        Raises:
        ValueError: if not bcm, Chromo can not and
                    should not be mutated
        """
        if not self.bcm:
            raise GAError("Mutating Chromo should only be done through"
                          "Pop simulations")

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
                i = np.random.randint(0, self.num_atoms - 1)

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
        self._calc_score()

    def _calc_score(self):
        """
        Returns CE of structure based on Bond-Centric Model
        - Yan, Z. et al., Nano Lett. 2018, 18 (4), 2696-2704.
        """
        self.ce = self.bcm.calc_ce(self.ordering)


class GA(object):
    def __init__(self, bcm: BCModel, composition: Iterable[int], shape: str,
                 popsize: int = 50, mute_pct: float = 0.8,
                 n_mute_atomswaps: int = None, spike: bool = False,
                 random: bool = False, e: int = 1, save_every: int = 100,
                 use_metropolis: bool = False):
        """
        Polymetallic nanoparticle genetic algorithm
        - initialize a population of nanoparticles
        - call self.run() to run a GA simulation
        - if random=True, self.run() will conduct a random search

        Args:
        bcm: BCModel containing atom skeleton info + metal_types
        composition: metal counts
        shape (str): nanoparticle shape

        KArgs:
        popsize (int): size of population (Default: 50)
        mute_pct: percentage of population to mutate each generation
                  (Default: 0.8 = 80%)
        n_mute_atomswaps: number of atom swaps to make
                          during a mutation
                          (Default: None: half min(n_metal1, n_metal2))
        spike: if True, following structures are added to generation 0
               - if same structure and composition found in DB,
                 it is added to population
               - minCN-filled or maxCN-filled nanoparticle
                 (depending on which is more fit) also added
                 to population - structure created using
                 fill_cn function
        random: if True, self.run does a random search and
                does not perform a GA simulation
        e: exploration - exploitation factor used to bias parent
           selection probabilities (Default: 1 = No effect)
        save_every: choose how often pop CEs are stored in all_data
        use_metropolis: use metropolis algorithm at end of GA sim
        """
        # store the datetime GA was instantiated
        self.dt_created = dt.now()

        # NP parameters
        self.bcm = bcm
        self.num_atoms = len(self.bcm)

        self.composition = np.array(composition).astype(int)

        # GA only works with polymetallic NPs (# metal > 1)
        nanop_is_monometallic = (self.composition == self.composition.sum()).any()
        if nanop_is_monometallic:
            raise GAError("GA is unnecessary for a monometallic Nanoparticle.")

        # make sure composition adds up to number of atoms
        if self.composition.sum() != len(self.bcm):
            raise GAError("Composition passed to GA does not match number of atoms")

        # np shape / name ID
        self.shape = shape

        # GA simulation parameters
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
            self.n_mute_atomswaps = max(min(self.composition) // 50, 1)

        # spike ga with previous run and structures from fill_cn
        self.spike = spike

        # NOTE: spike currently does nothing. Warn user if spike is True
        if self.spike:
            warnings.WarningMessage("spike=True does not change GA simulation -"
                                    " still need to implement polymetallic spiking methods")

        # keep track of how many times the sim has been continued
        self.continued = 0

        # previous GA sim results (npdb.datatables.PolymetallicResults)
        self.prev_results = None

        # whether or not to apply GA mating and mutation
        # (or just randomly generate orderings)
        self.random = random

        # exploration-exploitation factor
        self.e = e

        # store CEs of pops during simulation
        self.all_data = {}

        # how often pop data should be saved to all_data
        self.save_every = save_every

        # track whether metropolis should be used at end of GA sim
        self.use_metropolis = use_metropolis

        # create formula string
        self.formula = ''
        for m, n in zip(self.bcm.metal_types, self.composition):
            self.formula += f'{m}{n}'

        # population - list of chromosomes
        self.pop = []
        self.initialize_new_run()

    def __len__(self) -> int:
        return self.popsize

    def __getitem__(self, i: int) -> Nanoparticle:
        return self.pop[i]

    def run(self, max_gens: int = -1, max_nochange: int = 750, min_gens: int = -1):
        """
        Main method to run a GA simulation

        KArgs:
        max_gens: maximum generations the GA will run
                  -1: the GA only stops based on <max_nochange>
        max_nochange: specifies max number of generations GA will run
                      without finding a better NP
                      -1: GA will run to <max_gens>
        min_gens: minimum generations that the GA runs before checking
                  the max_nochange criteria
                  -1: no minimum

        Raises:
        - GAError: can only call run for first GA sim
        - GAError: max_gens and max_nochange cannot both equal -1
        """
        if self.has_run:
            raise GAError("Simulation has already run. "
                          "Please use continue method.")
        elif max_gens == max_nochange == -1:
            raise GAError("max_gens and max_nochange cannot both be "
                          "turned off (equal: -1)")

        self.max_gens = max_gens

        # GA will not continue if <max_nochange> generations are
        # taken without a change in minimum CE
        nochange = 0

        # store generation 0 into all_data dict
        self.all_data[self.generation] = [i.ce for i in self]

        # begin the simulation!
        breakline = '-' * 50
        print(breakline)
        print(f'GA Sim for {self.formula} - {self.shape}:')
        start = time.time()
        while self.generation != self.max_gens:
            # step to next generation
            self._step()

            # store all CE values periodically
            if self.generation % self.save_every == 0:
                self.all_data[self.generation] = [i.ce for i in self]

            # track if there was an improvement to best NP
            if (max_nochange and min_gens and
                    self.generation > min_gens):
                if self.stats[-1][0] == self.stats[-2][0]:
                    nochange += 1
                else:
                    nochange = 0

                # if no change has been made after <max_nochange>, stop GA
                if nochange == max_nochange:
                    break

        # capture runtime in seconds
        self.runtime += time.time() - start

        # sort final population
        self.sort_pop()

        # get info for last generation
        self.all_data[self.generation] = [i.ce for i in self]

        # print status of final generation
        self._print_status(end='\n')

        # set max_gens to actual generations simulated
        self.max_gens = self.generation

        # run James' metropolis algorithm function to search for
        # minimum struct near current min
        if not self.random and self.use_metropolis:
            print('Running metropolis.')
            best = min(self)
            best_ordering = best.ordering.copy()
            opt_order, opt_ce, _ = self.bcm.metropolis(
                best_ordering,
                num_steps=5000)

            # if metropolis alg finds a new minimum,
            # drop bottom one from pop
            if opt_ce < best.ce:
                print('Found new min with metropolis!')
                new_best = Nanoparticle(self.bcm, self.composition, opt_order)
                if new_best.ce < best.ce:
                    self.pop = [new_best] + self[:-1]
                self._update_stats()

        # convert stats to an array
        self.stats = np.array(self.stats)

        self.has_run = True

        # print summary of simulation
        self.summ_results(display=True)
        print(breakline)

    def continue_run(self, max_gens: int = -1, max_nochange: int = 50, min_gens: int = -1):
        """
        Used to continue GA sim from where it left off

        KArgs:
        max_gens: maximum generations the GA will run
                  -1: the GA only stops based on <max_nochange>
        max_nochange: specifies max number of generations GA will run
                      without finding a better NP
                      -1: GA will run to <max_gens>
        min_gens: minimum generations that the GA runs before checking
                  the max_nochange criteria
                  -1: no minimum
        """
        self.has_run = False
        self.stats = self.stats.tolist()
        self.run(max_gens=max_gens, max_nochange=max_nochange,
                 min_gens=min_gens)
        self.continued += 1

    def initialize_new_run(self):
        """
        Sets up for a new GA simulation (generation 0)
        - can simulate different cases by changing composition
          then running this method
        """
        # search for previous polymetallic result
        self.prev_results = db_inter.get_polymet_result(
            metals=self.bcm.metal_types,
            composition=self.composition,
            num_atoms=len(self.bcm),
            shape=self.shape,
            lim=1)

        self._initialize_pop()

        self.orig_min = min(self).ce

        self.stats = []
        self._update_stats()

        # track runtime
        self.runtime = 0

        # track generations
        self.generation = 0

        # minimum CE of last generation
        self.last_min = 100

        # keep track of whether a sim has been run
        self.has_run = False

    def is_new_min(self, check_db: bool = True) -> bool:
        """
        Returns True if GA sim found new minimum CE
        (compares to SQL DB if <check_db>)

        KArgs:
        check_db: if True, only compares to database min
                  else it'll compare to generation 0
        """
        cur_min = min(self).ce
        if check_db:
            if not self.prev_results:
                return True
            else:
                return cur_min < self.prev_results.CE

        return cur_min < self.orig_min

    def make_atoms_object(self, np_index: int = 0) -> ase.Atoms:
        """
        Makes atoms object of the calculated optimized atom

        KArgs:
        np_index (int): Creates atoms object of the desired nanoparticle
                        (0 being the most optimized nanoparticle)

        Returns:
        the desired atoms object
        """
        self.sort_pop()

        # Get atom ordering and number of unique atoms
        element_ordering = self.pop[np_index].ordering
        # makes new array of strings which can be used with ASE's ".symbols"

        elements = np.array(self.bcm.metal_types)[element_ordering]

        atoms = self.bcm.atoms.copy()
        atoms.symbols = elements

        return atoms

    def plot_results(self, save_path : str = None, ax: plt.Axes = None,
                     show: bool = False) -> Union[plt.Figure, plt.Axes]:
        """
        Method to create a plot of GA simulation
        - plots average, std deviation, and minimum score
          of the population at each step

        KArgs:
        save_path: path and file name to save the figure
                        - if None, figure is not saved
        ax: if given, results will be plotted on axis
        show: if True, call plt.show()

        Returns:
        fig and ax objs
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

        # minimum, average, and std deviation scores of population at each step
        low = self.stats[:, 0]
        mean = self.stats[:, 1]
        std = self.stats[:, 2]

        # plot minimum CE as a dotted line
        ax.plot(low, ':', color=color)
        # light blue fill of one std deviation
        ax.fill_between(range(len(self.stats)), mean + std, mean - std,
                        color=fillcolor)

        # plot mean as a solid line and minimum as a dotted line
        ax.plot(mean, color=color)

        # if any generations stored in all_data, plot them as scatter points
        for gen in self.all_data:
            ax.scatter(np.repeat(gen, self.popsize), self.all_data[gen],
                       color='gray', alpha=0.5, edgecolor='k')

        if not secondary:
            # create legend
            ax.plot([], [], ':', color='gray', label='MIN')
            ax.plot([], [], color='k', label='MEAN')
            ax.fill_between([], 0, 0, color='k', label='STD')
            ax.legend(ncol=3, loc='upper center')
            ax.set_ylabel('Cohesive Energy (eV / atom)')
            ax.set_xlabel('Generation')

            # format latex formula for plot title
            tex_form = re.sub('([A-Z][a-z]?)([0-9]+)', '\\1_{\\2}', self.formula)
            title = f'$\\rm{tex_form}$'
            if self.shape:
                title += f' -- {self.shape.title()}'
            title += f'\n{low[-1]:.3f} eV/atom'
            ax.set_title(title, fontdict={'weight': 'normal'})
            fig.tight_layout()

        # save figure if <save_path> was specified
        if save_path:
            fig.savefig(save_path)

        if show:
            plt.show()

        return fig, ax

    def save_ga_pickle(self, path: str = None) -> str:
        """
        Saves the GA instance as a pickle

        Args:
        path (str): path to save pickle file
                    - can include filename

        Returns:
        path of ga sim pickle file
        """
        # if path doesn't include a filename, make one
        if path is None:
            # start file name with datetime of when GA object was created
            path = self.dt_created.strftime('%Y-%m-%d__%H-%M-%S')
            path += f'__ga-sim__{self.formula}__{self.shape[:10]}.pickle'

        # pickle self
        with open(path, 'wb') as fidw:
            pickle.dump(self, fidw, protocol=pickle.HIGHEST_PROTOCOL)
    
        return path

    def save_to_db(self):
        """
        Save minimum CE nanoparticle to database
        - connection is made through npdb.db_inter
        - only updates database if new min is found
        """
        if not self.is_new_min():
            print('Database contains NP with as good/better fitness.')
            print('No action taken.')
            return

        # get best chrom
        best = min(self)

        # try to find nanoparticle in DB
        db_nanop = db_inter.get_nanoparticle(
            self.shape,
            num_atoms=self.num_atoms,
            lim=1)

        # if not found, add a new nanoparticle to DB
        if not db_nanop:
            db_nanop = db_inter.insert_nanoparticle(self.atom, self.shape,
                                                    self.num_shells)

        if best.ce != best.bcm.calc_ce(best.ordering):
            raise GAError("Nanoparticle CE does not match ordering!")

        # update DB
        db_inter.update_polymet_result(
            metals=self.bcm.metal_types,
            composition=self.composition,
            shape=self.shape,
            CE=best.ce,
            EE=best.bcm.calc_ee(best.ordering),
            ordering=best.ordering,
            nanop=db_nanop,
            allow_insert=True)

        print('New min NP added to database.')

    def sort_pop(self):
        """
        Sorts population based on cohesive energy
        - lowest cohesive energy = most fit = first in list
        """
        self.pop = sorted(self.pop)

    def summ_results(self, display: bool = False) -> str:
        """
        Creates string listing GA simulation stats and results info
        str includes:
            - minimum, mean, and std dev. of CEs for last population
            - mute and mate (crossover) rates of GA
            - total steps taken by GA
            - structure formula

        KArgs:
        display: if True, results string is printed to console

        Returns:
        result string

        Raises:
        GAError: if sim has not been run (has_run = False)
        """
        if not self.has_run:
            raise GAError('No simulation has been run')

        best = self.stats[:, 0].min()
        worst = self.stats[:, 0].max()

        res =  f' Form: {self.formula}\n'
        res += f'nAtom: {len(self.bcm)}\n'
        res += f'nGens: {self.max_gens}\n'
        res += f'Start: {worst:.5f} eV/atom\n'
        res += f' Best: {best:.5f} eV/atom\n'

        diff = best - worst
        diff_pct = diff / worst
        res += f' Diff: {best-worst:.5} eV/atom ({diff_pct:.2%})\n'

        res += f' Mute: {self.mute_pct:.1%}\n'
        res += f'  Pop: {self.popsize}\n'

        if self.has_run:
            res += f' Time: {timedelta(seconds=int(self.runtime))}'

        if display:
            print(res)

        return res

    def view_np(self, np_index: int = 0):
        """
        Views the selected nanoparticle of the last genenration

        KArgs:
        np_index (int): Creates atoms object of the desired nanoparticle
                        (0 being the most optimized nanoparticle)
        """
        ase.visualize.view(self.make_atoms_object(np_index))

    def _initialize_pop(self):
        """
        Initialize population of Nanoparticle objects
        - if self.random, population filled completely with random structures
        - if self.spike, following structures are added
            - if same structure and composition found in DB, it is added to
              population
            - minCN-filled or maxCN-filled nanoparticle (depending on which is
              more fit) also added to population - structure created using
              fill_cn function
        """
        # if random and stepping to next generation, keep current best
        if self.random:
            if self.pop:
                self.pop = [min(self)]

        # if not random and spike, add max or min CN-filled structure
        # plus current best found in database (if available)
        elif self.spike:
            # overload min and max CN with specific metal type
            mincn = fill_cn(self.bcm, self.composition, low_first=True, return_n=1)
            maxcn = fill_cn(self.bcm, self.composition, low_first=False, return_n=1)

            # initialize Nanoparticle objects with overloaded orderings
            spikes = [Nanoparticle(self.bcm, self.composition, cn_case)
                      for cn_case in (mincn, maxcn)]
            # only add the lowest-CE Nanoparticle (most stable / fittest)
            self.pop.append(min(spikes, key=lambda nanop: nanop.ce))

            # add current min CE structure if it exists
            if self.prev_results:
                nanop = Nanoparticle(self.bcm, self.composition, self.prev_results.ordering)
                self.pop.append(nanop)

        # create random structures for remaining popsize
        nanops_needed = self.popsize - len(self.pop)
        self.pop.extend(Nanoparticle(self.bcm, self.composition)
                        for _ in range(nanops_needed))

        # sort initial population
        self.sort_pop()

        # define array of pop indices (used in roulette_mate)
        # NOTE: do it here to ensure the array is correct in case 
        #       popsize changes between sims
        self._pop_indices = np.arange(len(self.pop))

    def _print_status(self, end: str = '\r'):
        """
        Prints info on current generation of GA

        KArgs:
        end: string used to end print statement
             - \r allows generations to overwrite on same line
               during simulation
        """
        # format of string to be written to console during sim
        string = f' Min: {self.stats[-1][0]:.5f} eV/atom'
        string += f' -- Gen: {self.generation:05d}'
        print(string, end=end)

    def _roulette_mate(self):
        """
        Roulette Wheel selection algorithm that chooses mates
        based on their fitness
        - probability of Nanoparticle <i> being selected:
          fitness_i = (CE_i - min(CE))^(e)
          P_i = (fitness_i / sum of all fitnesses)
        - fitter Nanoparticles more likely to be selected, but not guaranteed
          to help mitigate lack of diversity in population
        """
        # compute array of probabilities
        ces = np.array([abs(p.ce) for p in self])
        fitness = (ces - ces.min())**self.e
        probabilities = abs(fitness / fitness.sum())

        # select parent Nanoparticle pairs based on probabilities
        parents = np.random.choice(self._pop_indices,
                                   size=(self.popsize, 2),
                                   p=probabilities)

        # drop duplicate pairs - NP can't mate with itself
        parents = parents[parents[:, 0] != parents[:, 1]]

        # create children by mating parent pairs
        children = [child for p1, p2 in parents
                    for child in self[p1].mate(self[p2])]

        # keep the previous minimum NP
        self.pop = [min(self)] + children
    
        # ensure population is correct size (drops children if necessary)
        self.pop = self[:self.popsize]

    def _step(self):
        """
        Wrapper method that takes GA to next generation
        mates the population
        mutates the population
        calculates statistics of new population
        resorts population and increments self.generation
        """
        if self.random:
            self._initialize_pop()
        else:
            # MATE
            self._roulette_mate()

            # MUTATE - does not mutate most fit Nanoparticle
            for r in np.random.randint(1, self.popsize, size=self.n_mute):
                self[r].mutate(self.n_mute_atomswaps)

        self._update_stats()
        self._print_status()

        # increment generation
        self.generation += 1

    def _update_stats(self):
        """
        Adds statistics of current generation to self.stats
        """
        s = np.array([i.ce for i in self.pop])
        self.stats.append([s.min(),  # minimum CE
                           s.mean(), # mean CE
                           s.std()]) # STD CE


def build_ga(atoms: ase.Atoms, metal_types: Iterable[str] = None,
             composition: Iterable = None, shape: str = '',
             bonds: Iterable[int] = None, **ga_kwargs) -> GA:
    """
    Initializes a GA for the specified shape,
    metals, and number of shells
    - e.g. 55-atom (3 shell) AgCu Icosahedron Pop
    - **kwargs gets passed directly into Pop.__init__

    Args:
    atoms: atoms object of NP skeleton (holds atomic positions)
    metal_types: list of metal types that should make up NP
                 NOTE: this list will get sorted by BCModel
    composition: list of metal counts | % composition of each metal

    KArgs:
    shape: shape/name of NP
    bonds: array of atom indices that make up bonds
    **ga_kwargs: valid arguments to initialize GA object
                 e.g. popsize=50, save_every=25

    Returns:
    GA instance
    """
    # get metal info from atoms object if only atoms is given
    if metal_types is None or composition is None:
        metal_types = sorted(set(atoms.symbols))

        # use a counter to get metal counts
        counts = Counter(atoms.symbols)
        composition = [counts[m] for m in metal_types]

    # if composition is metal %'s, convert to metal counts
    elif np.isclose(1, sum(composition)):
        composition = (np.array(composition) * len(atoms)).astype(int)
        composition[-1] += len(atoms) - composition.sum()

    if bonds is None:
        bonds = adjacency.build_bonds_arr(atoms)

    bcm = BCModel(atoms, metal_types=metal_types, bond_list=bonds)
    ga = GA(bcm, composition, shape, **ga_kwargs)
    return ga


def load_ga_pickle(path : str) -> GA:
    """
    Load a pickled GA object

    Args:
    path: path to Pop object (*.pickle file)

    Returns:
    GA object
    """
    with open(path, 'rb') as fid:
        unpickler = pickle.Unpickler(fid)
        ga = unpickler.load()
    return ga


def fill_cn(bcm, n_metal2, max_search=50, low_first=True, return_n=None,
            verbose=False):
    """
    NOTE: Most likely broken - still need to extend to polymetallic cases

    Algorithm to fill the lowest (or highest) coordination sites with 'metal2'

    Args:
    bcm (atomgraph.AtomGraph | atomgraph.BCModel): bcm obj
    n_metal2 (int): number of dopants

    KArgs:
    max_search (int): if there are a number of possible structures with
                      partially-filled sites, the function will search
                      max_search options and return lowest CE structure
                      (Default: 50)
    low_first (bool): if True, fills low CNs, else fills high CNs
                      (Default: True)
    return_n (bool): if > 0, function will return a list of possible
                     structures
                     (Default: None)
    verbose (bool): if True, function will print info to console
                    (Default: False)

    Returns:
    if return_n > 0:
        (list): list of chemical ordering np.ndarrays
    else:
        (np.ndarray), (float): chemical ordering np.ndarray with its                             calculated CE

    Raises:
    ValueError: not enough options to produce <return_n> sample size
    """

    def ncr(n: int, r: int) -> int:
        """
        N choose r function (combinatorics)

        Args:
        n: from 'n' choices
        r: choose r without replacement

        Returns:
        total combinations
        """
        r = min(r, n - r)
        numer = functools.reduce(op.mul, range(n, n - r, -1), 1)
        denom = functools.reduce(op.mul, range(1, r + 1), 1)
        return numer // denom

    # handle monometallic cases efficiently
    if n_metal2 in [0, bcm.num_atoms]:
        # all '0's if no metal2, else all '1' since all metal2
        struct_min = [np.zeros, np.ones][bool(n_metal2)](bcm.num_atoms)
        struct_min = struct_min.astype(int)
        ce = bcm.getTotalCE(struct_min)
        checkall = True
    else:
        cn_list = bcm.cns
        cnset = sorted(set(cn_list))
        if not low_first:
            cnset = cnset[::-1]
        struct_min = np.zeros(bcm.num_atoms).astype(int)
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
                        pick = np.random.sample(list(spots), n_metal2)
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
                        pick = np.random.sample(list(spots), n_metal2)
                    base[pick] = 1
                    checkce = bcm.getTotalCE(base)
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
        ce = bcm.getTotalCE(struct_min)
    if return_n:
        return [struct_min]
    return struct_min, ce


if __name__ == '__main__':
    np.random.seed(15213)

    metal_types = ['Ag', 'Au', 'Cu']
    num_shells = 1

    # create icosahedron atoms objedct
    shape = 'icosahedron'
    atoms = structure_gen.NPBuilder.icosahedron(num_shells)

    # create a random composition
    comp = np.random.random(size=len(metal_types))
    comp = (len(atoms) * comp / comp.sum()).astype(int)
    comp[-1] += len(atoms) - comp.sum()

    # set composition of atoms object to <comp>
    atoms.symbols = np.repeat(metal_types, comp)

    # create ga object and run simulation
    ga = build_ga(atoms)
    ga.run(max_gens=100)
    ga.plot_results(show=True)
