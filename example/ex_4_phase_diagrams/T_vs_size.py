#!/usr/bin/env python

import collections
import copy
import os
import sys

import matplotlib.cm
import matplotlib.lines
import matplotlib.pyplot as plt
import numpy as np

data = os.path.join(os.path.realpath(__file__), '..', '..', '..', 'data', 'larson_et_al')
sys.path.append(data)
import npdb.db_inter

DEFAULT_DPI = 600  # Dots per inch
DEFAULT_POINTSIZE = 15
DEFAULT_MARKER = "o"  # square


def magic_number(n):
    start = n
    while True:
        yield int((2 * n + 1) * (5 * (n ** 2) + 5 * n + 3) / 3)
        n += 1


class Point(object):
    def __init__(self):
        self.comp = None
        self.size = None
        self.temp = None
        self.shape = None
        self.color = None
        self.symbol = DEFAULT_MARKER
        self.symbol_size = DEFAULT_POINTSIZE
        self.energy = None


class Result(object):
    def __init__(self, shape, size, composition, excess_energy, cohesive_energy):
        self.shape = shape
        self.size = size
        self.composition = composition
        self.cohesive_energy = cohesive_energy
        self.excess_energy = excess_energy

    def free_energy(self, T):
        """
        Calculates Excess energies plus an entropic contribution.

        Args:
            excess_energy (list): Excess energies from DB query
            comp (list): Compositions from DB query
            T (int): Temperature

        Returns:
            Free energy of mixing = excess energy (related to enthalpy of mixing) - entropy of mixing
        """

        if self.composition == 1 or self.composition == 0:
            return 0

        # k_b T [eV] = (25.7 mEV at 298 K)
        kt = 25.7E-3 / 298 * T
        del_s = self.composition * np.log(self.composition) + (1 - self.composition) * np.log(1 - self.composition)
        del_s *= -kt

        free_energy = self.excess_energy - del_s
        return free_energy


class OrderedSet(collections.UserList):
    """
    Wrapper around a list that allows it to operate somewhat like a set. Maintains the order things were added.
    """

    def add(self, value):
        """
        If the value passed in is not in the set, then adds it to the set. Otherwise, does nothing.

        :param value: The value to be added.
        """
        if value in self.data:
            pass
        else:
            self.data.append(value)


def get_best(alloy,
             size_range,
             temperature_range, temperature_res=100):
    """
    Produces a phase diagram

    Args:
        alloy (str): Alloy of interest
        size_range (list): Range of sizes to consider, in atoms, inclusive
        temperature_range (list): Range of temperatures to consider, in K, inclusive
        temperature_res (int): How fine our temperature mesh is, in K. Default = 1K
    Returns:
        A dictionary with {size : {temp : Result_object() } }
    """

    # Book-keeping and initialization
    shapes = ["icosahedron", "cuboctahedron", "elongated-pentagonal-bipyramid"]
    min_size = size_range[0]
    max_size = size_range[1]

    # Set the list of temperatures
    temperatures = np.arange(temperature_range[0], temperature_range[1] + temperature_res, temperature_res)

    # Set the list of sizes
    sizes = []
    magic_numbers = magic_number(0)
    for size in magic_numbers:
        if size < min_size:
            continue
        if size > max_size:
            break
        else:
            sizes.append(size)

    # DB Query
    best_results = {}
    for shape in shapes:
        for size in sizes:
            query = npdb.db_inter.get_bimet_result(metals=alloy, shape=shape, num_atoms=size)
            if len(query) > 0:
                best_results[size] = {}
                for result in query:
                    composition = np.round(result.n_metal1 / result.num_atoms, decimals=2)
                    excess_energy = result.EE
                    cohesive_energy = result.CE
                    result = Result(shape, size, composition, excess_energy, cohesive_energy)
                    for temp in temperatures:
                        try:
                            g = result.free_energy(temp)
                            if best_results[size][temp].free_energy(temp) > g:
                                best_results[size][temp] = copy.deepcopy(result)
                        except KeyError:
                            best_results[size][temp] = result
    return best_results


def plot_phase(results, alloy):
    """
    Plots a phase diagram
    Args:
        Results in the form of a dictionary. Format is {size : {temp : Result_object } }
    Returns:
        None
    """
    # Set up colors
    comprange = np.linspace(10, 110, 100) / 100
    reds = matplotlib.cm.Reds(comprange)
    blues = matplotlib.cm.Blues(comprange)
    greens = matplotlib.cm.Greens(comprange)
    color_map = {"icosahedron":reds,
                 "elongated-pentagonal-bipyramid":greens,
                 "cuboctahedron":blues}

    fig, ax = plt.subplots()
    xs = []
    ys = []
    colors = []
    for size in results:
        for temp in results[size]:
            xs.append(size)
            ys.append(temp)
            colors.append(color_map[results[size][temp].shape][int(results[size][temp].composition*100)])

    ax.scatter(xs, ys, c=colors, marker="s", s=5)
    plt.title(alloy)
    plt.xlabel("Size (N)")
    plt.ylabel("Temp (K)")

    # Set up legend
    labels = []
    entries = []
    abbreviations = {"icosahedron": "ico",
                     "cuboctahedron": "cuboct",
                     "elongated-pentagonal-bipyramid": "epb"}
    for shape in color_map:
        abbrev = abbreviations[shape]
        cmap = color_map[shape]
        labels.append("Pure " + alloy[0:2] + " " + abbrev)
        entries.append(matplotlib.lines.Line2D([0], [0], lw=0, markersize=10, marker="s", color=cmap[0]))
        labels.append("Pure " + alloy[2:] + " " + abbrev)
        entries.append(matplotlib.lines.Line2D([0], [0], lw=0, markersize=10, marker="s", color=cmap[-1]))

    plt.legend(entries, labels, loc="best", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()


    plt.savefig(alloy + ".png")
    plt.close()


for i in ["AgAu", "AgCu", "AuCu"]:
    best = get_best(i, [13, 20000], [0, 1000], 5)
    plot_phase(best, i)
