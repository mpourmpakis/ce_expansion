#!/usr/bin/env python

import os
import sys

import matplotlib.lines
import matplotlib.pyplot as plt
import numpy as np

data = os.path.join(os.path.realpath(__file__), '..', '..', '..', 'data', 'larson_et_al')
sys.path.append(data)
import npdb.db_inter

DEFAULT_DPI = 600  # Dots per inch


class Result(object):
    def __init__(self, shape, size, composition, excess_energy):
        self.shape = shape
        self.size = size
        self.composition = composition
        self.bin_index = None
        self.excess_energy = excess_energy

    def set_bin_index(self, index):
        assert index is not None
        self.bin_index = index
        assert self.bin_index is not None


def get_best(alloy,
             size_range):
    """
    Produces a phase diagram

    :param alloy: Alloy of interest
    :param size_range: Range of sizes to consider, in atoms, inclusive
    :return: None. Drops a plot of the alloy into the working directory.
    """

    # Book-keeping and initialization
    shapes = ["icosahedron", "cuboctahedron", "elongated-pentagonal-bipyramid"]
    min_size = size_range[0]
    max_size = size_range[1]
    compositions = np.arange(0, 1 + 0.01, 0.01)

    # DB Query
    results = []
    for shape in shapes:
        query = npdb.db_inter.get_bimet_result(metals=alloy, shape=shape)
        for result in query:
            # Calculate size
            size = result.num_atoms
            if size <= min_size or size >= max_size:
                continue
            # Calculate composition
            composition = np.round(result.n_metal1 / result.num_atoms,
                                   # Correctly round based on composition resolution
                                   decimals=2)
            excess_energy = result.EE
            # Add to the list of results objects
            results.append(Result(shape, size, composition, excess_energy))

    # Bin the results by composition
    bin_indices = np.digitize(x=list(map(lambda i: i.composition, results)),
                              bins=compositions,
                              right=True
                              )
    for result, index in zip(results, bin_indices):
        result.bin_index = index

    # Find the best structure at each bin
    points = []
    for bin in set(bin_indices):
        best_structure = None
        for result in results:
            if result.bin_index == bin:
                if best_structure is None:
                    best_structure = result
                elif result.excess_energy < best_structure.excess_energy:
                    best_structure = result
            else:
                continue
        if best_structure is None:
            continue
        else:
            comp = best_structure.composition
            shape = best_structure.shape
            size = best_structure.size
            energy = best_structure.excess_energy
        point = [comp, shape, size, energy]
        points.append(point)
    return points


def magic_number(start=0):
    """
    NP magic numbers. Uses formula from the Online Encyclopedia of Integer Sequences, Sequence#A005902
    :return: Next number in sequence.
    """
    n = start
    while True:
        n += 1
        result = (2 * n + 1) * (5 * n ** 2 + 5 * n + 3) / 3
        yield int(result)


def make_bar(results, alloy):
    """
    Does the plotting and makes the phase diagram

    :param results: The results to plot.
    :param alloy: The alloy.

    :return: Nothing
    """
    points = []
    colors = []
    sizes = set()
    # Set up the results for plotting
    for result in results:
        comp, shape, size, energy = result
        sizes.add(size)
        point = [comp, size]
        points.append(point)
        # Coloring
        mapping = {"icosahedron": "#FF5555",
                   "elongated-pentagonal-bipyramid": "#55FF55",
                   "cuboctahedron": "#5555FF"}
        colors.append(mapping[shape])

    # Magic numbers needed for lines
    gen = magic_number()
    magic_numbers = [next(gen)]
    while max(magic_numbers) < 2057:
        magic_numbers.append(next(gen))
    fig, ax = plt.subplots()
    plt.title(alloy)
    ax.bar(x=[x[0] for x in points], height=[y[1] for y in points], width=1.01 / len(points), color=colors)

    # Set up minor gridlines
    plt.minorticks_on()
    ax.grid(b=True, which="minor", axis="y", color="0", alpha=0.2)
    ax.set_yticks(magic_numbers, minor=True)
    for n in magic_numbers:
        label = str(n)
        ax.text(1.01, n, label)

    # Set up legend
    custom_legend = [matplotlib.lines.Line2D([0], [0], marker="s", color="#FF5555", lw=0),
                     matplotlib.lines.Line2D([0], [0], marker="s", color="#55FF55", lw=0),
                     matplotlib.lines.Line2D([0], [0], marker="s", color="#5555FF", lw=0)]
    ax.legend(custom_legend, ["Icosahedron", "Elongated-Pentagonal-Bipyramid", "Cuboctahedron", ], loc="upper left")

    # Set up axis labels and limits
    plt.xlabel("% " + alloy[0:2])
    plt.xlim([0, 1])
    plt.ylabel("Size (N Atoms)")
    plt.ylim([0, 2060])

    # Layout and save
    plt.tight_layout()
    plt.savefig("figure_" + alloy + ".png", dpi=600)
    plt.close()


alloys = ["AgAu", "AgCu", "AuCu"]
for alloy in alloys:
    results = get_best(alloy, [0, 100000])
    make_bar(results, alloy)
