#!/usr/bin/env python

import os
import sys

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

data = os.path.join(os.path.realpath(__file__), '..', '..', '..', 'data', 'larson_et_al')
sys.path.append(data)
import npdb.db_inter

DEFAULT_DPI = 600  # Dots per inch


class Result(object):
    def __init__(self, shape, size, composition, excess_energy, temps):
        self.shape = shape
        self.size = size
        self.composition = composition
        self.bin_index = None
        self.excess_energy = excess_energy

        self.free_energies = dict(zip(temps, list(map(self.get_free_energy_mix, temps))))

    def set_bin_index(self, index):
        assert index is not None
        self.bin_index = index
        assert self.bin_index is not None

    def get_free_energy_mix(self, T):
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
            return self.excess_energy

        # k_b T [eV] = (25.7 mEV at 298 K)
        kt = 25.7E-3 / 298 * T
        del_s = self.composition * np.log(self.composition) + (1 - self.composition) * np.log(1 - self.composition)
        del_s *= -kt

        free_energy = self.excess_energy - del_s
        return free_energy


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
        None. Drops a plot of the alloy into the working directory.
    """

    # Book-keeping and initialization
    shapes = ["icosahedron", "cuboctahedron", "elongated-pentagonal-bipyramid"]
    min_size = size_range[0]
    max_size = size_range[1]
    temperatures = np.arange(temperature_range[0], temperature_range[1] + temperature_res, temperature_res)
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

            # Calculate EE
            excess_energy = result.EE

            # Add to the list of results objects
            results.append(Result(shape, size, composition, excess_energy, temperatures))

    # Bin the results by composition
    bin_indices = np.digitize(x=list(map(lambda i: i.composition, results)),
                              bins=compositions,
                              right=True
                              )
    for result, index in zip(results, bin_indices):
        result.bin_index = index

    # Find the best structure at each temperature and bin
    points = []
    for bin in set(bin_indices):
        for temp in temperatures:
            best_structure = None
            for result in results:
                if result.bin_index == bin:
                    if best_structure is None:
                        best_structure = result
                    elif result.free_energies[temp] < best_structure.free_energies[temp]:
                        best_structure = result
                else:
                    continue
            if best_structure is None:
                continue
            else:
                comp = best_structure.composition
                shape = best_structure.shape
                size = best_structure.size
                energy = best_structure.free_energies[temp]
            point = [temp, comp, shape, size, energy]
            points.append(point)
    return points


def scale_colors(sizes, shapes):
    """
    Scales colors so they aren't all faded out. Returns R/G/B scales.

    Args:
        sizes (list): List of sizes
        shapes(list): List of shapes

    Returns:
        Colormap object
    """
    size_set = sorted(list(set(sizes)))
    linspace = np.linspace(0.2, 1, len(size_set))
    redmap = cm.Reds(linspace)
    greenmap = cm.Greens(linspace)
    bluemap = cm.Blues(linspace)

    colors = []

    for size, shape in zip(sizes, shapes):
        if shape == "icosahedron":
            # Red
            colors.append(redmap[size_set.index(size)])
        elif shape == "elongated-pentagonal-bipyramid":
            # Green
            colors.append(greenmap[size_set.index(size)])
        elif shape == "cuboctahedron":
            #Blue
            colors.append(bluemap[size_set.index(size)])
        else:
            raise ValueError

    return colors


def make_phase(results, colors, alloy):
    """
    Does the plotting and makes the phase diagram

    Args:
        points (list): Result from the get_best function
        colors (list): colors from the scale_colors function

    Returns:
        Nothing
    """
    points = []
    labels = []

    for result in results:
        temp, comp, shape, size, energy = result
        point = [comp, temp]
        points.append(point)
        label = shape + "_" + str(size)
        labels.append(label)

    plt.scatter([x[0] for x in points], [y[1] for y in points], c=colors, marker="s")
    plt.savefig("figure_" + alloy + ".png", dpi=1200)


alloys = ["AgAu", "AgCu", "AuCu"]
for alloy in alloys:
    results = get_best(alloy, [0, 20000], [0, 1000], temperature_res=10)
    colors = scale_colors([result[3] for result in results],
                          [result[2] for result in results])
    make_phase(results, colors, alloy)
