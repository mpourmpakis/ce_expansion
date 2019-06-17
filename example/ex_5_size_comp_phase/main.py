#!/usr/bin/env python

import collections
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

data = os.path.join(os.path.realpath(__file__), '..', '..', '..', 'data', 'larson_et_al')
sys.path.append(data)
import npdb.db_inter

DEFAULT_DPI = 600  # Dots per inch


class Point(object):
    def __init__(self):
        self.comp = None
        self.size = None
        self.temp = None
        self.shape = None
        self.color = None
        self.symbol = "s"
        self.symbol_size = 10
        self.energy = None


class Result(object):
    def __init__(self, shape, size, composition, excess_energy, temps):
        self.shape = shape
        self.size = size
        self.composition = composition
        self.bin_index = None
        self.excess_energy = excess_energy

        self.free_energies = dict(zip(temps, list(map(self.get_free_energy_mix, temps))))

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


class OrderedSet(collections.UserList):
    """
    Wrapper around a list that allows it to operate somewhat like a set.
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
        None. Drops a plot of the alloy into the working directory.
    """

    # Book-keeping and initialization
    shapes = ["icosahedron", "cuboctahedron", "elongated-pentagonal-bipyramid"]
    min_size = size_range[0]
    max_size = size_range[1]
    temperatures = np.arange(temperature_range[0], temperature_range[1] + temperature_res, temperature_res)

    sizes = OrderedSet()
    comps = OrderedSet()

    # DB Query
    results = []
    for shape in shapes:
        query = npdb.db_inter.get_bimet_result(metals=alloy, shape=shape)
        for result in query:
            # Calculate size
            size = result.num_atoms
            if size <= min_size or size >= max_size:
                continue
            sizes.add(size)
            # Calculate composition
            composition = np.round(result.n_metal1 / result.num_atoms, decimals=2)
            comps.add(composition)
            # Calculate EE
            excess_energy = result.EE
            # Add to the list of results objects
            results.append(Result(shape, size, composition, excess_energy, temperatures))

    # Initialize lists
    chosen_dict = {}
    for temp in temperatures:
        # List comprehension forces construction of new objects, instead of references to the same object
        chosen = [[Point() for i in range(len(sizes))] for i in range(len(comps))]
        chosen_dict[temp] = chosen.copy()

    # Choose minimum-energy objects to plot
    for result in results:
        for temp in result.free_energies:
            compared = chosen_dict[temp][comps.index(result.composition)][sizes.index(result.size)]
            if (compared.energy is None) or (compared.energy > result.free_energies[temp]):
                newpoint = Point()
                newpoint.size = result.size
                newpoint.comp = result.composition
                newpoint.temp = temp
                newpoint.shape = result.shape
                newpoint.energy = result.free_energies[temp]
                # Choose color
                if newpoint.shape == "icosahedron":
                    newpoint.color = "red"
                elif newpoint.shape == "elongated-pentagonal-bipyramid":
                    newpoint.color = "green"
                elif newpoint.shape == "cuboctahedron":
                    newpoint.color = "blue"

                else:
                    raise ValueError(newpoint.shape)

                chosen_dict[temp][comps.index(result.composition)][sizes.index(result.size)] = newpoint

    return chosen_dict


def make_phase(points, alloy):
    """
    Does the actual plotting. Makes several Size/Comp phase diagrams varying temperature.

    :param points: Dict with temps as keys, values are lists of point objects.
    """
    for temp in points:
        plt.figure()
        for composition in points[temp]:
            for point in composition:
                plt.scatter(point.comp, point.size, c=point.color, marker=point.symbol, s=point.symbol_size)
        plt.title(alloy + ", " + str(temp) + " K")
        plt.xlabel("Composition (%)")
        plt.ylabel("Size (N Atoms)")
        plt.show()


alloys = ["AgAu"]  # , "AgCu", "AuCu"]
for alloy in alloys:
    results = get_best(alloy, [0, 1000], [0, 1000], temperature_res=500)
    make_phase(results, alloy)
