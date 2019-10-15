#!/usr/bin/env python

import collections
import os
import sys

import matplotlib.colors
import matplotlib.lines
import matplotlib.pyplot as plt
import numpy as np
import sklearn.neighbors

data = os.path.join(os.path.realpath(__file__), '..', '..', '..', 'data', 'larson_et_al')
sys.path.append(data)
import npdb.db_inter

DEFAULT_DPI = 600  # Dots per inch
DEFAULT_POINTSIZE = 15
DEFAULT_MARKER = "o"  # square


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
    def __init__(self, shape, size, composition, excess_energy, cohesive_energy, temps):
        self.shape = shape
        self.size = size
        self.composition = composition
        self.bin_index = None
        self.cohesive_energy = None
        self.excess_energy = excess_energy

        self.free_energies = dict(zip(temps, list(map(self.get_free_energy_mix, temps))))

    def get_free_energy_mix(self, T):
        """
        Calculates Excess energies plus an entropic contribution.
        :param T: Temperature

        :return: Free energy of mixing = excess energy (related to enthalpy of mixing) - entropy of mixing
        """

        if self.composition == 1 or self.composition == 0:
            return self.cohesive_energy

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
    :param alloy: Alloy of interest
    :param size_range: Range of sizes to consider, in atoms, inclusive
    :param temperature_range: Range of temperatures to consider, in K, inclusive
    :param temperature_res: How fine our temperature mesh is, in K, defaults to 1K
    :return: A dictionary with {size : {temp : Result_object() } }
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
            print(result)
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
            cohesive_energy = result.CE
            # Add to the list of results objects
            results.append(Result(shape, size, composition, excess_energy, cohesive_energy, temperatures))

    # Initialize lists
    chosen_dict = {}
    for temp in temperatures:
        # List comprehension forces construction of new objects, instead of references to the same object
        chosen = [[Point() for i in range(len(sizes))] for i in range(len(comps))]
        chosen_dict[temp] = chosen.copy()

    # Choose minimum-energy objects to plot
    for result in results:
        print(result)
        for temp in result.free_energies:
            compared = chosen_dict[temp][comps.index(result.composition)][sizes.index(result.size)]
            if compared.energy is None:
                newpoint = Point()
                newpoint.size = result.size
                newpoint.comp = result.composition
                newpoint.temp = temp
                newpoint.shape = result.shape
                if newpoint.comp in [0, 1]:
                    newpoint.energy = result.cohesive_energy
                else:
                    newpoint.energy = result.free_energies[temp]
                # Choose color
                if newpoint.shape == "icosahedron":
                    newpoint.color = "#FF0000"
                elif newpoint.shape == "elongated-pentagonal-bipyramid":
                    newpoint.color = "#00FF00"
                elif newpoint.shape == "cuboctahedron":
                    newpoint.color = "#0000FF"
                else:
                    raise ValueError(newpoint.shape)
                assert newpoint.comp is not None
                chosen_dict[temp][comps.index(result.composition)][sizes.index(result.size)] = newpoint

            elif (compared.comp == 0 or compared.comp == 1) and (compared.energy > result.cohesive_energy):
                newpoint = Point()
                newpoint.size = result.size
                newpoint.comp = result.composition
                newpoint.temp = temp
                newpoint.shape = result.shape
                newpoint.energy = result.free_energies[temp]
                # Choose color
                if newpoint.shape == "icosahedron":
                    newpoint.color = "#FF0000"
                elif newpoint.shape == "elongated-pentagonal-bipyramid":
                    newpoint.color = "#00FF00"
                elif newpoint.shape == "cuboctahedron":
                    newpoint.color = "#0000FF"
                else:
                    raise ValueError(newpoint.shape)
                assert newpoint.comp is not None
                chosen_dict[temp][comps.index(result.composition)][sizes.index(result.size)] = newpoint

            elif compared.energy > result.free_energies[temp]:
                newpoint = Point()
                newpoint.size = result.size
                newpoint.comp = result.composition
                newpoint.temp = temp
                newpoint.shape = result.shape
                newpoint.energy = result.free_energies[temp]
                # Choose color
                if newpoint.shape == "icosahedron":
                    newpoint.color = "#FF0000"
                elif newpoint.shape == "elongated-pentagonal-bipyramid":
                    newpoint.color = "#00FF00"
                elif newpoint.shape == "cuboctahedron":
                    newpoint.color = "#0000FF"
                else:
                    raise ValueError(newpoint.shape)
                assert newpoint.comp is not None
                chosen_dict[temp][comps.index(result.composition)][sizes.index(result.size)] = newpoint

    return chosen_dict


def make_unfilled(points, alloy):
    """
    Does the actual plotting. Makes several Size/Comp phase diagrams varying temperature.

    :param points: Dict with temps as keys, values are lists of point objects.
    :param alloy: Alloy of interest
    """
    for temp in points:
        fig, ax = plt.subplots()

        # Add points
        for composition in points[temp]:
            for point in composition:
                if None in [point.comp, point.size, point.energy] or point.comp == 0 or point.comp==1:
                    pass
                ax.scatter(point.comp, point.size, c=point.color, marker=point.symbol, s=point.symbol_size)

        # Create legend
        legend_elements = [matplotlib.lines.Line2D([0], [0], marker="o", color="#FF0000", lw=0),
                           matplotlib.lines.Line2D([0], [0], marker="o", color="#00FF00", lw=0),
                           matplotlib.lines.Line2D([0], [0], marker="o", color="#0000FF", lw=0)]

        # Book keeping
        ax.legend(legend_elements, ["Ico", "EPB", "Cuboct"])
        ax.set_title(alloy + ", " + str(temp) + " K")
        ax.set_xlabel("Composition (% " + alloy[0:2] + ")")
        ax.set_ylabel("Size (N Atoms)")

        # Save figure
        plt.savefig("unfilled/size_comp_" + alloy + "_" + str(temp) + "K_nocolor.png")
        plt.close()

def make_filled(points, alloy, resolution=10):
    """
    Does the actual plotting. Makes several Size/Comp phase diagrams varying temperature. Fills with color through
    1st-nearest-neighbor interpolation

    :param points: Dict with temps as keys, values are lists of point objects.
    :param alloy: Alloy of interest
    :param resolution: Grid spacing, defaults to 10
    """
    for temp in points:
        # Get known observations
        sizes = OrderedSet()
        comps = OrderedSet()
        observations = []
        colors = []
        for composition in points[temp]:
            for point in composition:
                if None in [point.comp, point.size] or point.comp == 0 or point.comp == 1:
                    continue
                sizes.add(point.size)
                comps.add(point.comp)
                observations.append([point.comp, point.size])
                mapping = {"#FF0000": 0,
                           "#00FF00": 1,
                           "#0000FF": 2}
                colors.append(mapping[point.color])
        observations = np.vstack(observations)


        # KNN
        knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=1, weights="uniform", algorithm="auto")
        knn.fit(observations, colors)

        xmin, xmax = min(comps)-1, max(comps)+1
        ymin, ymax = min(sizes)-1, max(sizes)+1
        xx, yy = np.meshgrid(np.arange(0, 1, (xmax-xmin) / resolution),
                             np.arange(ymin, ymax, (ymax-ymin) / resolution))
        z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
        z = z.reshape(xx.shape)

        # Plotting
        fig, ax = plt.subplots()
        cmap_light = matplotlib.colors.ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
        cmap_bold = matplotlib.colors.ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
        data_range = matplotlib.colors.Normalize(vmin=0, vmax=2)

        ax.pcolormesh(xx, yy, z, cmap=cmap_light, norm=data_range)
        ax.scatter(observations[:,0], observations[:,1], c=colors, cmap=cmap_bold, norm=data_range)
        ax.set_title(alloy + ", " + str(temp) + " K")
        ax.set_xlabel("Composition (% " + alloy[0:2] + ")")
        ax.set_ylabel("Size (N Atoms)")

        # Legend
        custom_legend = [matplotlib.lines.Line2D([0],[0], color=cmap_bold(0), marker="o", lw=0),
                         matplotlib.lines.Line2D([0],[0], color=cmap_bold(1), marker="o", lw=0),
                         matplotlib.lines.Line2D([0],[0], color=cmap_bold(2), marker="o", lw=0)]
        #ax.legend(custom_legend, ["Ico", "EPB", "Cuboct"], fancybox=True, framealpha=0.5)
        plt.tight_layout()
        plt.ylim(13,3871)
        plt.savefig("size_comp_" + alloy + "_" + str(temp) + "K.png")
        plt.close()

alloys = ["AgAu" , "AgCu", "AuCu"]
for alloy in alloys:
    print(alloy)
    results = get_best(alloy, [0, 4000], [0, 1000], temperature_res=250)
    make_filled(results, alloy, resolution=1000)
