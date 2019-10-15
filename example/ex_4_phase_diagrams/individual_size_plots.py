import collections
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as tick
import numpy as np

data = os.path.join(os.path.realpath(__file__), '..', '..', '..', 'data', 'larson_et_al')
sys.path.append(data)
import npdb.db_inter

DEFAULT_DPI = 600  # Dots per inch
DEFAULT_POINTSIZE = 15
DEFAULT_MARKER = "o"  # square


class Result(object):
    def __init__(self, shape, size, composition, excess_energy, temp):
        self.shape = shape
        self.size = size
        self.composition = composition
        self.excess_energy = excess_energy
        self.free_energy = self.get_free_energy_mix(temp)

    def get_free_energy_mix(self, T):
        """
        Calculates Excess energies plus an entropic contribution.

        :param excess_energy: Excess energies from DB query
        :param comp: Compositions from DB query
        :param T: Temperature

        :return: Free energy of mixing = excess energy (related to enthalpy of mixing) - entropy of mixing
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


def get_data(alloy,
             size,
             temperature):
    """
    Gets data for phase diagram

    :param alloy: Alloy of interest
    :param size: Size to consider
    :param temperature: Temperature to use
    :return: results object.
    """

    # Book-keeping and initialization
    shapes = ["icosahedron", "cuboctahedron", "elongated-pentagonal-bipyramid"]

    # DB Query
    results = []
    for shape in shapes:
        query = npdb.db_inter.get_bimet_result(metals=alloy, shape=shape, num_atoms=size)
        for result in query:
            # Calculate composition
            composition = result.n_metal1 / result.num_atoms
            # Calculate EE
            excess_energy = result.EE
            # Add to the list of results objects
            results.append(Result(shape, size, composition, excess_energy, temperature))
    return results


def make_plot(results, axis, size):
    """
    Plots some results, y'know?

    :param results: A list of Results objects containing the shape, composition, and free energy of mixing
    :param axis: Pyplot axis to plot to
    :param size: size
    :return: None. Drops the plot in the working directory.
    """
    # Split into 3 lists, for icosahedrons, cubs, and epbs
    # Each list is of the format (composition, free energy of mixing)
    icos = []
    cubs = []
    epbs = []
    types = {"icosahedron": icos,
             "cuboctahedron": cubs,
             "elongated-pentagonal-bipyramid": epbs}

    colors = {"icosahedron": "red",
              "cuboctahedron": "blue",
              "elongated-pentagonal-bipyramid": "green"}
    for result in results:
        types[result.shape].append((result.composition, result.free_energy, colors[result.shape]))

    for shape in [icos, cubs, epbs]:
        x = [i[0] * 100 for i in shape]
        y = [i[1] for i in shape]
        color = shape[0][2]
        axis.plot(x, y, color)

        # Label size
        axis.text(0.9, 0.5, f"N={size}", transform=axis.transAxes, size=20)


alloys = ["AgCu"]#["AgAu", "AuCu", "AgCu"]
for alloy in alloys:
    tens_sizes = [3871, 2869, 2057, 1415, 561]  # sizes where we skipped 10% increments
    all_sizes = [309, 147, 55, 13]  # sizes where we looked at all possible compositions

    for sizes in [tens_sizes, all_sizes]:
        fig, axes = plt.subplots(nrows=5, ncols=1, sharex=True, sharey=True)
        ymin = 0
        ymax = 0
        for plot_index, size in enumerate(sizes):
            # Query
            results = get_data(alloy, size, 250)
            results.sort(key=lambda i: i.composition)

            # Plot
            make_plot(results, axes[abs(plot_index)], size)

        # plot labels
        fig.text(0.5, 0.04, "Composition (%)", ha="center", size=20)
        fig.text(0, 0.5, "Free Energy of Mixing (eV/atom)", va="center", rotation="vertical", size=20)
        fig.text(0.5, 0.95, f"{alloy} @ 250K", size=25, ha="center")

        # Tickmarks
        plt.xlim(0, 100)
        ylimits = {"AgAu": [-0.1, 0],
                   "AgCu": [-0.1+0.025, 0.025],
                   "AuCu": [-0.3, 0]}

        ymin = ylimits[alloy][0]
        ymax = ylimits[alloy][1]

        plt.ylim(ymin, ymax)
        for axis in axes:
            # Set up X tickmarks
            axis.tick_params(axis="x", labelsize=15)
            axis.xaxis.set_major_locator(tick.MultipleLocator(20))
            axis.xaxis.set_major_formatter(tick.FormatStrFormatter("%d"))
            axis.xaxis.set_minor_locator(tick.MultipleLocator(10))
            axis.xaxis.grid(True, which='major')

            # Set up Y tickmarks
            axis.tick_params(axis="y", labelsize=15)
            axis.yaxis.set_major_locator(tick.MultipleLocator((ymax - ymin) / 2))
            axis.yaxis.set_major_formatter(tick.FormatStrFormatter("%2.2f"))
            axis.yaxis.set_minor_locator(tick.MultipleLocator((ymax - ymin) / 4))

        # Save and quit
        plt.savefig(f"{alloy},{sizes[-1]}-{sizes[0]}.png")
        plt.close()
