import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

import npdb.db_inter

DEFAULT_DPI = 600  # Dots per inch
DEFAULT_SCALE = 8  # Scale in inches

# Define the colors
sizes = [13, 55, 147, 309, 561, 923, 1415, 2057, 2869, 5083]
colors = cm.rainbow(np.linspace(0, 1, len(sizes)))

# Set up the state machine

alloys = ["AgCu", "AgAu", "AuCu"]
symbols = (("icosahedron", "solid", "Ico"),
           ("cuboctahedron", "dashed", "Cub"),
           ("elongated-pentagonal-bipyramid", "dotted", "EPB"))
for alloy in alloys:
    for shape, style, lbl in symbols:
        fig = plt.figure()
        fig.set_size_inches(2 * DEFAULT_SCALE, DEFAULT_SCALE)
        ax = plt.subplot(111)
        for size, coloration in zip(sizes, colors):
            # Perform the database query
            query = npdb.db_inter.get_bimet_result(metals=alloy, shape=shape, num_atoms=size)
            query = sorted(query, key=lambda i: i.n_metal2)

            # Calculate the copper content and pull excess energy
            cu_content = list(map(lambda i: i.n_metal2 / size, query))
            excess_energy = list(map(lambda i: i.EE, query))

            # Make the plot
            ax.plot(cu_content, excess_energy,
                    color=coloration,
                    label=str(size) + "_" + lbl,
                    linestyle=style)

        dopant = alloy[2:]
        plt.title(alloy + "_" + lbl)
        plt.xlabel("% " + dopant)
        plt.ylabel("Excess Energy (eV)")
        plt.tight_layout()
        chartbox = ax.get_position()
        ax.set_position([chartbox.x0, chartbox.y0, chartbox.width * 0.6, chartbox.height])
        ax.legend(loc="upper left", fontsize=20, bbox_to_anchor=(0.98, 0.8), ncol=1)
        plt.savefig("Excess_Energies_" + alloy + "_" + lbl + ".png", dpi=DEFAULT_DPI)
        plt.close()

    # Get cubes
    shape = "fcc-cube"
    lbl = "cube"
    style = "dashdot"

    sizes = [13, 63, 171, 365, 665, 1099, 1687, 2457, 3429, 4631, 6083, 7813, 9841, 12195]
    colors = cm.rainbow(np.linspace(0, 1, len(sizes)))

    fig = plt.figure()
    fig.set_size_inches(2 * DEFAULT_SCALE, DEFAULT_SCALE)
    ax = plt.subplot(111)
    for size, coloration in zip(sizes, colors):
        # Perform the database query
        query = npdb.db_inter.get_bimet_result(metals=alloy, shape=shape, num_atoms=size)
        query = sorted(query, key=lambda i: i.n_metal2)

        # Calculate the copper content and pull excess energy
        cu_content = list(map(lambda i: i.n_metal2 / size, query))
        excess_energy = list(map(lambda i: i.EE, query))

        # Make the plot
        ax.plot(cu_content, excess_energy,
                color=coloration,
                label=str(size) + "_" + lbl,
                linestyle=style)

    plt.xlabel("% " + dopant)
    plt.title(alloy + "_" + lbl)
    plt.ylabel("Excess Energy (eV)")
    plt.tight_layout()
    chartbox = ax.get_position()
    ax.set_position([chartbox.x0, chartbox.y0, chartbox.width * 0.6, chartbox.height])
    ax.legend(loc="upper left", fontsize=20, bbox_to_anchor=(0.98, 0.8), ncol=1)
    plt.savefig("Excess_Energies_" + alloy + "_" + lbl + ".png", dpi=DEFAULT_DPI)
    plt.close
