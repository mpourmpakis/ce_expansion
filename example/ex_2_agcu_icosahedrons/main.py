import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

import npdb.db_inter

# Define the colors
sizes = [309, 561, 923, 1415, 2057, 2869, 5083]
colors = cm.rainbow(np.linspace(0, 1, len(sizes)))

for size, coloration in zip(sizes, colors):
    # Perform the database query
    query = npdb.db_inter.get_bimet_result(metals="AgCu", num_atoms=size)

    query = sorted(query, key=lambda i: i.n_metal2)

    # Calculate the copper content and pull excess energy
    cu_content = list(map(lambda i: i.n_metal2 / size, query))
    excess_energy = list(map(lambda i: i.EE, query))

    # Make the plot
    plt.plot(cu_content, excess_energy,
             color=coloration,
             label=str(size))

plt.legend(loc="lower left", fontsize=20)
plt.savefig("Excess_Energies.png", dpi=1200)
