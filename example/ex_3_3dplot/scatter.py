import numpy as np

import plots

systems = ["AgCu", "AgAu", "AuCu"]
system_colors = ["#FF5555", "#55FF55", "#5555FF"]

morphologies = ["icosahedron", "cuboctahedron", "elongated-pentagonal-bipyramid"]
morphology_markers = ["h", "s", "^"]

sizes = [147]#[13, 55, 147, 309, 561, 923, 1415, 2057, 2869, 3871]
minsize = 5
maxsize = minsize + len(sizes) + 30
point_sizes = np.linspace(minsize, maxsize, len(sizes))

plots.plot_bond_types_2D(systems, system_colors, morphologies, morphology_markers, sizes, point_sizes,
                         projection=["AA", "AB"], verbose=True, scale=True)

# for i in [True, False]:
#     plots.plot_bond_types_2D(systems, system_colors, morphologies, morphology_markers, sizes, point_sizes,
#                              projection=["AA", "AB"], verbose=True, scale=i)
#     plots.plot_bond_types_2D(systems, system_colors, morphologies, morphology_markers, sizes, point_sizes,
#                              projection=["BB", "AB"], verbose=True, scale=i)
#     plots.plot_bond_types_2D(systems, system_colors, morphologies, morphology_markers, sizes, point_sizes,
#                              projection=["AA", "BB"], verbose=True, scale=i)
