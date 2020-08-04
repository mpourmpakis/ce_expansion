import ce_expansion.plots

systems = ["AgCu", "AgAu", "AuCu"]
system_colors = ["red", "green", "blue"]

morphologies = ["icosahedron", "cuboctahedron", "elongated-pentagonal-bipyramid"]
morphology_markers = ["h", "s", "^"]

sizes = [13, 55, 147, 309, 561, 923, 1415, 2057, 2869, 3871]
point_sizes = range(1, len(sizes) + 1)

ce_expansion.plots.plot_bond_types_3D(systems, system_colors, morphologies, morphology_markers, sizes, point_sizes, verbose=True)
ce_expansion.plots.plot_bond_types_3D(systems, system_colors, morphologies, morphology_markers, sizes, point_sizes, verbose=False,
                         scale=True)
