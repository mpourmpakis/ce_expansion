from ga import run_ga

"""
       Main script to run a batch submission job for GA sims
"""


metal_opts = [('Ag', 'Cu'),
              ('Ag', 'Au'),
              ('Au', 'Cu')
              ]

shape_opts = ['icosahedron', 'fcc-cube', 'cuboctahedron',
              'elongated-pentagonal-bipyramid']

# start batch GA run
batch_tot = len(metal_opts) * len(shape_opts)
batch_i = 1
for metals in metal_opts:
    for shape in [shape_opts[0]]:
        run_ga(metals, shape, save_data=True, plotit=False,
               log_results=True,
               batch_runinfo='%i of %i' % (batch_i, batch_tot))
        batch_i += 1
