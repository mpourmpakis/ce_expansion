import ga
import os

"""
       Main script to run a batch submission job for GA sims
       - runs multiple ga sims sweeping size, shape, and composition
"""

datapath = os.path.join(os.path.expanduser('~'), 'Box Sync',
                        'Michael_Cowan_PhD_research', 'data', 'np_ce')

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
    for shape in shape_opts:
        ga.run_ga(metals, shape,
                  datapath=datapath,
                  save_data=True,
                  plotit=False,
                  log_results=True,
                  batch_runinfo='%i of %i' % (batch_i, batch_tot))
        batch_i += 1
