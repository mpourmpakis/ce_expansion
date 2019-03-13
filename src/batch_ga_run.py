import ga
import os
import numpy as np
import matplotlib.pyplot as plt
from npdb import db_inter

"""
       Main script to run a batch submission job for GA sims
       - runs multiple ga sims sweeping size, shape, and composition
       - creates/updates "new structures found" summary plot based on sims.log
"""

datapath = os.path.join(os.path.expanduser('~'), 'Box Sync',
                        'Michael_Cowan_PhD_research', 'data', 'np_ce')

# all metal options
metals = db_inter.build_metals_list()

# Current metal combinations being studied
metal_opts = [('Ag', 'Cu'),
              ('Ag', 'Au'),
              ('Au', 'Cu')
              ]

shape_opts = ['icosahedron', 'fcc-cube', 'cuboctahedron',
              'elongated-pentagonal-bipyramid']

# HOW MANY TIMES THE TOTAL BATCH RUN SHOULD REPEAT
niterations = 2

# start batch GA run
batch_tot = len(metal_opts) * len(shape_opts)
for n in range(niterations):
    batch_i = 1
    for metals in metal_opts:
        for shape in shape_opts:
            ga.run_ga(metals=metals,
                      shape=shape,
                      save_data=True,  # True,
                      batch_runinfo='%i of %i' % (batch_i, batch_tot))
            batch_i += 1

# update new structures plot in <datapath>
fig = db_inter.build_new_structs_plot(metal_opts, shape_opts)
fig.savefig(os.path.join(datapath, 'new_struct_summary.png'))
