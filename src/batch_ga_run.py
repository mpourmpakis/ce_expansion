import ga
import os
import itertools
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from npdb import db_inter

"""
       Main script to run a batch submission job for GA sims
       - runs multiple ga sims sweeping size, shape, and composition
       - creates/updates "new structures found" summary plot based on sims.log

       NOTE: Currently runs 4 metal combinations and 5 shapes over 1 iteration
             - should take a little over 17 hours
"""

datapath = os.path.join(os.path.expanduser('~'), 'Box Sync',
                        'Michael_Cowan_PhD_research', 'data', 'np_ce')

# all metal options
# 28 total options
metals = list(itertools.combinations(db_inter.build_metals_list(), 2))


# run 4 metal options a day to iterate once a week through all options
# e.g. (Saturday = 5) * 4 = 20
day = datetime.now().weekday()
start_index = day * 4

# chooses 4 metals from list of 28
metal_opts = metals[start_index:start_index + 4]

shape_opts = ['icosahedron', 'fcc-cube', 'cuboctahedron',
              'elongated-pentagonal-bipyramid']

# HOW MANY TIMES THE TOTAL BATCH RUN SHOULD REPEAT
niterations = 1

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
fig.savefig(os.path.join(datapath, '%02i_new_struct_log.png' % day))
