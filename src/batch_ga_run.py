import ga
import os
import numpy as np
import matplotlib.pyplot as plt

"""
       Main script to run a batch submission job for GA sims
       - runs multiple ga sims sweeping size, shape, and composition
       - creates/updates "new structures found" summary plot based on sims.log
"""

datapath = os.path.join(os.path.expanduser('~'), 'Box Sync',
                        'Michael_Cowan_PhD_research', 'data', 'np_ce')

metal_opts = [('Ag', 'Cu'),
              ('Ag', 'Au'),
              ('Au', 'Cu')
              ]

shape_opts = ['icosahedron', 'fcc-cube', 'cuboctahedron',
              'elongated-pentagonal-bipyramid']

# log batch GA results
log = True

# HOW MANY TIMES THE TOTAL BATCH RUN SHOULD REPEAT
niterations = 2

# start batch GA run
batch_tot = len(metal_opts) * len(shape_opts)
for n in range(niterations):
    batch_i = 1
    for metals in metal_opts:
        for shape in shape_opts:
            ga.run_ga(metals, shape,
                      datapath=datapath,
                      save_data=True,  # True,
                      plotit=False,
                      log_results=log,
                      add_current_min=True,
                      batch_runinfo='%i of %i' % (batch_i, batch_tot))
            batch_i += 1

# if results were logged, update new structures plot in <datapath>
if log:
    struct_info = {}
    z = 0
    for metals in metal_opts:
        for shape in ['Icos', 'FCC-Cube', 'Cuboct', 'J16']:
            struct_info[z] = '%s: %s' % (''.join(metals), shape)
            z += 1

    cs = []
    pcts = []
    with open(os.path.join(datapath, 'sims.log'), 'r') as fidr:
        for i in fidr:
            if 'New Min Structs' in i:
                cs.append(int(i.split()[-1].strip('\n')))
            elif '% New Structs' in i:
                pcts.append(float(i.split()[-1].strip('\n')[:-1]) / 100)

    cs = np.array(cs).reshape(int(len(cs) / z), z).T
    pcts = np.array(pcts).reshape(int(len(pcts) / z), z).T

    x = range(1, cs.shape[1] + 1)

    f, a = plt.subplots(2, 1, sharex=True, figsize=(7, 7))
    a1, a2 = a
    for i in range(len(cs)):
        colors = plt.cm.tab20c((i / float(z)) * 0.62)
        a1.plot(x, cs[i], '.-', linewidth=1, color=colors,
                label=struct_info[i])
        a2.plot(x, pcts[i], '.-', linewidth=1, color=colors)

    # change ax2's y-axis to percent
    vals = a2.get_yticks()
    a2.set_yticklabels(['{:,.0%}'.format(x) for x in vals])

    a1.legend(ncol=3)
    a1.set_title('New Minimum Structures Found')
    a1.set_ylabel('Total Count')
    a2.set_ylabel('Percentage')
    a2.set_xlabel('Batch Run #')
    f.tight_layout()
    f.savefig(os.path.join(datapath, 'new_struct_summary.png'))
    # f.savefig(os.path.join(datapath, 'new_struct_summary.svg'))
