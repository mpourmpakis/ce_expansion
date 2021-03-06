import os
import sys
import time
from datetime import datetime

from ce_expansion.ga import ga
from ce_expansion.npdb import db_inter


def run_ga():
    sys.path.append(os.path.dirname(os.path.realpath(__file__)))

    """
           Main script to run a batch submission job for GA sims
           - runs multiple ga sims sweeping size, shape, and composition
           - creates/updates "new structures found" summary plot based on sims.log
    
           NOTE: Currently runs 4 metal combinations and 5 shapes over 1 iteration
                 - should take a little over 17 hours
    """
    home = os.path.expanduser('~')
    datapath = os.path.join('D:\\MCowan', 'Box Sync',
                            'Michael_Cowan_PhD_research', 'data', 'np_ce')

    # all metal options
    # 28 total options
    # metals = list(itertools.combinations(db_inter.build_metals_list(), 2))

    min_generations = 500
    max_generations = -1

    max_nochange = 500
    spike = False

    # HOW MANY TIMES THE TOTAL BATCH RUN SHOULD REPEAT
    niterations = 3

    # run 4 metal options a day to iterate once a week through all options
    # e.g. (Saturday = 5) * 4 = 20
    day = datetime.now().weekday()
    start_index = day * 4

    # chooses 4 metals from list of 28
    # metal_opts = metals[start_index:start_index + 4]

    # ONLY RUN METALS THAT WE WILL FOCUS ON FOR PAPER
    metal_opts = [('Ag', 'Au'), ('Ag', 'Cu'), ('Au', 'Cu')][::-1]

    shape_opts = ['icosahedron', 'fcc-cube', 'cuboctahedron',
                  'elongated-pentagonal-bipyramid'][::-1]

    # run one metal combination if int arg (0, 1, or 2) is passed in
    if len(sys.argv) > 1:
        i = int(sys.argv[1])
        if -1 < i < 3:
            metal_opts = [metal_opts[i]]

    # create text file on desktop to indicate GA is running
    running = os.path.join(home, 'Desktop', 'RUNNING-GA.txt')
    with open(running, 'w') as fid:
        fid.write('GA is currently running...hopefully')

    start = time.time()
    startstr = datetime.now().strftime('%Y-%m-%d %H:%M %p')

    # start batch GA run
    batch_tot = len(metal_opts) * len(shape_opts)
    for n in range(niterations):
        batch_i = 1
        for metals in metal_opts:
            for shape in shape_opts:
                ga.run_ga(metals=metals,
                          shape=shape,
                          save_data=True,  # True,
                          batch_runinfo='%i of %i' % (batch_i, batch_tot),
                          max_generations=max_generations,
                          min_generations=min_generations,
                          max_nochange=max_nochange,
                          spike=spike)
                with open(running, 'a') as fid:
                    fid.write('\ncompleted %i of %i' % (batch_i, batch_tot))
                batch_i += 1

    # update new structures plot in <datapath>
    cutoff_date = datetime(2019, 4, 24, 18, 30)
    fig = db_inter.build_new_structs_plot(metal_opts, shape_opts, pct=False,
                                          cutoff_date=cutoff_date)
    # fig.savefig(os.path.join(datapath, '%02i_new_struct_log.png' % day))
    fig.savefig(os.path.join(datapath, 'agaucu_STRUCTS.png'))
    os.remove(running)

    runtime = (time.time() - start) / 3600.
    with open(running.replace('RUNNING', 'COMPLETED'), 'w') as fid:
        fid.write('completed batch GA in %.2f hours.' % runtime)
        fid.write('\nstarted: ' + startstr)
        fid.write('\n  ended: ' + datetime.now().strftime('%Y-%m-%d %H:%M %p'))
