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


def batch_run_ga_from_ga_module__needs_fixing(
        metals,
        shape,
        save_data=True,
        batch_runinfo=None, shells=None,
        max_generations=5000,
        min_generations=-1,
        max_nochange=2000,
        add_coreshell=True,
        **kwargs):
    """
    Submission function to run GAs of a given metal combination and
    shape, sweeping over different sizes (measured in number of shells)
    - capable of saving minimum structures as XYZs, logging GA stats, saving
      all run info as excel, and creating a 3D surface plot of results

    Args:
    - metals (iterator): list of two metals used in the bimetallic NP
    - shape (str): shape of NP that is being studied
                   NOTE: currently supports
                          - icosahedron
                          - cuboctahedron
                          - fcc-cube
                          - elongated-trigonal-pyramic

    KArgs:
    - plotit (bool): if true, a 3D surface plot is made of GA sims
                     dope concentration vs. size vs. excess energy
                     (Default: False)
    - save_data (bool): - if true, GA sim data is saved to BimetallicResults
                          table in database
                        (Default: True)
    - batch_runinfo (str): if str, add to BimetallicLog entry
                           (Default: None)
    - shells (int || list): if int, only that shell size is simulated
                            elif list of ints, nshell_range = shells
                            (Default: None)
    - max_generations (int): if not -1, use specified value as max
                             generations for each GA sim
                             (Default: 5000)
    - min_generations (int): if not -1, use specified value as min
                             generations that run before max_nochange
                             criteria is applied
                             (Default: -1)
    - max_nochange (int): maximum generations GA will go without a change in
                          minimum CE
                          (Default: 2000)
    - add_coreshell (bool): if True, core shell structures will be included in
                            GA simulations
                            (Default: True)

    Returns: None
    """

    # track start of run
    start_time = dt.now()

    # clear previous plots and define desktop and data paths
    plt.close('all')
    desk = os.path.join(os.path.expanduser('~'), 'desktop')

    # number of shells range to sim ga for each shape
    default_shell_range = [1, 11]
    shape2shell = {'cuboctahedron': default_shell_range,
                   'elongated-pentagonal-bipyramid': default_shell_range,
                   'fcc-cube': default_shell_range,
                   'icosahedron': default_shell_range
                   }
    nshell_range = shape2shell[shape]

    if shells:
        if isinstance(shells, int):
            nshell_range = [shells, shells + 1]
        else:
            nshell_range = shells
    nstructs = len(range(*nshell_range))
    if not nstructs:
        print(nshell_range)
        return

    # print run info
    print('')
    print('RUN INFO'.center(CENTER, '-'))
    print('             Metals: %s' % (' '.join(metals_types)))
    print('              Shape: %s' % shape)
    print('    Save GA Results: %s' % bool(save_data))
    print('        Shell Range: %s' % str(nshell_range))
    print('-' * CENTER)

    # keep track of total new minimum CE structs (based on saved structs)
    tot_new_structs = 0

    # count total structs
    tot_structs = 0

    for struct_i, nshells in enumerate(range(*nshell_range)):
        # build atom, adjacency list, and AtomGraph
        nanop = structure_gen.build_structure_sql(shape, nshells,
                                                  build_bonds_arr=True)
        num_atoms = len(nanop)

        diameter = nanop.get_diameter()

        BCModel(nanop.atoms, nanop.bonds_list, metal_types)
        ag = BCModel(nanop.bonds_list, metal1, metal2)

        # check to see if monometallic results exist
        # if not, calculate them
        mono1 = db_inter.get_bimet_result(metals=metals,
                                          shape=shape,
                                          num_atoms=num_atoms,
                                          n_metal1=num_atoms,
                                          lim=1)
        if not mono1:
            mono_ord = np.zeros(num_atoms)
            mono_ce1 = ag.getTotalCE(mono_ord)
            mono1 = db_inter.update_bimet_result(
                metals=metals,
                shape=shape,
                num_atoms=num_atoms,
                diameter=diameter,
                n_metal1=num_atoms,
                CE=mono_ce1,
                ordering=''.join(str(int(i)) for i in mono_ord),
                EE=0,
                nanop=nanop,
                allow_insert=True)

        mono2 = db_inter.get_bimet_result(metals=metals,
                                          shape=shape,
                                          num_atoms=num_atoms,
                                          n_metal1=0,
                                          lim=1)
        if not mono2:
            mono_ord = np.ones(num_atoms)
            mono_ce2 = ag.getTotalCE(mono_ord)
            mono2 = db_inter.update_bimet_result(
                metals=metals,
                shape=shape,
                num_atoms=num_atoms,
                diameter=diameter,
                n_metal1=0,
                CE=mono_ce2,
                ordering=''.join(str(int(i)) for i in mono_ord),
                EE=0,
                nanop=nanop,
                allow_insert=True)

        # USE THIS TO TEST EVERY CONCENTRATION
        if nanop.num_atoms < 366:
            n = np.arange(0, num_atoms + 1)
            x = n / float(num_atoms)
        else:
            # x = metal2 concentration [0, 1]
            x = np.linspace(0, 1, 11)
            n = (x * num_atoms).astype(int)

            # recalc concentration to match n
            x = n / float(num_atoms)

        # add core-shell structures list of comps to run
        if add_coreshell:
            srfatoms = db_inter.build_atoms_in_shell_dict(shape, nshells)
            nsrf = len(srfatoms)
            ncore = num_atoms - nsrf
            n = np.unique(n.tolist() + [ncore, nsrf])
            x = n / float(num_atoms)

        # total structures checked ( - 2 to exclude monometallics)
        tot_structs += float(len(n) - 2)

        starting_outp = '%s%s in %i atom %s' % (metal1, metal2,
                                                num_atoms, shape)
        print(starting_outp.center(CENTER))

        # track min structures for each size
        new_min_structs = 0

        # sweep over different compositions
        for i, nmet2 in enumerate(n):
            # INITIALIZE POP object
            pop = Pop(nanop.get_atoms_obj_skel().copy(), nanop.bonds_list,
                      metals, shape=shape, n_metal2=nmet2, **kwargs)

            # run GA simulation
            pop.run(max_generations, min_generations, max_nochange)

            # if new minimum CE found and <save_data>
            # store result in DB
            if pop.is_new_min() and save_data:
                new_min_structs += 1
                tot_new_structs += 1
                pop.save_to_db()

        outp = 'Completed Size %i of %i (%i new mins)' % (struct_i + 1,
                                                          nstructs,
                                                          new_min_structs)
        print('-' * CENTER)
        print(outp.center(CENTER))
        print('-' * CENTER)

    # insert log results into DB
    if save_data:
        db_inter.insert_bimetallic_log(
            start_time=start_time,
            metal1=metal1,
            metal2=metal2,
            shape=shape,
            ga_generations=max_generations,
            shell_range='%i - %i' % tuple(nshell_range),
            new_min_structs=tot_new_structs,
            tot_structs=tot_structs,
            batch_run_num=batch_runinfo)
