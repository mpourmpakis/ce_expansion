import os
import numpy as np
import matplotlib.pyplot as plt
from npdb import db_inter
import atomgraph
try:
    from ce_expansion.src.npdb import db_inter
    import ce_expansion.src.atomgraph as atomgraph
except:
    pass

# GLOBAL fontsize of axis labels and text
FS = 40

shape = ['icosahedron', 'cuboctahedron',
         'elongated-pentagonal-bipyramid', 'fcc-cube'][0]
metals = 'aucu'

minn = 1 if shape.startswith('cub') else 2

# number of shells on inside and outside to ignore in calculation
buffer = 3

for s in range(2 * buffer + 1, 11):
    if shape.startswith('cub'):
        s -= 1
    # get number of atoms
    n = db_inter.get_shell2num(shape, s)

    # get half such that metal1 has the extra atom (if natoms is odd)
    n = (n + n % 2) // 2

    res = db_inter.get_bimet_result(metals, shape=shape, num_shells=s,
                                    n_metal1=n)

    # get ordering array
    ordering = np.array([int(i) for i in res.ordering])

    # load bonds list
    bonds = res.nanoparticle.load_bonds_list()

    # build atomgraph object
    ag = atomgraph.AtomGraph(bonds, 'Au', 'Cu')

    # get atom indices for each shell
    shells = db_inter.build_atoms_in_shell_dict(shape, s)

    # create a 'Test Atom' to ensure shells are being correctly counted
    test_atom = res.build_atoms_obj()

    # remove shells from dict not in study
    maxshell = max(shells.keys())
    dropcount = 0
    for drop in range(buffer):
        # pop inner and outer <buffer> layers
        for d in shells.pop(drop) + shells.pop(maxshell - drop):
            # set symbol of dropped atoms to Br
            test_atom[d].symbol = 'Br'

            # track number of atoms dropped
            dropcount += 1

    # track counts
    # [Au-Au, Au-Cu, Cu-Cu]
    counts = np.zeros((len(test_atom) - dropcount, 3))

    # number of shells in study
    nshellstudy = len(shells)

    atomi = 0
    for s in sorted(shells):
        for i in shells[s]:
            # base atom type (0: Au, 1: Cu)
            a1 = ordering[i]
            matches = np.unique(bonds[np.where(bonds == i)[0]])
            i2s = np.array([j for j in matches if j != atomi])
            for i2 in i2s:
                a2 = ordering[i2]
                counts[atomi, a1 + a2] += 1
            atomi += 1

    # get each count type
    au_counts = counts[np.where(counts[:, 2] == 0)[0]][:, :2]
    cu_counts = np.flip(counts[np.where(counts[:, 0] == 0)[0]][:, 1:], 0)

    # ensure that all atoms have been correctly accounted for
    assert len(au_counts) + len(cu_counts) == len(test_atom) - dropcount
    assert len(au_counts) == (test_atom.symbols == 'Au').sum()
    assert len(cu_counts) == (test_atom.symbols == 'Cu').sum()

    # calc count fractions
    au_fracs = au_counts.mean(0) / 12
    cu_fracs = cu_counts.mean(0) / 12

    # TEMP FIX!!!
    # only look at CN 12 atoms
    tokeepcn = np.where(ag.cns == 12)[0]
    todropcn = np.where(ag.cns != 12)[0]

    # save half of test_atom
    if nshellstudy > 0:
        test_atom.positions -= test_atom.positions.mean(0)
        test_ato2 = test_atom[np.where(
                        abs(test_atom.positions[:, 0]) < 1)[0]]
        test_ato2.write(os.path.expanduser('~') +
                        '\\desktop\\SAMPLES\\slice-%ishells_%s.xyz'
                        % (nshellstudy, shape[:3]))

        del test_atom[test_atom.symbols == 'Br']
        test_atom.write(os.path.expanduser('~') +
                        '\\desktop\\SAMPLES\\%ishells_%s.xyz'
                        % (nshellstudy, shape[:3]))

    print(''.center(20, '-'))
    print(shape)
    print('%i total atoms' % res.num_atoms)
    print('%i atoms ignored' % dropcount)
    print('%i atoms studied' % len(counts))
    print('%i shells studied' % nshellstudy)
    print('Au: -Au (%.2f), -Cu (%.2f)' % (au_fracs[0], au_fracs[1]))
    print('Cu: -Cu (%.2f), -Au (%.2f)' % (cu_fracs[0], cu_fracs[1]))
print(''.center(20, '-'))
