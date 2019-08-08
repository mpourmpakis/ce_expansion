import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.tri as tri
from npdb import db_inter
import atomgraph
try:
    from ce_expansion.src.npdb import db_inter
    import ce_expansion.src.atomgraph as atomgraph
except:
    pass

# GLOBAL fontsize of axis labels and text
FS = 40

shape = ['icosahedron', 'cuboctahedron', 'elongated-pentagonal-bipyramid'][1]
metals = 'aucu'
minn = 1 if shape.startswith('cub') else 2
for s in range(5, 11):
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
    z = ag.countMixing(ordering)
    tot_bondfracs = z / z.sum()

    # drop <n> outer and <n> inner shell from calculations
    nlayerdrop = 3
    todrop = []
    for n in range(nlayerdrop):
        todrop += list(db_inter.build_atoms_in_shell_list(shape, s - n))
        todrop += list(db_inter.build_atoms_in_shell_list(shape, minn + n))

    # number of shells to study
    nshellstudy = s - bool(not shape.startswith('cub')) - 2 * nlayerdrop

    # list of atom indices to use in calculation
    tokeep = [i for i in range(len(ordering)) if i not in todrop]

    # track counts
    # [Au-Au, Au-Cu, Cu-Cu]
    counts = np.zeros((len(tokeep), 3))

    test_atom = res.build_atoms_obj()
    for i in todrop:
        test_atom[i].symbol = 'Br'

    for i, atomi in enumerate(tokeep):
        a1 = ordering[atomi]
        matches = np.unique(bonds[np.where(bonds == atomi)[0]])
        atomi2s = np.array([j for j in matches if j != atomi])
        for atomi2 in atomi2s:
            a2 = ordering[atomi2]
            counts[i, a1 + a2] += 1
    au_counts = counts[np.where(counts[:, 2] == 0)[0]][:, :2]
    cu_counts = np.flip(counts[np.where(counts[:, 0] == 0)[0]][:, 1:], 0)

    au_fracs = au_counts.mean(0) / 12
    cu_fracs = cu_counts.mean(0) / 12

    # save half of test_atom
    if nshellstudy > 0:
        test_atom.positions -= test_atom.positions.mean(0)
        test_atom = test_atom[np.where(
                        abs(test_atom.positions[:, 0]) < 0.5)[0]]
        test_atom.write(os.path.expanduser('~') +
                        '\\desktop\\SAMPLES\\%ishells_%s.xyz'
                        % (nshellstudy, shape[:3]))

    print(''.center(20, '-'))
    print(res.num_atoms)
    print('%i atoms studied' % len(tokeep))
    print('%i shells studied' % nshellstudy)
    print('Au: -Au (%.2f), -Cu (%.2f)' % (au_fracs[0], au_fracs[1]))
    print('Cu: -Cu (%.2f), -Au (%.2f)' % (cu_fracs[0], cu_fracs[1]))
print(''.center(20, '-'))
