import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import ase.cluster
import ase.io
import ase.optimize
from ase.calculators import emt
from npdb import db_inter
try:
    from ce_expansion.src.npdb import db_inter
except:
    pass

np_ga = os.path.join('D:\\MCowan', 'Box Sync', 'Michael_Cowan_PhD_research',
                     'np_ga', 'larson_comparison')


def save_lars(a):
    a.write(os.path.join(np_ga, 'larson_opt_structures',
                         '%s_emtopt.xyz' % a.get_chemical_formula()))


def save_bc(a):
    a.write(os.path.join(np_ga, 'bc_opt_structures',
                         '%s_emtopt.xyz' % a.get_chemical_formula()))

# 309-atom icosahedron skeleton
skel = ase.Atoms([a for a in ase.cluster.Icosahedron('Cu', 5)])

# mono Au NP
au = ase.Atoms(skel.copy())
for a in au:
    a.symbol = 'Au'

au.set_calculator(emt.EMT())
opt = ase.optimize.BFGS(au)
opt.run()
au_en = au.get_total_energy()
save_bc(au)

# mono Ag NP
ag = ase.Atoms(skel.copy())
for a in ag:
    a.symbol = 'Ag'

ag.set_calculator(emt.EMT())
opt = ase.optimize.BFGS(ag)
opt.run()
ag_en = ag.get_total_energy()
save_bc(ag)

lars_ag = ase.io.read(os.path.join(np_ga, 'larson_orig_structures',
                                   'Ag309.xyz'))
lars_ag.set_calculator(emt.EMT())
opt = ase.optimize.BFGS(lars_ag)
opt.run()
lars_ag_en = lars_ag.get_total_energy()
save_lars(lars_ag)

lars_au = ase.io.read(os.path.join(np_ga, 'larson_orig_structures',
                                   'Au309.xyz'))
lars_au.set_calculator(emt.EMT())
opt = ase.optimize.BFGS(lars_au)
opt.run()
lars_au_en = lars_au.get_total_energy()
save_lars(lars_au)

ns = list(range(0, 310))
laee = np.zeros(len(ns))
bcee = np.zeros(len(ns))
num_atoms = 309
for i, n in enumerate(ns):
    # calc composition data
    n_ag = n
    x_ag = n / num_atoms

    n_au = num_atoms - n
    x_au = n_au / num_atoms

    if n == 0:
        bcenergy = au_en
        laenergy = lars_au_en
    elif n == 309:
        bcenergy = ag_en
        laenergy = lars_ag_en
    else:
        # our bimet NP
        res = db_inter.get_bimet_result('agau', 'icosahedron',
                                        num_atoms=num_atoms,
                                        n_metal1=n, lim=1)
        atom = res.build_atoms_obj()
        atom = ase.Atoms(atom)
        atom.set_calculator(emt.EMT())
        opt = ase.optimize.BFGS(atom)
        opt.run()
        bcenergy = atom.get_total_energy()
        save_bc(atom)

        # Larson's minimum bimet NP
        larson = ase.io.read(os.path.join(np_ga, 'larson_orig_structures',
                             '%s.xyz' % atom.get_chemical_formula()))
        larson = ase.Atoms(larson)
        larson.set_calculator(emt.EMT())
        opt = ase.optimize.BFGS(larson)
        opt.run()
        laenergy = larson.get_total_energy()
        save_lars(larson)

    laee[i] = (laenergy - x_au * lars_au_en - x_ag * lars_ag_en) / num_atoms
    bcee[i] = (bcenergy - x_au * au_en - x_ag * ag_en) / num_atoms
    print(' %03i Left' % ns[-i - 1])

print('  ALL DONE  ')

# save EE arrays
for name, arr in zip(['LARSON', 'BC'], [laee, bcee]):
    fname = os.path.join(np_ga, '%s_EE-increasing-Ag.npy' % name)
    np.save(fname, arr)

plt.plot(ns, laee, 'o-', label='Larson',
         markeredgecolor='k', markersize=10)
plt.plot(ns, bcee, 'o-', label='BC',
         markeredgecolor='k', markersize=10)
plt.xlabel('$\\rm N_{Ag}$')
plt.ylabel('EE (eV / atom)')
plt.tight_layout()
plt.legend()
plt.savefig(os.path.join(np_ga, 'larson_vs_bc_EMTopt.svg'))
plt.savefig(os.path.join(np_ga, 'larson_vs_bc_EMTopt.png'))
plt.show()
