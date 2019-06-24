import matplotlib.pyplot as plt
import numpy as np
import ase.io
from ase.calculators import emt
from npdb import db_inter
try:
    from ce_expansion.src.npdb import db_inter
except:
    pass


# mono Au NP
au = ase.Atoms('Au')
au.set_calculator(emt.EMT())
au_en = au.get_potential_energy()

# mono Ag NP
ag = ase.Atoms('Ag')
ag.set_calculator(emt.EMT())
ag_en = ag.get_potential_energy()


ns = list(range(310))
ee = np.zeros(len(ns))
bcee = np.zeros(len(ns))
for i, n in enumerate(ns):
    # bimet NP
    res = db_inter.get_bimet_result('agau', 'icosahedron', num_atoms=309,
                                    n_metal1=n, lim=1)
    atom = res.build_atoms_obj()
    atom = ase.Atoms(atom)
    atom.set_calculator(emt.EMT())
    energy = atom.get_potential_energy()
    n_au = sum(atom.numbers == 79)
    x_au = n_au / len(atom)

    n_ag = len(atom) - n_au
    x_ag = n_ag / len(atom)

    ee[i] = (energy - x_au * au_en - x_ag * ag_en) / len(atom)
    bcee[i] = res.EE
    print(' %03i Left' % ns[-i - 1], end='\r')
print('  ALL DONE  ')
plt.plot(ns, ee, 'o-', label='EMT')
plt.plot(ns, bcee, 'o-', label='BC')
plt.xlabel('$\\rm N_{Au}$')
plt.ylabel('EE (eV / atom)')
plt.tight_layout()
plt.legend()
plt.show()
