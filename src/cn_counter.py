import ase.io
import ase.neighborlist
import numpy as np
from pandas import DataFrame

k = 0
perc_Cu_surface = [None] * 21
Ag_Composition = [None] * 21
Au_Composition = [None] * 21
for a in range(0, 101, 5):
    name = "CuboctahedronShell_#3Ag" + str((100 - a)) + "Au" + str(a) + "min.xyz"
    x = ase.io.read(name)
    atom_size = len(x)
    y = ase.neighborlist.neighbor_list("i", x, 3)
    symbols = x.get_chemical_symbols()
    z = zip(symbols, np.bincount(y))
    surface_atoms = []
    cu_surface = []
    ag_surface = []
    for i in range(0, len(np.bincount(y))):
        if np.bincount(y)[i] != 12:
            surface_atoms.append(z[i])

    for j in range(0, len(surface_atoms)):
        if surface_atoms[j][0] == 'Cu':
            cu_surface.append(surface_atoms[j])
        else:
            ag_surface.append(surface_atoms[j])
    Ag_Composition[k] = 100 - a
    Au_Composition[k] = a
    perc_Cu_surface[k] = float(len(cu_surface)) * 100 / float((len(cu_surface) + len(ag_surface)))
    k = k + 1
    # print len(surface_atoms)

print(perc_Cu_surface)

df = DataFrame({'Number of Atoms': atom_size, 'Cu Percentage': Cu_Composition, 'Ag Percentage': Ag_Composition,
                'Perc Cu on Surface': perc_Cu_surface})

df.to_excel('147_surface_perc_cu_cubocta_CuAg.xlsx', sheet_name='sheet1',
            columns=['Number of Atoms', 'Cu Percentage', 'Ag Percentage',
                     'Perc Cu on Surface'], index=False)
