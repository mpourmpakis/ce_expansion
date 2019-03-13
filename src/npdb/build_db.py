import base
import db_inter
import pandas as pd
import os
import ase.io

"""
    Script to build/update SQLite DB from GA Sim data
    (excel files and structure xyzs)
    - TODO: create interface b/n GA sim and DB
"""

# linking shape and num_atoms to num_shell (lazy way)
cubo = [13, 55, 147, 309, 561, 923, 1415, 2057, 2869, 3871,
        5083, 6525, 8217, 10179]
elo = [13, 55, 147, 309, 561, 923, 1415, 2057, 2869, 3871]
fcc = [13, 63, 171, 365, 665, 1099, 1687, 2457, 3429, 4631,
       6083, 7813, 9841, 12195]
ico = [13, 55, 147, 309, 561, 923, 1415, 2057, 2869, 3871, 5083, 6525]

shell = {'cuboctahedron': {i: v for i, v in zip(cubo, range(1, 15))},
         'elongated-pentagonal-bipyramid': {i: v for i, v
                                            in zip(elo, range(2, 12))},
         'fcc-cube': {i: v for i, v in zip(fcc, range(1, 15))},
         'icosahedron': {i: v for i, v in zip(ico, range(2, 14))}
         }

# if datatables don't exist yet, build them
base.Base.metadata.create_all(base.engine)

# rename excel columns to appropriate DB columns
rename_columns = ['num_atoms', 'diameter', 'composition',
                  'n_metal1', 'n_metal2', 'CE', 'EE']
db_columns = ['metal1', 'metal2', 'shape', 'num_atoms', 'diameter',
              'n_metal1', 'n_metal2', 'CE', 'EE', 'ordering']

np_ce_path = os.path.join(os.path.expanduser('~'), 'Box Sync',
                          'Michael_Cowan_PhD_research', 'data',
                          'np_ce')

# path to structure XYZ files (based on old GA structure saving method)
xyz_basepath = os.path.join(np_ce_path, 'structures')

# path to bimetallic excel result files (based on old GA data saving method)
path = str(os.sep.join(os.path.realpath(__file__)
                       .split(os.sep)[:-3] + ["data", 'bimetallic_results']))

# walk through result excel files to add/update their data to DB
for root, ds, fs in os.walk(path):
    for f in fs:
        if f:
            shape, metals = f.split('_')[:2]
            metal1, metal2 = metals[:2], metals[2:]
            df = pd.read_excel(os.path.join(root, f))
            df.columns = rename_columns
            df['metal1'] = metal1
            df['metal2'] = metal2
            df['shape'] = shape
            df['ordering'] = None
            df = df[db_columns]
            for i, r in df.iterrows():
                # check to see if result already exists in DB
                res_found = db_inter.get_bimet_result(metals, shape,
                                                      r.num_atoms,
                                                      r.n_metal1)

                # continue to next run if entry already exists and CE is lower
                if res_found and res_found.CE <= r.CE:
                    continue

                # read in xyz object
                atom = ase.io.read(os.path.join(xyz_basepath, metals, shape,
                                   str(r.num_atoms),
                                   '%s%i_%s%i.xyz' % (metal1, r.n_metal1,
                                                      metal2, r.n_metal2))
                                   )
                r.ordering = ''.join(['1' if at.symbol == metal2 else '0'
                                      for at in atom])

                # check to see if NP skeleton already exists in DB
                np = db_inter.get_nanoparticle(shape=shape,
                                               num_atoms=r.num_atoms)

                # if NP not in DB, add it (automatically adds atoms as well)
                if not np:
                    db_inter.insert_nanoparticle(atom, shape,
                                                 shell[shape][len(atom)])
                    np = db_inter.get_nanoparticle(shape=shape,
                                                   num_atoms=r.num_atoms)

                # insert/update BimetallicResults entry
                db_inter.update_bimet_result(metals, shape, r.num_atoms,
                                             r.diameter, r.n_metal1,
                                             r.CE, r.ordering, r.EE, nanop=np)
