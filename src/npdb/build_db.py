import sqlalchemy as db
from base import Session, Base, engine, np_ce_path
from bimetallic_results import BimetallicResults as BiMet
from atoms import Atoms
from nanoparticles import Nanoparticles
import pandas as pd
import os
import ase.io

"""
    Script to build/update SQLite DB from GA Sim data
    (excel files and structure xyzs)
"""

session = Session()
Base.metadata.create_all(engine)

rename_columns = ['num_atoms', 'diameter', 'composition',
                  'n_metal1', 'n_metal2', 'CE', 'EE']
db_columns = ['metal1', 'metal2', 'shape', 'num_atoms', 'diameter',
              'n_metal1', 'n_metal2', 'CE', 'EE', 'ordering']

xyz_basepath = os.path.join(np_ce_path, 'structures')

path = str(os.sep.join(os.path.realpath(__file__)
                       .split(os.sep)[:-3] + ["data", 'bimetallic_results']))

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
                new_min = False
                # check to see if result already exists in DB
                match = db.and_(BiMet.metal1 == metal1,
                                BiMet.metal2 == metal2,
                                BiMet.shape == shape,
                                BiMet.num_atoms == r.num_atoms,
                                BiMet.n_metal1 == r.n_metal1)

                found_in_db = session.query(BiMet).filter(match).first()
                if found_in_db:
                    # if new min structure is found, need to update CE, EE,
                    # and ordering
                    if found_in_db.CE < r.CE:
                        new_min = True
                    else:
                        continue

                # read in xyz object
                atom = ase.io.read(os.path.join(xyz_basepath, metals, shape,
                                   str(r.num_atoms),
                                   '%s%i_%s%i.xyz' % (metal1, r.n_metal1,
                                                      metal2, r.n_metal2))
                                   )
                r.ordering = ''.join(['1' if at.symbol == metal2 else '0'
                                      for at in atom])

                # update CE, EE, and ordering
                if new_min:
                    found_in_db.CE = r.CE
                    found_in_db.EE = r.EE
                    found_in_db.ordering = r.ordering
                    session.commit()
                    continue

                np = session.query(Nanoparticles) \
                    .filter(db.and_(Nanoparticles.shape == shape,
                                    Nanoparticles.num_atoms == r.num_atoms)) \
                    .first()

                if not np:
                    np = Nanoparticles(shape, r.num_atoms, None)
                    for a in atom:
                        session.add(Atoms(i, a.x, a.y, a.z, np))

                res = BiMet(metal1, metal2, shape, r.num_atoms,
                            r.diameter, r.n_metal1, r.n_metal2,
                            r.CE, r.EE, r.ordering)
                np.bimetallic_results.append(res)
                session.add(np)
                session.add(res)
                session.commit()
session.close()
