from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

"""
Base module:
- defines np_ce path
- initializes engine to connect to np.db (SQLite DB)
- session factory used to open connections with np.db
- Base class to create DB datatable objects
- Creates all datatables if not present in DB
"""


np_ce_path = os.path.join(os.path.expanduser('~'), 'Box Sync',
                          'Michael_Cowan_PhD_research', 'data',
                          'np_ce')

engine = create_engine('sqlite:///' + os.path.join(np_ce_path, 'np.db'))
Session = sessionmaker(bind=engine)

Base = declarative_base()
