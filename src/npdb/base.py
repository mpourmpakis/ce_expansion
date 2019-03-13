from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

"""
Base module:
- defines database path
- initializes engine to connect to np.db (SQLite DB)
- session factory used to open connections with np.db
- Base class to create DB datatable objects
"""

db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       '..',
                       '..',
                       'data',
                       'np.db')
engine = create_engine('sqlite:///' + db_path)
Session = sessionmaker(bind=engine)

Base = declarative_base()
