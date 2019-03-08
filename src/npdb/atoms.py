from base import Base
import sqlalchemy as db


class Atoms(Base):
    """
    Atom coordinates that link to a Nanoparticles skeleton
    - no atom type needed since these only correspond to a skeleton
    - BimetallicResults holds atom type info

    Columns:
    index (int): atom index to ensure atom types from
                 BimetallicResults.ordering can be correctly mapped
    x, y, z (float): coordinates of atom
    nanoparticle (many-to-one): Nanoparticles entry that atom belongs to
                                - atom must belong to one NP

    Autofilled Columns:
    id (int): primary key (unique)
    structure_id (int): Foreign key from Nanoparticles to link atom
                        to a single NP
    """
    __tablename__ = 'atoms'

    id = db.Column(db.Integer, primary_key=True, unique=True)
    index = db.Column('index', db.Integer, nullable=False)
    x = db.Column(db.Float, nullable=False)
    y = db.Column(db.Float, nullable=False)
    z = db.Column(db.Float, nullable=False)
    structure_id = db.Column(db.String(36),
                             db.ForeignKey('nanoparticles.id'))
    nanoparticle = db.orm.relationship('Nanoparticles', uselist=False,
                                       backref='atoms')

    def __init__(self, index, x, y, z, nanoparticle):
        self.index = index
        self.x = x
        self.y = y
        self.z = z
        self.nanoparticle = nanoparticle
