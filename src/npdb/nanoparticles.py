from base import Base
import sqlalchemy as db
import ase


class Nanoparticles(Base):
    __tablename__ = 'nanoparticles'

    id = db.Column(db.Integer, primary_key=True, unique=True)
    shape = db.Column(db.String, nullable=False)
    num_atoms = db.Column(db.Integer, nullable=False)
    num_shells = db.Column(db.Integer)
    bimetallic_results = db.orm.relationship('BimetallicResults',
                                             backref='nanoparticle')

    def __init__(self, shape, num_atoms, num_shells):
        self.shape = shape
        self.num_atoms = num_atoms
        self.num_shells = num_shells

    def get_atoms_obj_skel(self):
        return ase.Atoms([ase.Atom('Cu', (i.x, i.y, i.z)) for i in self.atoms])
