from base import Base
import sqlalchemy as db
import ase


class BimetallicResults(Base):
    __tablename__ = 'bimetallic_results'

    id = db.Column(db.Integer, primary_key=True, unique=True)
    metal1 = db.Column(db.String(2), nullable=False)
    metal2 = db.Column(db.String(2), nullable=False)
    shape = db.Column(db.String, nullable=False)
    num_atoms = db.Column(db.Integer, nullable=False)
    diameter = db.Column(db.Float, nullable=False)
    n_metal1 = db.Column(db.Integer, nullable=False)
    n_metal2 = db.Column(db.Integer, nullable=False)
    CE = db.Column(db.Float, nullable=False)
    EE = db.Column(db.Float)
    ordering = db.Column(db.String(50000), nullable=False)
    structure_id = db.Column(db.String(36),
                             db.ForeignKey('nanoparticles.id'))

    def __init__(self, metal1, metal2, shape, num_atoms, diameter,
                 n_metal1, n_metal2, CE, EE, ordering):
        self.metal1 = metal1
        self.metal2 = metal2
        self.shape = shape
        self.num_atoms = num_atoms
        self.diameter = diameter
        self.n_metal1 = n_metal1
        self.n_metal2 = n_metal2
        self.CE = CE
        self.EE = EE
        self.ordering = ordering

    def get_chem_formula(self):
        return '%s%i_%s%i' % (self.metal1, self.n_metal1,
                              self.metal2, self.n_metal2)

    def get_atoms_obj(self):
        atom = self.nanoparticle.get_atoms_obj_skel()
        for i, a in zip(self.ordering, atom):
            a.symbol = self.metal1 if i == '0' else self.metal2
        return atom

    def save_xyz(self, path):
        atom = self.get_atoms_obj()
        atom.write(path)
