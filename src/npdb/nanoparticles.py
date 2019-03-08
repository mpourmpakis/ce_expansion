from base import Base
import sqlalchemy as db
import ase


class Nanoparticles(Base):
    """
    Nanoparticles (NP) Skeleton Datatable
    - contains header info for NP skeletons used in GA sims

    Columns:
    shape (string): shape of NP
    num_atoms (int): number of atoms in NP
    num_shells (int): number of shells used to build NP
                      from structure_gen module
                      - not required in initial entry

    Autofilled Columns:
    id: primary key (unique)
    bimetallic_results (one-to-many): links to entries in BimetallicResults
                                      that used this NP skeleton
                                      - new BimetallicResults entries require
                                        a NP to be entered
                                      - one NP can have many BimetallicResults
    """
    __tablename__ = 'nanoparticles'

    id = db.Column(db.Integer, primary_key=True, unique=True)
    shape = db.Column(db.String, nullable=False)
    num_atoms = db.Column(db.Integer, nullable=False)
    num_shells = db.Column(db.Integer)
    bimetallic_results = db.orm.relationship('BimetallicResults',
                                             backref='nanoparticle')

    def __init__(self, shape, num_atoms, num_shells=None):
        self.shape = shape
        self.num_atoms = num_atoms
        self.num_shells = num_shells

    def get_atoms_obj_skel(self):
        """
        Returns NP in the form of a Cu NP ase.Atoms object
        """
        return ase.Atoms([ase.Atom('Cu', (i.x, i.y, i.z)) for i in self.atoms])
