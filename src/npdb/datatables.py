try:
    from base import Base
except:
    from npdb.base import Base
import sqlalchemy as db
import ase


class BimetallicResults(Base):
    """
    Bimetallic GA Simulation Results Datatable
    - contains data for most stable structure found
      (based on CE) at a given shape, size, and metal composition
    - GA varies chemical ordering to find stable NP

    Columns:
    metal1 and metal2 (string(2)): metal element names that are always
                                   converted to alphabetical order
                                   - e.g. metal1 = Ag, metal2 = Cu
    shape (string): shape of NP
    num_atoms (int): number of atoms in NP
    diameter (float): "diameter" of NP measured using (atom.cell.max() / 10)
                      - measured in ga.run_ga
                      - NOTE: std measuring, but there might be a better way
    n_metal1, n_metal2 (int): number of metal(1)(2) in NP
                              - must add up to num_atoms
                              - constrains composition of NP
    CE (float): cohesive energy of NP (in ev / atom)
    EE (float): excess energy of NP (in ev / atom)
    ordering (string): string of 1's and 0's mapping atom type
                       to Atoms skeleton
                       - 1: metal2
                       - 0: metal1
                       - atoms of each NP are ordered with an index to
                         ensure ordering maps correctly

    Autofilled Columns:
    id: primary key (unique)
    structure_id (int): Foreign key from Nanoparticles to link GA result
                        to a single NP

    Mapped Properties:
    nanoparticle: (Nanoparticle Datatable entry) links to NP skeleton used
                  in GA sim (size, and shape constraints)
    """
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
    structure_id = db.Column(db.Integer, db.ForeignKey('nanoparticles.id'))

    def __init__(self, metal1, metal2, shape, num_atoms, diameter,
                 n_metal1, n_metal2, CE, EE, ordering):
        self.metal1, self.metal2 = sorted([metal1, metal2])
        self.shape = shape
        self.num_atoms = num_atoms
        self.diameter = diameter
        self.n_metal1 = n_metal1
        self.n_metal2 = n_metal2
        self.CE = CE
        self.EE = EE
        self.ordering = ordering

    def get_chem_formula(self):
        """
        Returns chemical formula of bimetallic NP
        in alphabetical order
        - e.g. Ag6_Au7

        Returns:
            (str)
        """
        return '%s%i_%s%i' % (self.metal1, self.n_metal1,
                              self.metal2, self.n_metal2)

    def get_atoms_obj(self):
        """
        Returns ase.Atoms object of stable NP found
        - NP built with Nanoparticle.get_atoms_obj_skel
          and atom type added in using self.ordering

        Returns:
            (ase.Atoms): NP from entry
        """

        atom = self.nanoparticle.get_atoms_obj_skel()
        for i, a in zip(self.ordering, atom):
            a.symbol = self.metal1 if i == '0' else self.metal2
        return atom

    def save_np(self, path):
        """
        Saves stable NP to desired path
        - uses ase to save
        - supports all ase save types
          e.g. xyz, pdb, png, etc.

        Returns:
            (bool): True if saved successfully
        """
        atom = self.get_atoms_obj()
        atom.write(path)
        return True


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

    Other Attributes:
    bonds_list (np.array): used to carry bonds_list for GA sims
    atoms_obj (ase.Atoms): stores an ase.Atoms skeleton after being built with
                           get_atoms_obj_skel
    """
    __tablename__ = 'nanoparticles'

    id = db.Column(db.Integer, primary_key=True, unique=True)
    shape = db.Column(db.String, nullable=False)
    num_atoms = db.Column(db.Integer, nullable=False)
    num_shells = db.Column(db.Integer)
    atoms = db.orm.relationship('Atoms', cascade='all, delete',
                                backref='nanoparticle')
    bimetallic_results = db.orm.relationship('BimetallicResults',
                                             backref='nanoparticle')

    # used to attach bond_list to data entry
    # does not store bond_list in DB
    bonds_list = None
    atoms_obj = None

    def __init__(self, shape, num_atoms, num_shells=None):
        self.shape = shape
        self.num_atoms = num_atoms
        self.num_shells = num_shells

    def get_atoms_obj_skel(self):
        """
        Builds NP in the form of a Cu NP ase.Atoms object
        - stores under self.atoms_obj property
        - returns atoms_obj
        """
        if not self.atoms_obj:
            self.atoms_obj = ase.Atoms([ase.Atom('Cu', (i.x, i.y, i.z))
                                       for i in self.atoms])
        return self.atoms_obj


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
                             db.ForeignKey('nanoparticles.id',
                                           ondelete='CASCADE'))

    def __init__(self, index, x, y, z, nanoparticle):
        self.index = index
        self.x = x
        self.y = y
        self.z = z
        self.nanoparticle = nanoparticle


class ModelCoefficients(Base):
    """
        Bond-Centric Model Coefficients (gamma) precalculated
        based on atom types and coordination number
    """
    __tablename__ = 'model_coefficients'

    id = db.Column(db.Integer, primary_key=True, unique=True)
    element1 = db.Column(db.String(2), nullable=False)
    element2 = db.Column(db.String(2), nullable=False)
    cn = db.Column(db.Integer, nullable=False)
    bond_energy = db.Column(db.Float)

    def __init__(self, element1, element2, cn, bond_energy):
        self.element1 = element1
        self.element2 = element2
        self.cn = cn
        self.bond_energy = bond_energy
