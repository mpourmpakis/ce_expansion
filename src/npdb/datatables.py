try:
    from base import Base
except:
    from npdb.base import Base
import sqlalchemy as db
from datetime import datetime
import tempfile
import matplotlib.pyplot as plt
import os
import numpy as np
import ase
from ase.data.colors import jmol_colors
from ase.data import chemical_symbols
import ase.visualize.plot


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
                      - units = nm
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

    -------
    METHODS
    -------
    build_atoms_obj: Returns atoms object of nanoparticle using ordering

    build_chem_formula: Returns chemical formula of string e.g. 'Au23_Cu32'

    build_prdf: returns data of partial radial distribution function
        Args:
            alpha (str): reference position metal
            beta (str): metal to measure distance from reference metal
            dr (float): shell size to quantify counts in PRDF

    build_prdf_plot: creates 4 subplots containing
                     - metal1 : metal1 PRDF (blue)
                     - metal2 : metal2 PRDF (red)
                     - metal1 : metal2 PRDF (purple)
                     - png image of NP

    save_np: saves atoms object of nanoparticle
        Args:
            - path (str): path to save Atoms object (*.xyz, *.pdb, etc.)
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
    last_updated = db.Column(db.DateTime, default=datetime.now,
                             onupdate=datetime.now)

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

        # attribute to store atoms object once it has been built
        self.atoms_obj = None

    def build_atoms_obj(self):
        """
        Returns ase.Atoms object of stable NP found
        - NP built with Nanoparticle.get_atoms_obj_skel
          and atom type added in using self.ordering

        Returns:
            (ase.Atoms): NP from entry
        """

        atom = self.nanoparticle.get_atoms_obj_skel().copy()
        for i, a in zip(self.ordering, atom):
            a.symbol = self.metal1 if i == '0' else self.metal2
        self.atoms_obj = atom.copy()
        return atom

    def build_chem_formula(self, latex=False, bold=False):
        """
        Returns chemical formula of bimetallic NP
        in alphabetical order
        - e.g. Ag6_Au7

        KArgs:
        latex (bool): if True, chemical formula is returned in Latex Math form
                      (Default: False)

        Returns:
            (str)
        """
        if latex:
            form = '$\\rm %s_{%i}%s_{%i}$' % (self.metal1, self.n_metal1,
                                              self.metal2, self.n_metal2)
            if bold:
                form = form.replace('\\rm', '\\rm \\bf')
            return form
        return '%s%i_%s%i' % (self.metal1, self.n_metal1,
                              self.metal2, self.n_metal2)

    def build_prdf(self, alpha=None, beta=None, dr=0.1):
        """
        Calculates partial radial distribution functions
        of metal <alpha> vs. metal <beta>

        Args:
            alpha (str):
            beta (str):
            dr (float):

        Returns:
            Histogram data of each metal in the shells
        """

        # cutoff = diameter in angstrom
        cutoff = (self.diameter * 10) / 2

        atoms = self.build_atoms_obj().copy()

        if not (alpha or beta):
            alpha = self.metal1
            beta = self.metal2

        # booleans to determine if RDF is comparing all atoms,
        # same type of atoms, or two different atom types
        pos = []
        if beta:
            bet = np.array([i.position for i in atoms if i.symbol == beta])
            if len(bet) == 0:
                raise ValueError('%s not present in system' % beta)
        if alpha:
            alp = np.array([i.position for i in atoms if i.symbol == alpha])

            if alpha == beta or not beta:
                atoms = ase.Atoms([i for i in atoms if i.symbol == alpha])
                pos = atoms.positions
        else:
            pos = atoms.positions

        vol = (4 / 3.) * np.pi * (cutoff / 2.) ** 3
        if len(pos):
            N = float(len(pos))
            rho = len(pos) / vol
        else:
            N = float(len(alp))
            rho = len(bet) / vol

        def hist(distances, dr=dr, max_d=cutoff):
            half_dr = dr / 2.
            high = max_d + (dr - max_d % dr)
            bins = int(high / dr)
            gr, x = np.histogram(distances, bins=bins,
                                 range=(0, high))
            gr = gr.astype(float)
            if not isinstance(distances, np.ndarray):
                return gr, x

            x = x[:-1].astype(float)
            return gr

        counts, x = hist([])
        x = x[:-1]

        # if all atoms or same type of atoms
        if len(pos):
            for i in range(len(pos)):
                a = pos[i, :]
                temp = np.delete(pos, i, axis=0)
                dists = np.sqrt(((temp - a) ** 2).sum(1))
                counts += hist(dists)

        # if two different types of atoms
        else:
            for a in alp:
                dists = np.sqrt(((bet - a) ** 2).sum(1))
                counts += hist(dists)
        gr = np.divide(counts, (4 * np.pi * x ** 2 * dr * rho * N),
                       out=counts, where=(x != 0))
        x = x + (dr / 2.)

        return x, gr

    def build_prdf_plot(self):
        """
        Builds a plt.Figure with PRDF subplots:
        M1 vs M1, M1 vs M2
        M2 vs M1, M2 vs M2
        """
        #        red        purple         blue
        cols = ['#1f77b4', '#9467bd', '', '#d62728']
        fig, axes = plt.subplots(2, 2, figsize=(10, 7))

        metals = [self.metal1, self.metal2]
        max_y = 0
        for i in range(2):
            for j in range(2):
                # select subplot axis
                ax = axes[i, j]

                if (i, j) == (1, 0):
                    # ase.visualize.plot.plot_atoms(self.atoms_obj, ax=ax)
                    path = os.path.join(tempfile.gettempdir(), 'temp.png')
                    self.save_np(path)
                    im = plt.imread(path)
                    ax.imshow(im)
                    ax.axis('off')
                    os.remove(path)
                else:
                    # calculate PRDF
                    x, prdf = self.build_prdf(alpha=metals[i], beta=metals[j])
                    if max(prdf) > max_y:
                        max_y = max(prdf)

                    # plot PRDF
                    ax.plot(x, prdf, color=cols[2 * i + j])
                    ax.set_xlabel('r ($\\rm \\AA$)', fontweight='normal',
                                  fontsize=14)
                    ax.set_title('$\\rm g_{%s, %s}(r)$'
                                 % (metals[i], metals[j]),
                                 fontsize=14)

        max_y += 0.2 * max_y
        # axes[0, 0].set_ylim(0, max_y)
        # axes[0, 1].set_ylim(0, max_y)
        # axes[1, 1].set_ylim(0, max_y)
        fig.suptitle('Partial Radial Distribution Functions for\n' +
                     self.build_chem_formula() + ' ' + self.shape,
                     fontweight='normal')
        fig.tight_layout(rect=(0, 0, 1, 0.9))
        return fig

    def build_central_rdf(self, nbins=5, pcty=False):
        atoms = self.build_atoms_obj().copy()

        # center atoms at origin (COP)
        atoms.positions -= atoms.positions.mean(0)

        # atomic numbers for each metal
        num_m1 = chemical_symbols.index(self.metal1)
        num_m2 = chemical_symbols.index(self.metal2)

        # calculate distances from origin
        dists = np.linalg.norm(atoms.positions, axis=1)

        # calculate distances from COP for each metal type
        dist_m1 = dists[atoms.numbers == num_m1]
        dist_m2 = dists[atoms.numbers == num_m2]

        fig, ax = plt.subplots()

        m1_color = jmol_colors[num_m1]
        m2_color = jmol_colors[num_m2]

        # set nbins to num shells if not given
        if not nbins:
            nbins = self.nanoparticle.num_shells

        counts_m1, bin_edges = np.histogram(dist_m1, bins=nbins)
        counts_m2, bin_edges = np.histogram(dist_m2, bins=nbins)

        # get bins for ax.hist
        bins = np.linspace(0, dists.max(), nbins + 1)


        """
        # if pcty, normalize counts to concentrations
        ax.hist(dist_m1, bins=bins, label=self.metal1, edgecolor='k',
                color=m1_color, alpha=0.5)
        ax.hist(dist_m2, bins=bins, label=self.metal2, edgecolor='k',
                color=m2_color, alpha=0.5)
        """

        x = (bin_edges[1:] + bin_edges[:-1]) / 2
        ax.plot(x, counts_m1, 'o-', markeredgecolor='k',
                color=m1_color, markersize=8, label=self.metal1)
        ax.plot(x, counts_m2, 'o-', markeredgecolor='k',
                color=m2_color, markersize=8, label=self.metal2)

        ax.set_xlabel('Distance from Core ($\\rm \\AA$)')
        ax.set_ylabel('Number of Atoms')
        ax.legend()
        fig.tight_layout()
        plt.show()
        return fig

    def save_np(self, path):
        """
        Saves stable NP to desired path
        - uses ase to save
        - supports all ase save types
          e.g. xyz, pdb, png, etc.

        Returns:
            (bool): True if saved successfully
        """
        atom = self.build_atoms_obj()
        atom.info['CE'] = self.CE
        atom.info['EE'] = self.EE
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
    num_bonds = None
    atoms_obj = None

    def __init__(self, shape, num_atoms, num_shells=None):
        self.shape = shape
        self.num_atoms = num_atoms
        self.num_shells = num_shells

    def __len__(self):
        return self.num_atoms

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

    def get_diameter(self):
        """
        """
        self.get_atoms_obj_skel()
        return abs(self.atoms_obj.positions[:, 0].max() -
                   self.atoms_obj.positions[:, 0].min()) / 10

    def load_bonds_list(self):
        if isinstance(self.bonds_list, np.ndarray):
            self.num_bonds = len(self.bonds_list)
            return self.bonds_list

        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '..', '..', 'data', 'bond_lists',
                            self.shape, '%i.npy' % self.num_shells)

        if os.path.isfile(path):
            self.bonds_list = np.load(path)
            self.num_bonds = len(self.bonds_list)
            return self.bonds_list
        else:
            return None


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
    structure_id = db.Column(db.Integer,
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


class BimetallicLog(Base):
    """
    Table to log batch GA sims
    """
    __tablename__ = 'bimetallic_log'

    id = db.Column(db.Integer, primary_key=True, unique=True)
    date = db.Column(db.DateTime)
    runtime = db.Column(db.Interval)
    metal1 = db.Column(db.String(2))
    metal2 = db.Column(db.String(2))
    shape = db.Column(db.String)
    ga_generations = db.Column(db.Integer)
    shell_range = db.Column(db.String)
    new_min_structs = db.Column(db.Integer)
    tot_structs = db.Column(db.Integer)
    batch_run_num = db.Column(db.String)

    def __init__(self, start_time, metal1, metal2, shape, ga_generations,
                 shell_range, new_min_structs, tot_structs,
                 batch_run_num=None):
        self.date = datetime.now()
        self.runtime = self.date - start_time
        self.metal1 = metal1
        self.metal2 = metal2
        self.shape = shape
        self.ga_generations = ga_generations
        self.shell_range = shell_range
        self.new_min_structs = new_min_structs
        self.tot_structs = tot_structs
        self.batch_run_num = batch_run_num


class PartialRadialDists(Base):
    """
    Holds partial radial distribution functions.

    Columns:
        prdf_type (str) : The type of PRDF being calculated here (atom or NP centered)
                          The type can be one of the two following values:
                                Atom_Centered
                                NP_Centered
                          and will raise a ValueError if one of these two values is not used.
        element_center (str[2]) : The element at the center of the distribution (if atomic prdf being used)
                                  If this is supplied for an NP-centered PRDF, the program will raise a ValueError.
        element_tracked (str[2]) : The element whose radial distribution we are following
        distance (float) : How far away from the center of the distribution we are
        frequency (float) : Frequency of element_tracked found at that particular distance
        np_id (int) : The NP this radial distribution function corresponds to

    """
    __tablename__ = "PartialRadialDists"
    id = db.Column(db.Integer, primary_key=True, unique=True)
    prdf_type = db.Column(db.String, nullable=False)
    element_center = db.Column(db.String(2))
    element_tracked = db.Column(db.String(2), nullable=False)
    distance = db.Column(db.Float, nullable=False)
    frequency = db.Column(db.Float, nullable=False)
    np_id = db.Column(db.Integer, nullable=False)

    def __init__(self, np_id, prdf_type, element_tracked, distance, frequency, element_center=None):
        if prdf_type in ["Atom_Centered", "NP_Centered"]:
            self.prdf_type = prdf_type
        else:
            raise ValueError("Expected either Atom_Centered or NP_Centered for prdf_type")
        self.element_tracked = element_tracked
        self.distance = distance
        self.frequency = frequency
        self.np_id = np_id
        if element_center:
            if prdf_type == "Atom_Centered":
                self.element_center = element_center
            else:
                raise ValueError("Element_center only makes sense for atom-centered PRDFs.")


class PostProcessing(Base):
    """
    Logs post-processed data for a NP.

    Currently we have:
        - Chemical Ordering Parameter
        - Partial Radial Distribution Function (atom-centric)
        - Partial Radial Distribution Function (NP-centric)

    Colunns:
    """
    __tablename__ = "PostProcessing"
    id = db.Column(db.Integer, primary_key=True, unique=True)
    chemical_ordering_parameter = db.Column(db.Float)
    atomic_prdf_pointer = db.Column(db.Integer)
    np_prdf_pointer = db.Column(db.Integer)
    np_id = db.Column(db.Integer, nullable=False)  # Corresponds to the Nanoparticle Class's ID

    def __init__(self, chemical_ordering_parameter = None,
                 atomic_prdf_pointer = None,
                 np_prdf_pointer = None):
        self.np_id = np_id
