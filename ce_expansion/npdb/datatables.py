import os
from datetime import datetime
import json
from typing import Iterable

import ase
import ase.visualize
import matplotlib.pyplot as plt
import numpy as np
import sqlalchemy as db
from ase.data import chemical_symbols, covalent_radii
from ase.data.colors import jmol_colors
import ase.units as units

from ce_expansion.npdb.base import Base
import ce_expansion.npdb.db_utils as db_utils
from ce_expansion.atomgraph import adjacency


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
    build_atoms_obj: DEPRECATED: use get_atoms_obj method or atoms_obj attr
                     (Returns atoms object of nanoparticle using ordering)

    build_chem_formula: DEPRECATED: use get_chemical_formula
                        (Returns chemical formula of string e.g. 'Au23_Cu32')

    get_chemical_formula: Returns chemical formula as string, e.g. 'Au23Cu32'

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
    # compressed_ordering string = integer value
    # assuming ordering string is in binary
    compressed_ordering = db.Column(db.String, nullable=False)
    structure_id = db.Column(db.Integer, db.ForeignKey('nanoparticles.id'))
    last_updated = db.Column(db.DateTime, default=datetime.now,
                             onupdate=datetime.now)

    # stores actual ordering string
    # (instead of compressed ordering which is stored in DB)
    _actual_ordering = None

    # attribute to store atoms object once it has been built
    atoms_obj = None
    _atoms_obj = None

    # store Smix
    smix = None
    _smix = None

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

    @property
    def ordering(self):
        # convert compressed_ordering to binary = actual_orderingstr
        if self._actual_ordering is None:
            self._actual_ordering = f'{int(self.compressed_ordering):b}'
            self._actual_ordering = self._actual_ordering.zfill(self.num_atoms)
            self._actual_ordering = np.array(list(self._actual_ordering), int)
        return self._actual_ordering

    @ordering.setter
    def ordering(self, ordering):
        # ordering must be str or iterable
        if isinstance(ordering, str):
            ordering_str = ordering
            ordering = np.array(list(ordering), int)
        elif isinstance(ordering, Iterable):
            ordering_str = ''.join(map(str, ordering))
        else:
            raise ValueError("Invalid ordering! Must be array of ints or str.")

        # set compressed ordering using str representation
        self.compressed_ordering = str(int(ordering_str, 2))
        # set actual ordering attr to array of ints
        self._actual_ordering = ordering

    @property
    def atoms_obj(self):
        return self.get_atoms_obj()

    @property
    def smix(self):
        if self._smix is None:
            # get atomic percent, x
            x = np.bincount(self.ordering) / self.num_atoms
            self._smix = -units.kB * (x * np.log(x)).sum()
        return self._smix

    def get_atoms_obj(self):
        """
        Returns ase.Atoms object of stable NP found
        - NP built with Nanoparticle.get_atoms_obj_skel
          and atom type added in using self.ordering

        Returns:
            (ase.Atoms): NP from entry
        """

        if self._atoms_obj is None:
            atom = self.nanoparticle.atoms_obj.copy()
            syms = np.array([self.metal1, self.metal2])
            atom.symbols = syms[[*map(int, self.ordering)]]
            self._atoms_obj = atom.copy()
        return self._atoms_obj

    def build_atoms_obj(self):
        """
        DEPRECATED: use get_atoms_obj method
        """
        return self.get_atoms_obj()

    def get_chemical_formula(self, latex=False, bold=False):
        """
        Returns chemical formula of bimetallic NP
        in alphabetical order (similar to ase's Atoms.get_chemical_formula)
        - e.g. Ag6Au7

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
        return '%s%i%s%i' % (self.metal1, self.n_metal1,
                             self.metal2, self.n_metal2)

    def get_gmix(self, T=298):
        return self.EE - T * self.smix

    def build_chem_formula(self, latex=False, bold=False):
        """
        DEPRECATED: use get_chemical_formula

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

                # visualize NP
                if (i, j) == (1, 0):
                    atom = self.build_atoms_obj()
                    for a in atom:
                        circ = plt.Circle((a.x, a.y),
                                          radius=covalent_radii[a.number],
                                          facecolor=jmol_colors[a.number],
                                          edgecolor='k',
                                          linewidth=1,
                                          zorder=a.z)
                        ax.add_patch(circ)
                    ax.autoscale()
                    ax.set_aspect(1)
                    ax.axis('off')
                # calculate PRDF
                else:
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

        fig.suptitle('Partial Radial Distribution Functions for\n' +
                     self.build_chem_formula(latex=True) + ' ' + self.shape,
                     fontweight='normal')
        fig.tight_layout(rect=(0, 0, 1, 0.9))
        return fig

    def build_central_rdf(self, nbins=5, pcty=False):
        atoms = self.build_atoms_obj().copy()

        # center atoms at origin (COP)
        atoms.positions -= atoms.positions.mean(0)

        # calculate distances from origin
        dists = np.linalg.norm(atoms.positions, axis=1)

        # calculate distances from COP for each metal type
        dist_m1 = dists[atoms.symbols == self.metal1]
        dist_m2 = dists[atoms.symbols == self.metal2]

        fig, ax = plt.subplots()

        # get jmol color for each metal
        m1_color = jmol_colors[chemical_symbols.index(self.metal1)]
        m2_color = jmol_colors[chemical_symbols.index(self.metal2)]

        # set nbins to num shells if not given
        if not nbins:
            nbins = self.nanoparticle.num_shells

        counts_m1, bin_edges = np.histogram(dist_m1, bins=nbins)
        counts_m2, bin_edges = np.histogram(dist_m2, bins=nbins)

        # get bins for ax.hist
        bins = np.linspace(0, dists.max(), nbins + 1)

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
        atom.info['shape'] = self.shape
        atom.info['CE'] = self.CE
        atom.info['EE'] = self.EE
        atom.info[f'x_{self.metal1}'] = self.n_metal1 / self.num_atoms
        atom.info[f'x_{self.metal2}'] = self.n_metal2 / self.num_atoms

        # save a chemical json file
        if path.endswith('json'):
            path = path.replace('.json', '.cjson')
            # create name
            name = self.get_chemical_formula()
            name += '_' + self.shape.lower()[:3].replace('elo', 'epb')

            # create chemical JSON formula string
            formula = f'{self.metal1} {self.n_metal1} '
            formula += f'{self.metal2} {self.n_metal2}'

            # get ase.Atoms object
            atoms_obj = self.get_atoms_obj()

            # create list of atomic numbers
            numbers = atoms_obj.numbers.tolist()

            symbols = list(atoms_obj.symbols)

            positions = atoms_obj.positions.flatten().tolist()

            data = {'chemical json': 0,
                    'name': name,
                    'formula': formula,
                    'atoms': {'elements': {'number': numbers,
                                           'type': symbols},
                              'coords': {'3d': positions}
                              },
                    'properties': {'cohesive energy': self.CE,
                                   'excess energy': self.EE,
                                   'shape': self.shape.lower()}
                    }
            with open(path, 'w') as fidw:
                json.dump(data, fidw, indent=4)
        else:
            atom.write(path)
        return True

    def show(self):
        """
        Shows nanoparticle using ase.visualize.view
        """
        ase.visualize.view(self.build_atoms_obj())


class PolymetallicResults(Base):
    """
    Polymetallic GA Simulation Results Datatable
    - contains data for most stable structure found
      (based on CE) at a given shape, size, and metal composition
    - GA varies chemical ordering to find stable NP

    -------
    COLUMNS
    -------
    metals_list (str): comma-separated list of unique metal types
                       - order should match ordering numbers
    composition_list (str): comma-separated list of metal counts
                            - must match length of
    shape (str): shape of NP (defined by user)
    CE (float): cohesive energy of NP (in eV / atom)
    EE (float): excess energy of NP (in eV / atom)
    smix (float): ideal entropy of mixing (in eV / K atom)
    ordering_string (str): string of ints mapping metal type
                           to ase.Atoms skeleton
                           - atoms of each NP are ordered with an index to
                             ensure ordering maps correctly

    ------------------
    AUTOFILLED COLUMNS
    ------------------
    id (int): primary key (unique)
    num_atoms (int): number of atoms in NP (computed from ordering array)
    structure_id (int): Foreign key from Nanoparticles to link GA result
                        to a single NP
    last_updated (datetime): tracks the last time the data was created/updated

    Mapped Properties:
    nanoparticle: (Nanoparticle Datatable entry) links to NP skeleton used
                  in GA sim (size, and shape constraints)

    ----------
    PROPERTIES
    ----------
    metals (np.ndarray): array[str] of unique metal types
                         (metal order matches ordering numbers)
    composition (np.ndarray): array[int] of metal counts (sum == num_atoms)
    ordering (np.ndarray): array[int] of chemical ordering (maps to Atoms obj)
    atoms_obj (ase.Atoms): atoms object representation of polymetallic NP

    -------
    METHODS
    -------
    get_chemical_formula: Returns chemical formula as string
                          e.g. 'Ag12Au11Cu32'

    save_np: saves atoms object of nanoparticle
        Args:
        path (str): path to save Atoms object (*.xyz, *.pdb, etc.)

    show: opens ase gui to visualize NP
    """
    __tablename__ = 'polymetallic_results'

    id = db.Column(db.Integer, primary_key=True, unique=True)
    metals_list = db.Column(db.String, nullable=False)
    num_atoms = db.Column(db.Integer, nullable=False)
    composition_list = db.Column(db.Integer, nullable=False)
    shape = db.Column(db.String, nullable=False)
    CE = db.Column(db.Float, nullable=False)
    EE = db.Column(db.Float)
    smix = db.Column(db.Float)
    structure_id = db.Column(db.Integer, db.ForeignKey('nanoparticles.id'),
                             nullable=False)
    ordering_string = db.Column(db.VARCHAR, nullable=False)
    last_updated = db.Column(db.DateTime, default=datetime.now,
                             onupdate=datetime.now)

    # metals and composition array
    # init _ attrs for use of getter-setters
    metals = None
    _metals = None

    composition = None
    _composition = None

    # ordering array
    ordering = None
    _ordering = None

    # attribute to store atoms object once it has been built
    atoms_obj = None
    _atoms_obj = None

    def __init__(self, metals: Iterable[str], composition: Iterable[int],
                 shape: str, CE: float, EE: float, ordering: Iterable[int],
                 smix: float = None):
        """
        Args:
        metals (Iterable[str]): ordered list of unique metal types
        composition (Iterable[int]): metal counts (length >= (len(metals) - 1)
        CE (float): cohesive energy of NP (eV / atom)
        EE (float): excess energy of NP (eV / atom)
        smix (float): ideal entropy of mixing (eV / K atom)
        ordering (Iterable[int]): chemical ordering that maps metals to Atoms
        """
        # DB column should be a string of comma-separated metals
        self.metals_list = ','.join(metals)

        # metals attr is an array of metal strings
        self._metals = np.array(list(metals))

        # get num_atoms from length of ordering
        self.num_atoms = len(ordering)

        # cast composition to list
        composition = list(composition)

        # must give all compositions or len(metals) - 1 (due to DOF)
        if len(self.metals) == len(composition):
            # ensure composition is equal to number of atoms
            if self.num_atoms != sum(composition):
                raise ValueError("Composition does not match number of atoms")
        elif len(self.metals) - len(composition) == 1:
            # add the last composition based on total number of atoms
            composition += [self.num_atoms - sum(composition)]
        else:
            raise ValueError("Invalid composition.")

        # set composition attribute as np array
        self._composition = np.array(list(composition))

        # DB column is a comma-separated string of comps
        self.composition_list = ','.join(map(str, composition))

        # set shape, CE and EE
        self.shape = shape
        self.CE = CE
        self.EE = EE

        # if smix is None, compute configurational entropy
        # using db_utils.smix
        self.smix = smix
        if self.smix is None:
            self.smix = db_utils.smix(self.composition)

        # DB column is a string of ordering characters
        self.ordering_string = ''.join(map(str, ordering))

        # set ordering attr as np array
        self._ordering = np.array(list(ordering))

    @property
    def metals(self):
        """
        Read-only metals property

        Returns:
        (np.ndarray[str]): array of unique metal types
        """
        if self._metals is None:
            self._metals = np.array([*map(str, self.metals_list.split(','))])
        return self._metals

    @property
    def composition(self):
        """
        Read-only metal composition property (i.e. metal counts)

        Returns:
        (np.ndarray[int]): array of metal counts
        """
        if self._composition is None:
            self._composition = np.array(
                                    [*map(int,
                                          self.composition_list.split(','))
                                     ])
        return self._composition

    @property
    def ordering(self):
        """
        Chemical ordering property

        Returns:
        (np.ndarray[int]): ordering array
        """
        # convert compressed_ordering to binary = actual_orderingstr
        if self._ordering is None:
            self._ordering = np.array(list(self.ordering_string), int)
        return self._ordering

    @ordering.setter
    def ordering(self, ordering: Iterable):
        """
        Set chemical ordering property and update ordering_string column

        Args:
        ordering (Iterable): new chemical ordering list

        Raises:
        ValueError: invalid ordering array (wrong length | wrong values)
        """
        if len(ordering) != self.num_atoms:
            raise ValueError("Invalid ordering length.")
        if int(max(ordering)) != len(self.metals) - 1:
            raise ValueError("Invalid ordering numbers. "
                             f"Cannot exceed {len(self.metals) - 1}.")

        self._ordering = np.array([int(i) for i in ordering])
        self.ordering_string = ''.join(map(str, self._ordering))

    @property
    def atoms_obj(self):
        """
        Read-only ase.Atoms object property
        - NP built with Nanoparticle.atoms_obj
          and atom type added in using metals and ordering arrays

        Returns:
        (ase.Atoms): NP from entry
        """
        if self._atoms_obj is None:
            self._atoms_obj = self.nanoparticle.atoms_obj.copy()
            self._atoms_obj.symbols = self.metals[self.ordering]
        return self._atoms_obj

    def get_chemical_formula(self, latex=False, bold=False):
        """
        Returns chemical formula of polymetallic NP
        in alphabetical order (similar to ase's Atoms.get_chemical_formula)
        - e.g. Ag6Au7

        KArgs:
        latex (bool): if True, chemical formula is returned in Latex Math form
                      (Default: False)
        bold (bool): if True, latex string will have bold font key
                     (Default: False)

        Returns:
            (str): chemical formula
        """
        if latex:
            form = '$\\rm '
            form += ''.join([f'{m}_{{{n}}}' for m, n
                             in zip(self.metals, self.composition)])
            form += '$'
            if bold:
                form = form.replace('\\rm', '\\rm \\bf')
            return form
        return ''.join([f'{m}{n}' for m, n
                        in zip(self.metals, self.composition)])

    def get_gmix(self, T=298):
        return self.EE - T * self.smix

    def save_np(self, path):
        """
        Saves stable NP to desired path
        - uses ase to save
        - supports all ase save types
          e.g. xyz, pdb, png, etc.

        Returns:
            (bool): True if saved successfully
        """
        # save a chemical json file
        if path.endswith('json'):
            path = path.replace('.json', '.cjson')
            # create name
            name = self.get_chemical_formula()
            name += '_' + self.nanoparticle \
                              .shape.lower()[:3].replace('elo', 'epb')

            # create chemical JSON formula string
            formula = ' '.join(f'{n} {m}' for n, m
                               in zip(self.metals, self.composition))

            # create list of atomic numbers
            numbers = self.atoms_obj.numbers.tolist()

            symbols = list(self.atoms_obj.symbols)

            positions = self.atoms_obj.positions.flatten().tolist()

            data = {'chemical json': 0,
                    'name': name,
                    'formula': formula,
                    'atoms': {'elements': {'number': numbers,
                                           'type': symbols},
                              'coords': {'3d': positions}
                              },
                    'properties': {'cohesive energy': self.CE,
                                   'excess energy': self.EE,
                                   'shape': self.shape.lower()}
                    }
            with open(path, 'w') as fidw:
                json.dump(data, fidw, indent=4)
        # else save geometry file
        else:
            # get ase Atoms object
            atom = self.atoms_obj.copy()
            atom.info['shape'] = self.nanoparticle.shape
            atom.info['CE'] = self.CE
            atom.info['EE'] = self.EE
            atom.info['Smix'] = self.smix
            atom.info['runtype'] = 'ga-opt'
            # atom.info['composition'] = self.composition
            atom.write(path)
        return True

    def show(self):
        """
        Shows nanoparticle using ase.visualize.view
        """
        ase.visualize.view(self.atoms_obj)


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

    polymetallic_results = db.orm.relationship('PolymetallicResults',
                                               backref='nanoparticle')
    # used to attach bond_list to data entry
    # does not store bond_list in DB
    bonds_list = None
    _bonds_list = None
    num_bonds = None
    atoms_obj = None
    _atoms_obj = None

    def __init__(self, shape, num_atoms, num_shells=None):
        self.shape = shape
        self.num_atoms = num_atoms
        self.num_shells = num_shells

    def __len__(self):
        return self.num_atoms

    @property
    def atoms_obj(self):
        return self.get_atoms_obj()

    @property
    def bonds_list(self):
        return self.get_bonds_list()

    def get_atoms_obj(self):
        """
        Builds NP in the form of a Cu NP ase.Atoms object
        - stores under self.atoms_obj property
        - returns atoms_obj
        """
        if self._atoms_obj is None:
            self._atoms_obj = ase.Atoms([ase.Atom('Cu', (i.x, i.y, i.z))
                                         for i in self.atoms])
        return self._atoms_obj

    def get_atoms_obj_skel(self):
        """
        DEPRECATED: use get_atoms_obj method
        Builds NP in the form of a Cu NP ase.Atoms object
        - stores under self.atoms_obj property
        - returns atoms_obj
        """
        return self.get_atoms_obj()

    def get_diameter(self):
        """
        """
        return abs(self.atoms_obj.positions[:, 0].max() -
                   self.atoms_obj.positions[:, 0].min()) / 10

    def get_bonds_list(self):
        if self._bonds_list is None:
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'data', 'bond_lists',
                                self.shape, '%i.npy' % self.num_shells)

            if os.path.isfile(path):
                self._bonds_list = np.load(path)
            else:
                # Ensure directory actually exists before we save to it
                if not os.path.exists(os.path.dirname(path)):
                    os.makedirs(os.path.dirname(path))
                self._bonds_list = adjacency.build_bonds_arr(self.atoms_obj)
                np.save(path, self._bonds_list)

        if self.num_bonds is None:
            self.num_bonds = len(self._bonds_list) // 2

        return self._bonds_list

    def load_bonds_list(self):
        """
        DEPRECATED: use get_bonds_list
        """
        return self.get_bonds_list()


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
