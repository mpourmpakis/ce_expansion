from functools import lru_cache

import ase.cluster
import ase.lattice

from ce_expansion.npdb import db_inter


@lru_cache
def num_atoms_in_shell(shell: int) -> int:
    """Calculates number of atoms in a magic number NP shell
    - icosahedron shells
    - cuboctahedron shells
    - elongated-pentagonal-bipyramid shells
    """
    if shell == 0:
        return 1
    return 10 * (shell + 1) * (shell - 1) + 12


@lru_cache
def shell_to_magic_number(num_shells: int) -> int:
    """Calculates the number of atoms in magic number:
    - icoshaedron NPs
    - cuboctahedron NPs
    - elongated-pentagonal-bipyramid NPs
    """
    return sum(num_atoms_in_shell(s) for s in range(num_shells + 1))


def build_structure_sql(shape: str, num_shells: int):
    """
    Creates NP of specified shape and size (based on num_shells)

    Args:
    shape (str): shape of NP
                 NOTE: currently supported methods (found in NPBuilder)
                 - cuboctahedron
                 - elongated-trigonal-pyramid
                 - fcc-cube
                 - icosahedron
                 - sphere
    num_shells (int): Number of shells in NP
                      e.g. icosahedron with 2 shells makes a 55-atom object
                      ( 1 in core (shell_0) + 12 in shell_1 + 42 in shell_2)

    Kargs:
    build_bonds_arr (bool): if True, builds bonds list attribute
                             (default: True)

    Returns:
    (npdb.datatables.Nanoparticle)

    Raises:
            NotImplementedError: given shape has not been implemented
    """
    if num_shells <= 0:
        raise ValueError('Can only build NPs with at least one shell.')

    nanop = db_inter.get_nanoparticle(shape, num_shells=num_shells, lim=1)
    atom = None
    if nanop:
        atom = nanop.get_atoms_obj_skel()
    else:
        try:
            # build atom object
            atom = getattr(NPBuilder, shape.replace('-', '_'))(num_shells)
        except:
            raise NotImplementedError('%s has not been implemented' % shape)

        # insert nanoparticle into DB
        nanop = db_inter.insert_nanoparticle(atom, shape, num_shells)

    return nanop


class NPBuilder(object):
    """
    Static class that contains methods to build NPs of various shapes

    Args:
    num_shells (int): Number of shells in NP

    KArgs:
    kind (str): What element will be used for the monometallic NPs
                (DEFAULT: Cu)

    Returns:
    (ase.Atoms): the NP skeleton"""

    @staticmethod
    def cuboctahedron(num_shells, kind="Cu"):
        """
        Creates a cuboctahedral NP.

        Args:
        num_shells (int): Number of shells in NP

        KArgs:
        kind (str): What element will be used for the monometallic NPs
                    (DEFAULT: Cu)

        Returns:
        (ase.Atoms): the NP skeleton
        """
        assert num_shells >= 0
        if num_shells == 0:
            return ase.Atoms(kind)
        return ase.Atoms(ase.cluster.Octahedron(kind, 2 * num_shells + 1,
                                                cutoff=num_shells), pbc=False)

    @staticmethod
    def elongated_pentagonal_bipyramid(num_shells,
                                       kind="Cu"):
        """
        Creates an elongated-pentagonal-bipyramidal NP.

        Args:
        num_shells (int): Number of shells in NP

        KArgs:
        kind (str): What element will be used for the monometallic NPs
                    (DEFAULT: Cu)

        Returns:
        (ase.Atoms): the NP skeleton
        """
        num_shells += 1
        return ase.Atoms(
            ase.cluster.Decahedron("Cu", num_shells, num_shells, 0),
            pbc=False)

    @staticmethod
    def fcc_cube(num_units, kind="Cu"):
        """
        Creates an FCC-cube with faces on the {100} family of planes.

        Args:
        num_units (int): Number of primitive FCC units across each side of NP
                        - NOTE: units share a face

        KArgs:
        kind (str): What element will be used for the monometallic NPs
                    (DEFAULT: Cu)

        Returns:
        (ase.Atoms): the NP skeleton
        """
        assert num_units >= 0
        if num_units == 0:
            return ase.Atoms(kind)
        atom = ase.Atoms(ase.cluster.FaceCenteredCubic('Cu', [(1, 0, 0),
                                                              (0, 1, 0),
                                                              (0, 0, 1)],
                                                       [num_units] * 3))
        return atom

    @staticmethod
    def icosahedron(num_shells, kind="Cu"):
        """
        Creates an icosahedral NP.

        Args:
        num_shells (int): Number of shells in NP

        KArgs:
        kind (str): What element will be used for the monometallic NPs
                    (DEFAULT: Cu)

        Returns:
        (ase.Atoms): the NP skeleton
        """
        assert num_shells >= 0
        return ase.Atoms(ase.cluster.Icosahedron(kind, num_shells + 1),
                         pbc=False)

    # WIP - sphere is not perfectly symmetric
    @staticmethod
    def _sphere(num_layers, kind="Cu",
               unit_cell_length=3.61):
        """
        Inscribes a sphere inside a cube and makes it a nanoparticle.
        NOTE: Perfect symmetry not guaranteed.

        :param num_layers: The size of the lattice containing the inscribed sphere.
        :type num_layers: int
        :param kind: The element making up the skeleton. Defaults to "Cu"
        :type kind: str
        :param unit_cell_length: The edge-length of the unit cell.
        :type unit_cell_length: float

        :return: An ASE atoms object containing the sphere skeleton.
        """
        raise NotImplementedError
        # Create the cube
        trimmed_cube = fcc_cube(num_layers, kind)

        # Simple geometry
        center = trimmed_cube.positions.mean(0)
        cutoff_radius = num_layers * unit_cell_length / 1.99
        distance_list = map(ase.np.linalg.norm,
                            ase.geometry.get_distances(trimmed_cube.get_positions(), p2=center)[1])

        # Build the sphere using atoms that are within the cutoff
        sphere = ase.Atoms()
        for atom, distance in zip(trimmed_cube, distance_list):
            if distance <= cutoff_radius:
                sphere += atom

        return sphere
