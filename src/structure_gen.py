#!/usr/bin/env python3

import ase.cluster
import ase.lattice


def cube(num_layers: "int", kind: "str" = "Cu") -> "ase.Atoms":
    """
    Creates an FCC cube with faces on the {100} family of planes.

    :param num_layers: Number of unit cells along each side of the cube.
    :type num_layers: int
    :param kind: The element making up the skeleton. Defaults to "Cu"
    :type kind: str

    :return: An ASE atoms object containing the cube skeleton
    """

    lattice = ase.lattice.cubic.FaceCenteredCubic(kind, size=[num_layers] * 3)
    cube = ase.build.cut(lattice, extend=1.01)
    return cube


def sphere(num_layers: "int", kind: "str" = "Cu", unit_cell_length: "float" = 3.61) -> "ase.Atoms":
    """
    Inscribes a sphere inside a cube and makes it a nanoparticle. Perfect symmetry not guaranteed.

    :param num_layers: The size of the lattice containing the inscribed sphere.
    :type num_layers: int
    :param kind: The element making up the skeleton. Defaults to "Cu"
    :type kind: str
    :param unit_cell_length: The edge-length of the unit cell.
    :type unit_cell_length: float

    :return: An ASE atoms object containing the sphere skeleton.
    """

    # Create the cube
    trimmed_cube = cube(num_layers, kind)

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


def elongated_pentagonal_bipyramid(num_layers: "int", kind: "str" = "Cu") -> "ase.Atoms":
    """
    Creates an elongated pentagonal bipyramidal NP.

    :param num_layers: Number of layers in the EPB
    :type num_layers: int
    :param kind: These are monometallic. What element will be used. Defaults to "Cu"
    :type kind: str

    :return: An ASE atoms object containing the elongated pentagonal bipyramid skeleton.
    """
    return ase.cluster.Decahedron("Cu", num_layers, num_layers, 0)


def cuboctahedron(num_layers: "int", kind: "str" = "Cu") -> "ase.Atoms":
    """
    Creates a cuboctahedral NP.

    :param num_layers: Number of layers in the cuboctahedron.
    :type num_layers: int
    :param kind: These are monometallic. What element will bee used. Defaults to "Cu".
    :type kind: str

    :return: An ASE atoms object containing the cuboctahedron skeleton.
    """
    return ase.cluster.Octahedron(kind, 2 * num_layers + 1, cutoff=num_layers)
