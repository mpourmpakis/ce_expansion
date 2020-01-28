#!/usr/bin/env python

import ctypes
import os
import sys

# Selects whether to use the debug libs or not
DEBUG_MODE = False

# Other options
DEFAULT_NUM_ELEMENTS = 2  # Polymetallicity of the system; 2=bimetallic. Only works for bimetallics at the moment.
DEFAULT_MAX_COORDINATION = 12  # Maximum possible coordination number

bin_directory = os.sep.join(os.path.realpath(__file__).split(os.sep)[:-1])

# Set the correct library for the given platform
if DEBUG_MODE:
    if sys.platform in ['win32', 'cygwin']:
        dll = ('_lib_debug.dll')
    else:
        dll = ('_lib_debug.so')
else:
    if sys.platform in ['win32', 'cygwin']:
        dll = ('_lib.dll')
    elif sys.platform in ['darwin']:
        dll = ('_libmac.so')
    else:
        dll = ('_lib.so')

# Load the library
dll_location = os.sep.join(bin_directory.split(os.sep) + [dll])
_libCalc = ctypes.CDLL(dll_location)

# Function return type
_libCalc.calculate_ce.restype = ctypes.c_double
_libCalc.calculate_mixing.restype = ctypes.c_int

# Argument types
_libCalc.calculate_ce.argtypes = [ctypes.POINTER(ctypes.c_double),  # bond_energies
                                  ctypes.c_long,  # num_atoms
                                  ctypes.POINTER(ctypes.c_long),  # cns
                                  ctypes.c_long,  # num_bonds
                                  ctypes.POINTER(ctypes.c_long),  # adj_table
                                  ctypes.POINTER(ctypes.c_long)  # id_array
                                  ]
_libCalc.calculate_mixing.argtypes = [ctypes.c_long,  # num atoms
                                      ctypes.c_long,  # num bonds
                                      ctypes.POINTER(ctypes.c_long),  # adj_table
                                      ctypes.POINTER(ctypes.c_long),  # id_array
                                      ctypes.POINTER(ctypes.c_long)  # Holder for return value
                                      ]


def pointerized_calculate_ce(bond_energies, num_atoms, cns, num_bonds, adjacency_table, id_string):
    """
    Version of the calculate_CE function which takes in pre-converted c/pointer objects

    Args:
    bond_energies (c_double pointer): A 2x2x12 table of bond energies
    num_atoms (c_long): Number of atoms in the system
    cns (c_long pointer): A 1D array of the coordination numbers in the sytem, of length num_atoms
    num_bonds (c_long): Number of bonds in the system
    adjacency_table (c_long pointer): An Nx2 table of bonds in the system, of length num_bonds
    id_string (c_long pointer): An array representing which elements are in the nanoparticle, of length num_atoms
    """
    return _libCalc.calculate_ce(bond_energies, num_atoms, cns, num_bonds, adjacency_table, id_string)


def calculate_ce(bond_energies, num_atoms, cns, num_bonds, adjacency_table, id_string):
    """
    Version of the calculate_CE function which creates the pointers on its own. This is slower than the pointerized
    version.

    Args:
    bond_energies (np.array): A 2x2x12 table of bond energies
    num_atoms (int): Number of atoms in the system
    cns (np.array): A 1D array of the coordination numbers in the sytem, of length num_atoms
    num_bonds (int): Number of bonds in the system
    adjacency_table (np.array): An Nx2 table of bonds in the system, of length num_bonds
    id_string (np.array): An array representing which elements are in the nanoparticle, of length num_atoms
    """
    p_bond_energies = bond_energies.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    p_cns = cns.ctypes.data_as(ctypes.POINTER(ctypes.c_long))
    p_adjacency_table = adjacency_table.ctypes.data_as(ctypes.POINTER(ctypes.c_long))
    p_id_string = id_string.ctypes.data_as(ctypes.POINTER(ctypes.c_long))

    return _libCalc.calculate_ce(p_bond_energies,
                                 ctypes.c_long(num_atoms),
                                 p_cns,
                                 ctypes.c_long(num_bonds),
                                 p_adjacency_table,
                                 p_id_string
                                 )


def pointerized_calculate_mixing(num_atoms, num_bonds, adjacency_table, id_string, return_array):
    """
    Version of the calculate_mixing function that takes in all the pointers pre-done

    Args:
        num_atoms (int): Number of atoms in the system
        num_bonds (int): Number of bonds in the system
        adjacency_table (np.array): An Nx2 tableof bonds in the system, of length num_bonds
        id_string (np.array): An array representing which elements are in the NP, of length num_atoms
        return_array (int): A length-2 array to hold return values.
                              -First value is the number of A-A bonds.
                              -Second value is the number of B-B bonds.
                              -Third value is the number of A-B bonds.

    Returns:
        None. Return_array is changed in place.
    """
    error_code = _libCalc.calculate_mixing(num_atoms, num_bonds, adjacency_table, id_string, return_array)

    assert (error_code == 0)
    return None


def calculate_mixing(num_atoms, num_bonds, adjacency_table, id_string):
    """
    Version of the calculate_mixing function which creates the pointers on its own.
    
    Args:
        num_atoms (int): Number of atoms in the system
        num_bonds (int): Number of bonds in the system
        adjacency_table (np.array): An Nx2 tableof bonds in the system, of length num_bonds
        id_string (np.array): An array representing which elements are in the NP, of length num_atoms
        
    Returns:
        A length-3 holding return values.
          -First value is the number of A-A bonds.
          -Second value is the number of B-B bonds.
          -Third value is the number of A-B bonds.
    """
    p_adjacency_table = adjacency_table.ctypes.data_as(ctypes.POINTER(ctypes.c_long))
    p_id_string = id_string.ctypes.data_as(ctypes.POINTER(ctypes.c_long))
    return_array = np.zeros(3, dtype=ctypes.c_long)
    p_return_array = return_array.ctypes.data_as(ctypes.POINTER(ctypes.c_long))

    error_code = _libCalc.calculate_mixing(ctypes.c_long(num_atoms),
                                           ctypes.c_long(num_bonds),
                                           p_adjacency_table,
                                           p_id_string,
                                           p_return_array)
    assert (error_code == 0)
    return return_array


if __name__ == "__main__":
    # Just some test stuff
    import numpy as np

    bond_array = np.ones([DEFAULT_NUM_ELEMENTS, DEFAULT_NUM_ELEMENTS, DEFAULT_MAX_COORDINATION])
    cns = np.array([12, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6])  # 13-atom icosahedron
    num_atoms = cns.shape[0]
    bondList = np.array([(0, 1), (0, 12), (0, 11), (0, 10), (0, 8), (0, 7), (0, 9), (0, 5), (0, 4), (0, 3), (0, 2),
                         (0, 6), (1, 8), (1, 12), (1, 11), (1, 2), (1, 0), (1, 6), (2, 10), (2, 9), (2, 8), (2, 6),
                         (2, 1), (2, 0), (3, 12), (3, 11), (3, 7), (3, 5), (3, 4), (3, 0), (4, 10), (4, 9), (4, 7),
                         (4, 5), (4, 3), (4, 0), (5, 12), (5, 10), (5, 0), (5, 4), (5, 3), (5, 6), (6, 12), (6, 10),
                         (6, 5), (6, 2), (6, 1), (6, 0), (7, 11), (7, 9), (7, 8), (7, 4), (7, 3), (7, 0), (8, 11),
                         (8, 9), (8, 7), (8, 2), (8, 1), (8, 0), (9, 7), (9, 8), (9, 10), (9, 4), (9, 2), (9, 0),
                         (10, 9), (10, 6), (10, 5), (10, 4), (10, 2), (10, 0), (11, 12), (11, 8), (11, 7), (11, 3),
                         (11, 1), (11, 0), (12, 6), (12, 5), (12, 3), (12, 1), (12, 0), (12, 11)])
    adjacency_table = np.array([[]])
    num_bonds = bondList.shape[0]
    id_string = np.array([1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1])
    print(id_string.dtype)
    test_3Darray = np.array([
        [
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        ],
        [
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        ]
    ]
    )
    print("Testing a 13-atom icosahedron with fake coefficients:")
    print(calculate_ce(test_3Darray, num_atoms, cns, num_bonds, bondList, id_string))

    print("Testing the mixing")
    print(calculate_mixing(num_atoms, num_bonds, bondList, id_string))
