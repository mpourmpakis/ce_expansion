#!/usr/bin/env python

# Need the following numpy arrays:
    # double representing bond energies. 2x2x12. Basicaly the dict, but in numpy form
    # the adjacency table, an Nx2 list of ints of which atoms are bound to which
    # char array for the ID string.
    # TODO: Check how much faster the code is if we use an int array instead of a char array, to
    # avoid needing conversions

   #!/usr/bin/env python

import ctypes
import numpy as np

DEFAULT_NUM_ELEMENTS = 2
DEFAULT_MAX_COORDINATION = 12

_libCalc = ctypes.CDLL('../bin/_lib.so')

# Create python wrappers
''' Test functions here to get the C interace right
 def ctypes_fib(a):
     return _libcalc.fib(ctypes.c_int(a))
    
 def char_to_int(character):
     return _libCalc.char_to_int(ctypes.c_char(character))
    
 def print_array(array, size):
     array_pointer = array.ctypes.data_as(ctypes.POINTER(ctypes.c_long))
     return _libCalc.print_array(array_pointer, ctypes.c_long(size))
     
 def print_2D(array):
     array_pointer = array.ctypes.data_as(ctypes.POINTER(ctypes.c_long))
     dim1, dim2 = array.shape
     return _libCalc.print_2D(ctypes.c_long(dim1),ctypes.c_long(dim2),
                        array_pointer
                       )
   
 def print_3D(array):
     array_pointer = array.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
     dim1, dim2, dim3 = array.shape
     return _libCalc.print_3D(ctypes.c_long(dim1), ctypes.c_long(dim2), ctypes.c_long(dim3),
                        array_pointer
                        )
   
'''

# Actual functions
_libCalc.calculate_ce.restype = ctypes.c_double;
def pointerized_calculate_ce(bond_energies, num_atoms, cns, num_bonds, adjacency_table, id_string):
    '''
    TODO: Docstring for pointerized calculate_CE
    '''
    return _libCalc.calculate_ce(bond_energies, num_atoms, cns, num_bonds, adjacency_table, id_string)
def calculate_ce(bond_energies, num_atoms, cns, num_bonds, adjacency_table, id_string):
    '''
    TODO: Docstring for calculate_CE
    '''
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
                                                   

                                
if __name__ == "__main__":
    bond_array = np.ones([DEFAULT_NUM_ELEMENTS, DEFAULT_NUM_ELEMENTS, DEFAULT_MAX_COORDINATION])
    cns = np.array([12, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]) # 13-atom icosahedron
    num_atoms = cns.shape[0]
    bondList = np.array([(0, 1),(0, 12),(0, 11),(0, 10),(0, 8),(0, 7),(0, 9),(0, 5),(0, 4),(0, 3),(0, 2),
             (0, 6),(1, 8),(1, 12),(1, 11),(1, 2),(1, 0),(1, 6),(2, 10),(2, 9),(2, 8),(2, 6),
             (2, 1),(2, 0),(3, 12),(3, 11),(3, 7),(3, 5),(3, 4),(3, 0),(4, 10),(4, 9),(4, 7),
             (4, 5),(4, 3),(4, 0),(5, 12),(5, 10),(5, 0),(5, 4),(5, 3),(5, 6),(6, 12),(6, 10),
             (6, 5),(6, 2),(6, 1),(6, 0),(7, 11),(7, 9),(7, 8),(7, 4),(7, 3),(7, 0),(8, 11),
             (8, 9),(8, 7),(8, 2),(8, 1),(8, 0),(9, 7),(9, 8),(9, 10),(9, 4),(9, 2),(9, 0),
             (10, 9),(10, 6),(10, 5),(10, 4),(10, 2),(10, 0),(11, 12),(11, 8),(11, 7),(11, 3),
             (11, 1),(11, 0),(12, 6),(12, 5),(12, 3),(12, 1),(12, 0),(12, 11)])
    num_bonds = bondList.shape[0]
    # single_character_string = np.array([b'1',b'1',b'1',b'0',b'0',b'0',b'1',b'1',b'1',b'0',b'0',b'0',b'1'])
    id_string = np.array([1,1,1,0,0,0,1,1,1,0,0,0,1])
    # test_2Darray = np.array([[1,2],[3,4]])
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
    print("Testing a 13-atom icosahedron:")
    print(calculate_ce(test_3Darray, num_atoms, cns, num_bonds, bondList, id_string))
