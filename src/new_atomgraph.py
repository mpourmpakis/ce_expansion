import pickle
import ctypes
import interface

DEFAULT_ELEMENTS = ("Cu", "Cu")
DEFAULT_RADIUS = 2.8
with open("../data/precalc_coeffs.pickle", "rb") as precalcs:
    DEFAULT_BOND_COEFFS = pickle.load(precalcs)
    
class AtomGraph(object):
    def __init__(self, bond_list: "list",
                 bond_energies: "list"):
    '''
    TODO: Add docstring
    '''
        self.bond_list = bond_list
        self.num_bonds = len(bond_list)
        self.bond_energies = bond_energies
        
        self.num_atoms = len(set(bond_energies[:,0]))
        self.cns = np.bincount(bond_list[:,0])
        
        # Create pointers and explicit types for c interface
        self.p_bond_energies = bond_energies.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        self.long_num_atoms = ctypes.c_long(self.num_atoms)
        self.p_cns = self.cns.ctypes.data_as(ctypes.POINTER(ctypes.c_long))
        self.long_num_bonds = ctypes.c_long(self.num_bonds)
        self.p_bond_list = bond_list.ctypes.data_as(ctypes.POINTER(ctypes.c_long))
        
    def getTotalCE(self, ordering):
        '''
        TODO: Add docstring
        '''
        # Pointerize ordering
        p_ordering = ordering.ctypes.data_as(ctypes.POINTER(ctypes.c_long))
        
        return interface.pointerized_calculate_ce(self.p_bond_energies,
                                                  self.long_num_atoms,
                                                  self.p_cns,
                                                  self.long_num_bonds,
                                                  self.p_bond_list,
                                                  self.p_ordering)
        
        