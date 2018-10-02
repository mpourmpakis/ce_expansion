#!/usr/bin/env python

import ase.neighborlist
import numpy as np

DEFAULT_ELEMENTS = ("Cu", "Cu")
DEFAULT_RADIUS = 2.8
DEFAULT_BOND_COEFFS = {'Au': {'Au': [None,  # 0
                                     -1.0998522628062373,  # 1
                                     -0.77771299333365917,  # 2
                                     -0.63500000000000012,  # 3
                                     -0.54992613140311863,  # 4
                                     -0.49186888496834202,  # 5
                                     -0.44901280605345778,  # 6
                                     -0.41570508089956554,  # 7
                                     -0.38885649666682959,  # 8
                                     -0.36661742093541244,  # 9
                                     -0.34780382401578053,  # 10
                                     -0.33161793459560446,  # 11
                                     -0.31750000000000006],  # 12

                              'Ag': [None,  # 0
                                     -1.4078108963919838,  # 1
                                     -0.99547263146708376,  # 2
                                     -0.81280000000000019,  # 3
                                     -0.70390544819599188,  # 4
                                     -0.62959217275947776,  # 5
                                     -0.57473639174842595,  # 6
                                     -0.53210250355144395,  # 7
                                     -0.49773631573354188,  # 8
                                     -0.4692702987973279,  # 9
                                     -0.44518889474019907,  # 10
                                     -0.42447095628237369,  # 11
                                     -0.40640000000000009],  # 12

                              'Cu': [None,  # 0
                                     -2.5956513402227199,  # 1
                                     -1.8354026642674355,  # 2
                                     -1.4986000000000004,  # 3
                                     -1.29782567011136,  # 4
                                     -1.1608105685252872,  # 5
                                     -1.0596702222861605,  # 6
                                     -0.98106399092297469,  # 7
                                     -0.91770133213371774,  # 8
                                     -0.86521711340757335,  # 9
                                     -0.82081702467724205,  # 10
                                     -0.7826183256456265,  # 11
                                     -0.74930000000000019]  # 12
                              },
                       'Ag': {
                           'Au': [None,  # 0
                                  -0.61314598587938263,  # 1
                                  -0.43355968447262255,  # 2
                                  -0.35400000000000004,  # 3
                                  -0.30657299293969131,  # 4
                                  -0.27420722091148514,  # 5
                                  -0.25031580054003788,  # 6
                                  -0.23174739943062392,  # 7
                                  -0.21677984223631128,  # 8
                                  -0.20438199529312753,  # 9
                                  -0.1938937853568288,  # 10
                                  -0.18487047062495113,  # 11
                                  -0.17700000000000002],  # 12

                           'Ag': [None,  # 0
                                  -0.85159164705469814,  # 1
                                  -0.60216622843419798,  # 2
                                  -0.49166666666666675,  # 3
                                  -0.42579582352734907,  # 4
                                  -0.38084336237706268,  # 5
                                  -0.34766083408338594,  # 6
                                  -0.3218713880980888,  # 7
                                  -0.30108311421709899,  # 8
                                  -0.28386388235156607,  # 9
                                  -0.26929692410670669,  # 10
                                  -0.25676454253465436,  # 11
                                  -0.24583333333333338],  # 12

                           'Cu': [None,  # 0
                                  -1.1837123894060304,  # 1
                                  -0.83701105752353522,  # 2
                                  -0.68341666666666678,  # 3
                                  -0.5918561947030152,  # 4
                                  -0.5293722737041171,  # 5
                                  -0.48324855937590649,  # 6
                                  -0.44740122945634342,  # 7
                                  -0.41850552876176761,  # 8
                                  -0.39457079646867682,  # 9
                                  -0.37432272450832232,  # 10
                                  -0.35690271412316954,  # 11
                                  -0.34170833333333339]  # 12
                       },
                       'Cu': {
                           'Au': [None,  # 0
                                  0.36269143910492291,  # 1
                                  0.25646157606939873,  # 2
                                  0.2094,  # 3
                                  0.18134571955246145,  # 4
                                  0.1622005425391666,  # 5
                                  0.14806815998046308,  # 6
                                  0.1370844786462504,  # 7
                                  0.12823078803469937,  # 8
                                  0.12089714636830763,  # 9
                                  0.11469310354158178,  # 10
                                  0.10935558347136938,  # 11
                                  0.1047],  # 12

                           'Ag': [None,  # 0
                                  -0.61456049403889723,  # 1
                                  -0.43455989278425899,  # 2
                                  -0.35481666666666672,  # 3
                                  -0.30728024701944862,  # 4
                                  -0.27483980819136566,  # 5
                                  -0.25089327107800691,  # 6
                                  -0.2322820332617021,  # 7
                                  -0.2172799463921295,  # 8
                                  -0.20485349801296573,  # 9
                                  -0.19434109211212472,  # 10
                                  -0.18529696088204259,  # 11
                                  -0.17740833333333336],  # 12

                           'Cu': [None,  # 0
                                  -1.0074762197358971,  # 1
                                  -0.71239326685944104,  # 2
                                  -0.58166666666666678,  # 3
                                  -0.50373810986794854,  # 4
                                  -0.45055706260879619,  # 5
                                  -0.41130044439017521,  # 6
                                  -0.38079021846180672,  # 7
                                  -0.35619663342972052,  # 8
                                  -0.33582540657863236,  # 9
                                  -0.31859195428217163,  # 10
                                  -0.30376550964269278,  # 11
                                  -0.29083333333333339]  # 12
                       }
                       }


def buildAdjacencyMatrix(atoms_object, radius_dictionary={DEFAULT_ELEMENTS: DEFAULT_RADIUS}):
    """
    Sparse matrix representation from an ase atoms object.

    Args:
    atoms_object (ase.Atoms): An ASE atoms object representing the system of interest
    radius_dictionary (dict): A dictionary with the atom-atom radii at-which a bond is considered a
                              bond. If no dict is supplied, Cu-Cu bonds of max-len 2.8 are assumed.

    Returns:
    np.ndarray : A numpy array representing the sparse matrix of the ase object
    """
    # Construct the list of bonds
    sources, destinations = ase.neighborlist.neighbor_list("ij", atoms_object, radius_dictionary)
    # Generate the matrix
    adjacency_matrix = np.zeros((len(atoms_object), len(atoms_object)))
    for bond in zip(sources, destinations):
        adjacency_matrix[bond[0], bond[1]] += 1

    return adjacency_matrix


def buildAdjacencyList(atoms_object, radius_dictionary={DEFAULT_ELEMENTS: DEFAULT_RADIUS}):
    """
      Adjacency list representation for an ase atoms object.

      Args:
      atoms_object (ase.Atoms): An ASE atoms object representing the system of interest
      radius_dictionary (dict): A dictionary with the atom-atom radii at-which a bond is considered a
                                bond. If no dict is supplied, Cu-Cu bonds of max-len 2.8 are assumed.

      Returns:
      np.ndarray : A numpy array representing the adjacency list of the ase object

    """

    # Construct the list of bonds
    sources, destinations = ase.neighborlist.neighbor_list("ij", atoms_object, radius_dictionary)
    bonds = np.array(zip(sources, destinations))
    # Sort along first column, to ensure when we slice the array it's cut at the correct places
    # Mergesort has a slightly better worst-case time complexity than quicksort or heapsort, and is stable
    sorted_destinations = destinations[bonds[:, 0].argsort(kind='mergesort')]

    # Figure out how the list of bonds will be sliced, and slice it
    bins = np.bincount(sources)
    splitting = np.zeros(len(bins), dtype=int)
    for count, item in enumerate(bins):
        if count == 0:
            splitting[count] = item
        else:
            splitting[count] = item + splitting[count - 1]

    # Slice the list of bonds to get the adjacency list
    adjacency_list = np.split(sorted_destinations, splitting)

    # Check that the final entry is an empty list, otherwise something weird happened
    if len(adjacency_list[-1]) != 0:
        raise ValueError(
            "The following atoms have bonds yet do not appear to be bound to any item: " + str(adjacency_list[-1]))
    else:
        return np.delete(adjacency_list, -1)


class AtomGraph(object):
    def __init__(self, adj_list, colors, kind0, kind1, coeffs=DEFAULT_BOND_COEFFS):
        """
        A graph representing the ase Atoms object to be investigated. First axis is the atom index, second axis containst bonding information.
        First entry of the second axis corresponds to a 1/0 representing the atomic kind
        Second entry of the second axis corresponds to the indices of the atoms that atom is bound to

        Args:
        adj_list (np.array) : A numpy array containing adjacency information. Assumed to be produced by the buildAdjacencyList function
        colors (np.array) : A numpy array containing a binary representation of the molecule
        kind0 (str) : Atomic symbol indicating what a "0" in colors means
        kind1 (str) : Atomic symbol indicating what a "1" in colors means
        coeffs (dict) : A dictionary of the various bond coefficients we have precalculated, using coeffs.py. Defaults to the global DEFAULT_BOND_COEFFS.


        Attributes:
        adj_list (np.array) : A numpy array containing adjacency information
        colors (np.array) : A numpy array containing a binary representation of the molecule
        symbols (tuple) : Atomic symbols indicating what the binary representations of the elements in self.colors means

        """

        self.adj_list = adj_list
        self.colors = colors
        self.coeffs = coeffs
        self.symbols = (kind0, kind1)

    def __len__(self):
        return len(self.adj_list)

    def __getitem__(self, atom_key):
        return (self.symbols[self.colors[atom_key]], self.adj_list[atom_key])

    def getCN(self, atom_key):
        """
        Returns the coordination number of the given atom.

        Args:
        atom_key (int) : Index of the atom of interest
        """
        return len(self[atom_key][1])

    def getAllCNs(self):
        """
        Returns a numpy array containing all CN's in the cluster.
        """
        return np.array([entry.size for entry in self.adj_list])

    def getHalfBond(self, atom_key, bond_key):
        """
        Returns the half-bond energy of a given bond for a certain atom.

        Args:
        atom_key (int) : Index of the atom of interest
        bond_key (int) : Index of the bond of interest

        Returns:
        float : The half-bond energy of that bond at that atom, in units of eV
        """
        atom1 = self.colors[atom_key]
        atom2 = self.colors[self.adj_list[atom_key]]
        return self.coeffs[atom1][atom2][self.getCN(atom_key)]

    def getLocalCE(self, atom_key):
        """
        Returns the sum of half-bond energies for a particular atom.

        Args:
        atom_key : Index of the atom of interest

        Returns:
        float : The sum of all half-bond energies at that atom, in units of eV
        """
        local_CE = 0
        for bond_key in range(len(self.adj_list[atom_key])):
            local_CE += self.getHalfBond(atom_key, bond_key)
        return local_CE

    def getTotalCE(self):
        """
        Returns the cohesive energy of the cluster as a whole.

        Returns
        float : The CE, in units of eV
        """
        total_CE = 0
        for atom_key in len(self):
            total_CE += self.getLocalCE(atom_key)
        return total_CE
