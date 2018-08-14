#!/usr/bin/env python

import ase.neighborlist
import numpy as np


def build_adjacency_matrix(atoms_object, radius_dictionary={("Cu","Cu"):3}):
    """
    Sparse matrix representation from an ase atoms object.

    Args:
    atoms_object (ase.Atoms): An ASE atoms object representing the system of interest
    radius_dictionary (dict): A dictionary with the atom-atom radii at-which a bond is considered a
                              bond

    Returns:
    (np.ndarray): A numpy array representing the sparse matrix of the ase object
    """
    # Construct the list of bonds
    sources, destinations = ase.neighborlist.neighbor_list("ij",atoms_object, radius_dictionary)
    # Generate the matrix
    adjacency_matrix = np.zeros((len(atoms_object),len(atoms_object)))
    for bond in zip(sources, destinations):
        adjacency_matrix[bond[0], bond[1]] += 1

    return adjacency_matrix

def build_adjacency_list(atoms_object, radius_dictionary={("Cu","Cu"):3}):
  """
    Adjacency list representation for an ase atoms object.

    Args:
    atoms_object (ase.Atoms): An ASE atoms object representing the system of interest
    radius_dictionary (dict): A dictionary with the atom-atom radii at-which a bond is considered a
                              bond

    Returns:
    (np.ndarray): A numpy array representing the adjacency list of the ase object

  """

  # Construct the list of bonds
  sources, destinations = ase.neighborlist.neighbor_list("ij",atoms_object, radius_dictionary)
  bonds = np.array(zip(sources, destinations))
  # Sort along first column, to ensure when we slice the array it's cut at the correct places
  # Mergesort has a slightly better worst-case time complexity than quicksort or heapsort, and is stable
  sorted_destinations = destinations[bonds[:,0].argsort(kind='mergesort')]

  # Figure out how the list of bonds will be sliced, and slice it
  bins = np.bincount(sources)
  splitting = np.zeros(len(bins), dtype=int)
  for count, item in enumerate(bins):
    if count == 0:
      splitting[count] = item
    else:
      splitting[count] = item + splitting[count-1]

  # Slice the list of bonds to get the adjacency list
  adjacency_list = np.split(sorted_destinations,splitting)

  # Check that the final entry is an empty list, otherwise something weird happened
  if len(adjacency_list[-1]) != 0:
    raise ValueError("The following atoms have bonds yet do not appear to be bound to any item: " + str(adjacency_list[-1]))
  else:
    return np.delete(adjacency_list,-1)
