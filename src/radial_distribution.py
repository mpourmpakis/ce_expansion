#!/usr/bin/env python

# Radial distribution calculator for NPs

def calculate_radial_distribution(coordinates, 
                                  ordering)
  """
  Given a set of atomic coordinates, and the chemical ordering string, will
  calculate radial distributions of the atoms in the cluster, at a given
  resolution. This should be O(n) in time complexity.
  
  Args:
  coordinates (np.array): N x 3 set of coordinates in the atoms. Indices must
                          correspond with the indices of chemical ordering.
  ordering (np.array): N-length vector containing the chemical ordering string

  Returns:
  A 2*N array. Row 0 is the radial coordinates for kind0. Likewise for row 1.
  """

  # Take average of all coordinates for center
  

  # Calculate distance of all atoms to center
