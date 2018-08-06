#!/usr/bin/env python

from __future__ import division, print_function
import ase
import numpy as np
import pickle

def find_equiv_positions(old_positions, new_positions):
  """
  Finds equivalent positions between two structures, specifically following a symmetry operation that does result in a change of the coordinate system.

  Args:
    old_positions (ase.Atoms):  the positions of the object prior to transformation.
    new_positions (ase.Atoms):  the positions of the object following its transformation.

  Returns:
    np.ndarray: A 1-D array containing the mapping information for that particular transformation.
      For example, [0, 4, 2, 3, 1] indicates that atom 0 remained in its current position, atom 1 moved to the position of atom 4,
      atom 2 remained in its current position, atom 3 remained in its current position, and atom 4 moved to the position of atom 1.

  """
  for old_index, original in enumerate(old_positions):
    min_distance = 999
    tag = None
    for tf_index,transformed in enumerate(new_positions):
      distance = np.linalg.norm(original.position - transformed.position)
      if distance < min_distance:
        min_distance = distance
        new_tag = tf_index
    new_positions[old_index].tag = new_tag
  return new_positions.get_tags()

def generate_rotations(structure, axis, symmetry):
  """
  Given an axis of rotation and the symmetry about that axis, generates all transformations stemming from that rotation.

  Args:
    structure (ase.Atoms): The structure to be rotated.
    axis (np.ndarray): A vector (preferably normalized) which the rotation will be carried out about.
    symmetry (int/str): The degree of rotational symmetry about that axis, expressed as "360/symmetry."
      If a str is supplied in the form "CN", a lookup is performed to find the symmetry in a table.
      For example, supplying "C5" would result in a lookup to get the value 5, and indicates 5-fold symmetry about the supplied axis.

  Returns:
    dict: A dictionary of the form {rotation:positions}, where rotation refers to the length of the rotation (in degrees) and
      positions refers to the 1-D array containing the mapping information about that particular transformation. See the docstring
      for find_equiv_positions() for more information about this array.
  """
  old_structure = structure.copy()
  for index,atom in enumerate(structure):
    old_structure[index].tag = index
  rotation_increments = 360/symmetry
  transforms = {}
  for i in range(1,symmetry+1):
    rotation = rotation_increments * i
    new_structure = old_structure.copy()
    new_structure.rotate(rotation, v=axis)
    mapping = find_equiv_positions(old_structure, new_structure)
    transforms[rotation] = mapping
  return transforms

def get_all_rotations(structure, axes, symmetry):
  """
  Given a set of axes and the symmetry of that set, as well as an atomic structure, generates all transformations stemming from that set.

  Args:
    structure (ase.Atoms): The structure to be rotated.
    axis (np.ndarray): A vector (preferably normalized) which the rotation will be carried out about.
    symmetry (int/str): The degree of rotational symmetry about that axis, expressed as "360/symmetry."
      If a str is supplied in the form "CN", a lookup is performed to find the symmetry in a table.
      For example, supplying "C5" would result in a lookup to get the value 5, and indicates 5-fold symmetry about the supplied axis.

  Returns:
    dict: A dictionary containing an index of all axes of symmetry, each containing a dictionary of the form {rotation:positions}. See
      the docstring for generate_rotations() for more information about this dict.
  """
  if isinstance(symmetry,str):
    symmetry_dict = {"C2":2,
                     "C3":3,
                     "C5":5}
    symmetry = symmetry_dict[symmetry]
  axis_dict = {}
  print("Getting rotations...", end="")
  for count, axis in enumerate(axes):
    transforms = generate_rotations(structure, axis, symmetry)
    axis_dict[count] = transforms
  print("Done.")
  print("Checking sanity...")
  sanity_check(axis_dict)
  print("Done.")
  return axis_dict

def sanity_check(tf_positions):
  """
  A sanity-checker, to ensure that the positions returned are reasonable.
    If a set of positions contains a rotation of 360 degrees about an axis, it checks that the positions are equivalent to range(0,len(positions)),
    e.g. [0, 1, 2, ..., N].
    If a set of positions contains another rotation, it just checks that no duplicates have appeared in the list.

    If a sanity-check is failed, the item which failed the check is printed to the console.
  """
  for tf in tf_positions:
    for rotation in tf_positions[tf]:
      failed = False
      if rotation == 360:
        for count in range(0,len(tf_positions[tf][rotation])):
          if (count != tf_positions[tf][rotation][count]):
            print("Failed a sanity check: " + str(count) + " != " + str(tf_positions[tf][rotation][count]))
            print(tf_positions[tf][rotation])
            failed = True
            break
      else:
        collected = []
        for position in tf_positions[tf][rotation]:
          if position in collected:
            print("Failed a sanity check: observed " + str(position) + " >1 times in array.")
            print(tf, rotation, tf_positions[tf][rotation])
            failed = True
            break
          else:
            collected.append(position)
  if failed == True:
    print("Failed at least one sanity check. Check log for more details.")
