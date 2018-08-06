#!/usr/bin/env python
import ase
import ase.cluster
import math
import ase.io
import shell_separator
import icos_rotations
import pickle
import numpy


icos = shell_separator.shell_separator(4)
icos.center(about=0)

c5_axes = numpy.array(pickle.load(open('cn6.pickle')))
c2_axes = numpy.array(pickle.load(open('cn8.pickle')))
c3_axes = numpy.array(pickle.load(open('cn9.pickle')))

axis_dict_C2 = icos_rotations.get_all_rotations(icos,c2_axes,'C2')
axis_dict_C3 = icos_rotations.get_all_rotations(icos,c3_axes,'C3')
axis_dict_C5 = icos_rotations.get_all_rotations(icos,c5_axes,'C5')

print axis_dict_C2
print axis_dict_C3
print axis_dict_C5
#Prints dictionaries of the C2, C3, and C5 axes transformations.