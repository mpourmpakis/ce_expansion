#! /usr/bin/env python

# Tool for finding coordination numbers (cns) of metal nanoparticles (mnps) with arbitrary shapes
# Original version published in:
#     Yan, Z.; Taylor, M. G.; Mascareno, A.; Mpourmpakis, G.
#     "Size-, Shape-, and Composition-Dependent Model for Metal Nanoparticle Stability Prediction"
#     Nano Lett 2018, 18 (4), 2696-2704
# Minor modifications made for readability and interoperability with our new code

import ase
from ase.io import read, write
from ase.data import vdw_radii, chemical_symbols
from ase.atoms import *
from ase.calculators.neighborlist import *
import numpy as np
import sys

# Input: .xyz file of mnp (call this script with a .xyz file as the first argument)
# Output: modified .xyz file with average cn and radii scales along in the header 2 extra columns
# extra column 1 = cn value for every atom, extra column(s) 2 = neighbor list of every atom

# Notes:
# This code assigns a bimetallic mnp as amorphous if the ratio of vdw_radii of 2 elements is < 1.1.
######################################
# For amorphous structures, the code will keep decreasing atomic scales until 1 atom has cn>12, except
# when no smaller atoms are in the bulk (i.e. a Zr-core structure for a CuZr MNP)
# the maxinum number of atoms with cn>12 allowed to be num_total_atoms^(1/3).
# The upper limit for  maximum cn is 13.
# In general, scales of the atoms with more cn>12 deviations are preferentailly reduced, except
# for structures with all larger atoms on surface/subsurface, the larger atom will take priority in cutting.
##############################################################
# if a mnp is not amorphous the code will keep decreasing the scale until no atom will have cn>12
##############################################################
# for pure metals, the start scale is 0.875 instead of 1 to get more accurate cn results.
# so the scale starting at 0.875 works for all elements and structures tested thusfar.
##############################################################
# Vdw_radii for metals taken from:
# Hu, S. Z.; Zhou, Z. H.; Robertson, B. E., Consistent approaches to van der Waals radii for the
# metallic elements. Zeitschrift Fur Kristallographie 2009, 224 (8), 375-383.
##############################################################

if len(sys.argv) < 1:
    print 'Error: Not enough arguments passed!'
    sys.exit()

count_0 = 0
newvdw = [0., 0., 0., 2.14, 1.69, 1.68, 0., 0., 0.,
          0., 0., 2.38, 2., 1.92, 1.93, 0., 0., 0.,
          0., 2.52, 2.27, 2.15, 2.11, 2.07, 2.06, 2.05, 2.04,
          2., 1.97, 1.96, 2.01, 2.03, 2.05, 2.08, 0., 0.,
          0., 2.61, 2.42, 2.32, 2.23, 2.18, 2.17, 2.16, 2.13,
          2.1, 2.1, 2.11, 2.18, 2.21, 2.23, 2.24, 0., 0.,
          0., 2.75, 2.59, 2.43, 0., 0., 0., 0., 0.,
          0., 0., 0., 0., 0., 0., 0., 0., 0.,
          2.23, 2.22, 2.18, 2.16, 2.16, 2.13, 2.13, 2.14, 2.23,
          2.27, 2.37, 2.38, 2.49, 0., 0., 0., 0., 0.,
          2.45, 0., 2.41, 0., 0., 0., 0., 0., 0., 0.]

# Step_size parameter determines how much the radii will be shifted when CN exceptions occur
step_size = -0.001
# Difference between changes in scales for more frequent element that has higher cn than 12
difference = 0.3


def cns(neiglist):
    cns = []
    complete_nl = []  # include neighborlist of all atoms
    for i in range(len(atoms1)):
        neighs = neiglist.get_neighbors(i)
        neighs = neighs[0].tolist()
        neighs = [x for x in neighs if x != i]
        # print neighs
        cns.append(len(neighs))
        complete_nl.append(neighs)
    # print final_nl
    return np.array(cns), np.array(complete_nl)  # turn into an array


for item in range(1, len(sys.argv)):
    # Set a scaling so that the cns code will automatically adjust
    # outflag used to determine if a scale has been set for both elements that makes all cns < 12
    outflag = 0
    nl = None
    moleculename = sys.argv[item]
    atoms1 = read(moleculename)
    atomic_numbers = atoms1.get_atomic_numbers()
    num_atoms = len(atomic_numbers)
    max_more_num = int(num_atoms ** (1. / 3))  # max num of atoms allowed with CN>12
    # min and max anums determine the identify of the bimetallics
    min_anum = min(atomic_numbers)
    max_anum = max(atomic_numbers)
    # The following line are calculating ratio of vdw_radii of elements
    vdw_radi_1 = newvdw[min_anum]
    vdw_radi_2 = newvdw[max_anum]
    # Decide "larger" atom by comparison of vdw_radii of elements
    if vdw_radi_1 > vdw_radi_2:
        large_atom = min_anum
        small_atom = max_anum
    else:
        large_atom = max_anum
        small_atom = min_anum
    min_vdw_radi = min(vdw_radi_1, vdw_radi_2)
    max_vdw_radi = max(vdw_radi_1, vdw_radi_2)
    ratio_vdw_r = max_vdw_radi / min_vdw_radi
    # Determine if pure metal MNP - set initial scales to 0.875
    # scale_1 assigned to the smaller element in bimetallic MNP
    # scale_2 assigned to the larger element in bimetallic MNP
    if ratio_vdw_r == 1:
        scale_1 = 0.875
        scale_2 = 0.875
    else:
        scale_1 = 1
        scale_2 = 1
    ####### New Section ###################
    while outflag == 0:  # While used to continue loop until outflag is true
        cutoffs = []
        # Generate cutoffs by atomic_numbers
        for i in range(len(atomic_numbers)):
            temp = atomic_numbers[i]
            if temp == min_anum:
                cutoffs.append(scale_1 * newvdw[temp])
            else:
                cutoffs.append(scale_2 * newvdw[temp])
        # update neighborlist and calculate new cns
        nl = NeighborList(cutoffs, bothways=True, skin=0.0)
        nl.update(atoms1)
        cordns, final_nl = cns(nl)  # final_nl = complete_nl
        # countmin/max are counters for the number of atoms which have cns>12
        count_small = 0
        count_large = 0
        less = 0
        more = 0
        too_much = 0  # atom with CN>13
        bulk_large = 0
        bulk_small = 0
        for i in range(len(cordns)):
            # Check if the cn of atom i is greater than 12 or less than 3
            if cordns[i] < 3:
                less += 1
            if cordns[i] > 13:
                too_much += 1
            if cordns[i] > 12:
                # Check element of atom i that has deviated from normal CN range
                more += 1
                if atomic_numbers[i] == small_atom:
                    count_small += 1
                else:
                    count_large += 1
            if cordns[i] >= 12 and atomic_numbers[i] == large_atom:
                bulk_large += 1  # looking for large atoms in the bulk
            if cordns[i] >= 12 and atomic_numbers[i] == small_atom:
                bulk_small += 1
        # check if any of the counters were incremented (i.e. if any atoms were CN>12 or CN<3)
        if ratio_vdw_r >= 1.1:  # when ratio>=1.1, the structure is "amporphous":
            if bulk_small == 0:  # when all small atoms are not in bulk
                if (count_large + count_small) > max_more_num or too_much > 0:
                    scale_2 += step_size
                    scale_1 += step_size * difference
                else:
                    outflag = 1
            elif bulk_large == 0:  # No large atoms in the bulk (all large atoms on surface)
                if (count_large + count_small) > 0:  # No overcutting will happen in this case, to cut down to 0
                    scale_2 += step_size
                    scale_1 += step_size * difference
                else:
                    outflag = 1
            elif (count_large + count_small) > max_more_num or too_much > 0:
                # If any are, increment their scale by the the step_size, favor "larger" element
                if count_large >= count_small:
                    scale_2 += step_size
                    scale_1 += step_size * difference
                else:
                    scale_1 += step_size
                    scale_2 += step_size * difference
            else:
                outflag = 1
        # set outflag=1 to exit while loop!
        elif ratio_vdw_r == 1.0:  # Only true if this is a pure metal MNP, reduce scale factor
            if (count_large + count_small) > 0:
                scale_2 += step_size
                scale_1 = scale_2
            else:
                outflag = 1
        else:  # Structure is not over distorted
            if (count_large + count_small) > 0:
                # if any are, increment their scale by the the step_size, favor "larger" element
                if count_large >= count_small:
                    scale_2 += step_size
                    scale_1 += step_size * difference
                else:
                    scale_1 += step_size
                    scale_2 += step_size * difference
            else:
                outflag = 1
    ################################################
    # Read and write out the _cns.xyz file
    file1 = open(moleculename, 'r')
    lines = file1.readlines()
    file1.close()
    outname = moleculename.replace(".xyz", "", 1)
    file2 = open('{}_cns.xyz'.format(outname), 'w')
    count = 0
    for line in lines:
        if count == 0:
            file2.write(line)
            count += 1
        elif count == 1:
            file2.write(
                'Average CN={} , Scale {} = {} , Scale {} = {} \n'.format(cordns.mean(), chemical_symbols[small_atom], scale_1,
                                                                          chemical_symbols[large_atom], scale_2))
            count += 1
        else:
            vect = line.split()
            a = float(cordns[count - 2])
            b = final_nl[count - 2]
            vect.append('{}'.format(a))
            vect.append('{}\n'.format(b))
            out = ' '.join(vect)
            file2.write(out)
            count += 1
    file2.close()
    # Write out the name, scales, and average coordination number!
    print '{}  Scale {} = {},  Scale {} ={}, Avg CN= {}\n'.format(outname, chemical_symbols[small_atom], scale_1,
                                                                  chemical_symbols[large_atom], scale_2, cordns.mean())
