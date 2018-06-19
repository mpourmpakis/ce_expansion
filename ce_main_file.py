#!/usr/bin/env python
import ce_shuffle
import structure_gen
import time
from pandas import DataFrame


# shell_nums is a list with the shell sizes
def structure(shape, shell_nums):
    atoms_list = []
    for s in shell_nums:
        if shape == 'Icosahedron':
            shell_args = [s]
        elif shape in ['Decahedron', 'Lattice', 'Sphere']:
            shell_args = [s] * 3

        atoms_list.append(structure_gen.structuregen(shape, shell_args))
        # last_atom_obj = atoms_list[-1]
        # last_atom_obj.shape = shape
        atoms_list[-1].shape = shape
        atoms_list[-1].shell_num = s

    return atoms_list


def randomizer(atoms):
    """
            replaces generic structure with selected atomic symbols; randomizes; prints CE into excel file
    """

    t = time.clock()
    meancelist = ce_shuffle.atomchanger(atoms, 'Cu', 'Ag')
    t2 = time.clock()
    print '1= ' + str(t2 - t)
    meancelist2 = ce_shuffle.atomchanger(atoms, 'Au', 'Cu')
    t2 = time.clock()
    print '2= ' + str(t2 - t)
    meancelist3 = ce_shuffle.atomchanger(atoms, 'Ag', 'Au')
    t2 = time.clock()
    meancelist.extend(meancelist2)
    meancelist.extend(meancelist3)
    print '3 = ' + str(t2 - t)
    return meancelist


def fileprint(morphlist, morphologyname, shellnums, atomform):
    el1list = ['Cu'] * 21
    el1list.extend(['Au'] * 21)
    el1list.extend(['Ag'] * 21)
    el2list = ['Ag'] * 21
    el2list.extend(['Cu'] * 21)
    el2list.extend(['Au'] * 21)
    y = el1list
    formula = []
    for i in range(0, len(shellnums)):
        formula.extend([atomform[i]] * 63)

    el1list.extend(el1list * (len(shellnums) - 1))
    el2list.extend(el2list * (len(shellnums) - 1))

    perclist1 = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100] * 3 * len(shellnums)
    perclist2 = [100, 95, 90, 85, 80, 75, 70, 65, 60, 55, 50, 45, 40, 35, 30, 25, 20, 15, 10, 5, 0] * 3 * len(shellnums)

    # print len(el1list)
    # print len(perclist1)
    # print len(el2list)
    # print len(perclist2)
    # print len(morphlist)
    # print len(formula)

    df = DataFrame({'Element 1': el1list, 'Percentage 1': perclist1, 'Element 2': el2list, 'Percentage 2': perclist2,
                    'Cohesive Energy': morphlist, 'Number of Atoms': formula})

    df.to_excel(morphologyname + '.xlsx', sheet_name='sheet1',
                columns=['Element 1', 'Percentage 1', 'Element 2', 'Percentage 2',
                         'Number of Atoms', 'Cohesive Energy', ], index=False)


def main():
    """
        runs the selected structure 
    """
    t = time.clock()
    print 'start = ' + str(t)

    shellnums = [2, 3, 4, 5]
    atomform = [None] * len(shellnums)
    morphologyname = 'Icosahedron'
    atoms_list = structure(morphologyname, shellnums)
    i = 0
    j = 0
    for atom in atoms_list:
        meanshellist = randomizer(atom)
        atomform[j] = len(atom.get_chemical_symbols())
        print atomform

        j += 1
        if i == 0:
            morphlist = meanshellist
            i = 1
        else:
            morphlist.extend(meanshellist)
    print 'Done'
    fileprint(morphlist, morphologyname, shellnums, atomform)


main()
