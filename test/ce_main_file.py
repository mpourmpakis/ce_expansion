#!/usr/bin/env python
import ce_shuffle
import structure_gen
import time
import ase.io
from pandas import DataFrame 
from math import floor

 
# shell_nums is a list with the shell sizes
def structure(shape, shell_nums):
    atoms_list = []
    for s in shell_nums:
        if shape in['Icosahedron','Cuboctahedron']:
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
    meancelist1, stdlist1, totalloops, mincelist1, maxcelist1, minatomlist1, maxatomlist1 = ce_shuffle.atomchanger(atoms, 'Cu', 'Ag')
    #meancelist1, stdlist1, totalloops, mincelist1, maxcelist1, minatomlist1, maxatomlist1 = ce_shuffle.atomchanger(atoms, element1, element2)
    t2 = time.clock()
    print '1= ' + str(t2 - t)
    meancelist2, stdlist2, totaloops, mincelist2, maxcelist2, minatomlist2, maxatomlist2 = ce_shuffle.atomchanger(atoms, 'Au', 'Cu')
    t2 = time.clock()
    print '2= ' + str(t2 - t)
    meancelist3, stdlist3, totalloops, mincelist3, maxcelist3, minatomlist3, maxatomlist3= ce_shuffle.atomchanger(atoms, 'Ag', 'Au')
    t2 = time.clock()
    meancelist = meancelist1 + meancelist2 + meancelist3
    mincelist = mincelist1 + mincelist2 + mincelist3
    maxcelist = maxcelist1 + maxcelist2 + maxcelist3
    minatomlist = minatomlist1 + minatomlist2 + minatomlist3
    maxatomlist = maxatomlist1 + maxatomlist2 + maxatomlist3
    stdlist = stdlist1 + stdlist2 + stdlist3

    print '3 = ' + str(t2 - t)
    return meancelist, stdlist, totalloops, mincelist, maxcelist, minatomlist, maxatomlist


def fileprint(morphlist, morphologyname, shellnums, atomform, standarddev, smpnum, mincelist, maxcelist):
    #Prints nanoparticle data to an .xlsx file

    el1list = ['Cu'] * 21
    el1list.extend(['Au'] * 21)
    el1list.extend(['Ag'] * 21)
    el2list = ['Ag'] * 21
    el2list.extend(['Cu'] * 21)
    el2list.extend(['Au'] * 21)
    formula = []
    smplist = [smpnum]*63*len(shellnums)
    #Initializes lists to be printed to file

    for i in range(0, len(shellnums)):
        formula.extend([atomform[i]] * 63)

    el1list.extend(el1list * (len(shellnums) - 1))
    el2list.extend(el2list * (len(shellnums) - 1))

    perclist1 = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100] * 3 * len(shellnums)
    perclist2 = [100, 95, 90, 85, 80, 75, 70, 65, 60, 55, 50, 45, 40, 35, 30, 25, 20, 15, 10, 5, 0] * 3 * len(shellnums)

    #print len(el1list)
    #print len(perclist1)
    #print len(el2list)
    #print len(perclist2)
    #print len(morphlist)
    #print len(formula)
    #print len(standarddev)
    #print len(smplist)
    #print len(mincelist)
    #print len(maxcelist)

    df = DataFrame({'Element 1': el1list, 'Percentage 1': perclist1, 'Element 2': el2list, 'Percentage 2': perclist2,
                    'Average Cohesive Energy': morphlist, 'Number of Atoms': formula, 'Standard Deviation': standarddev
                    , '# of Samples': smplist, 'Minimum CE': mincelist, 'Maximum CE': maxcelist})

    df.to_excel(morphologyname + '.xlsx', sheet_name='sheet1',
                columns=['Element 1', 'Percentage 1', 'Element 2', 'Percentage 2',
                         'Number of Atoms', 'Average Cohesive Energy','Minimum CE', 'Maximum CE', 'Standard Deviation',
                         '# of Samples'], index=False)



def boundce(minatomlist, maxatomlist, shellnums, morphologyname):
    #Prints min and max CE nanoparticles at each individual composition
    
    perclist1 = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100] * 3 * len(shellnums)
    el1list = ['Cu'] * 21
    el1list.extend(['Au'] * 21)
    el1list.extend(['Ag'] * 21)
    el2list = ['Ag'] * 21
    el2list.extend(['Cu'] * 21)
    el2list.extend(['Au'] * 21)
    el1list.extend(el1list * (len(shellnums) - 1))
    el2list.extend(el2list * (len(shellnums) - 1))
    k=1
    for i in range(0, 63*len(shellnums)):
        k = shellnums[int(floor(i/63))]
        namemin = morphologyname + 'Shell_#' + str(k) + str(el1list[i]) + str(perclist1[i]) + str(el2list[i]) + str(100 - perclist1[i]) + 'min.xyz'
        namemax = morphologyname + 'Shell_#' + str(k) + str(el1list[i]) + str(perclist1[i]) + str(el2list[i]) + str(100 - perclist1[i]) + 'max.xyz'
        ase.io.write(namemin, minatomlist[i], format=None, parallel=True, append=False)
        ase.io.write(namemax, maxatomlist[i], format=None, parallel=True, append=False)




def main():
    """
        runs the selected structure 
    """
    t = time.clock()
    print 'start = ' + str(t)

    shellnums = [2]
    #Determines the number of shells that will be present in the nanoparticle

    morphologyname = 'Icosahedron'
    #Determines the morphology of the nanoparticle

    atomform = [None] * len(shellnums)

    atoms_list = structure(morphologyname, shellnums)
    i = 0
    j = 0
    for atom in atoms_list:
        meanshellist, stdlist, smpnum, mincelist, maxcelist, minatomlist, maxatomlist = randomizer(atom)
        atomform[j] = len(atom.get_chemical_symbols())
        print atomform

        j += 1
        if i == 0:
            morphlist = meanshellist
            standarddev = stdlist
            i = 1
        else:
            morphlist.extend(meanshellist)
            standarddev.extend(stdlist)
    print 'Done'
    fileprint(morphlist, morphologyname, shellnums, atomform, standarddev, smpnum, mincelist, maxcelist)
    boundce(minatomlist, maxatomlist, shellnums, morphologyname)

main()
