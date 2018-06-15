#!/usr/bin/env python
from random import shuffle
import csv
import time
from ase import Atoms, Atom
from ase.cluster.icosahedron import Icosahedron
from ase.cluster.decahedron import Decahedron
from ase.cluster.cubic import FaceCenteredCubic
from math import floor
import ase, ase.neighborlist
import numpy as np
from pandas import DataFrame


def atomchanger(nanop, el1, el2, shape, shells):
    # nanop = ase Atom object
    # el1 = first element requested
    # el2 = second element requested
    nanoplist = nanop.get_chemical_symbols()
    nanonum = len(nanoplist)  #number of atoms in nanop
    print nanoplist
    # Read in data from atom object.
    name = el1 + el2 + shape + str(shells) + '.xlsx'

    totalrows = 0
    counter2 = 100
    counter3 = 101
    celist = [None]*counter2*counter3
    el1list = [el1]*counter2*counter3
    perc1list = [None]*counter2*counter3
    numb1list=[None]*counter2*counter3
    numb2list=[None]*counter2*counter3
    el2list = [el2]*counter2*counter3
    perc2list = [None]*counter2*counter3
    excesslist = [None]*counter2*counter3
    flist= [None]*counter2*counter3

    j = 1
    while j == 1:
        el1num = nanonum
        el2num = 0
        x = 0
        while x < nanonum:
            if x < el1num:
                nanoplist[x] = el1
            else:
                nanoplist[x] = el2
            x = x + 1

        modlist = list(nanoplist)

        for i in range(0, counter2):
                perc1list[totalrows] = j 
                perc2list[totalrows] = 100 - j 
                numb1list[totalrows]=((j)*nanonum/100)
                numb2list[totalrows]=((100-(j))*nanonum/100)
                #f[totalrows]=nanop.get_chemical_formula()
                shuffle(modlist)
                nanop.set_chemical_symbols(modlist)
                celist[totalrows] = calculate_CE(nanop)
                totalrows = totalrows + 1
    el1num -=1

    if el1num == 0
        j=0

    df = DataFrame({'Element 1': el1list, 'Percentage 1': perc1list, 'Element 2': el2list, 'Percentage 2': perc2list, 'Number 1': numb1list,'Number 2': numb2list, 'Cohesive Energy': celist, 'Excess Energy': i})
                        #biglist = [el1list,perc1list,el2list,perc2list,celist,excesslist]
                        #print biglist
    df.to_excel(name, sheet_name='sheet1', columns = ['Element 1', 'Percentage 1','Number 1', 'Element 2', 'Percentage 2','Number 2', 'Cohesive Energy', 'Excess Energy',], index=False)
 