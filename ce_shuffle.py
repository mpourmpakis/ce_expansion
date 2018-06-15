#!/usr/bin/env python
from random import shuffle
import time
from ase import Atoms, Atom

import ase, ase.neighborlist
import numpy as np
from pandas import DataFrame
import ce_calc


def atomchanger(nanop, el1, el2, shape, shells):
    # nanop = ase Atom object
    # el1 = first element requested
    # el2 = second element requested
    nanoplist = nanop.get_chemical_symbols()
    nanonum = len(nanoplist)  # number of atoms in nanop
    # Read in data from atom object.
    name = el1 + el2 + shape + str(shells) + '.xlsx'
    print nanonum
    totalrows = 0
    counter2 = 10
    celist = [None] * counter2 * (nanonum + 1)
    el1list = [el1] * counter2 * (nanonum + 1)
    perc1list = [None] * counter2 * (nanonum + 1)
    numb1list = [None] * counter2 * (nanonum + 1)
    numb2list = [None] * counter2 * (nanonum + 1)
    el2list = [el2] * counter2 * (nanonum + 1)
    perc2list = [None] * counter2 * (nanonum + 1)
    excesslist = [None] * counter2 * (nanonum + 1)
    flist = [None] * counter2 * (nanonum + 1)
    el1num = nanonum

    j = 1
    while j == 1:
        x = 0
        while x < nanonum:
            if x < el1num:
                nanoplist[x] = el1
            else:
                nanoplist[x] = el2
            x = x + 1


        modlist = list(nanoplist)
        formula = nanop.get_chemical_formula()
        perc1 = float(el1num) / nanonum * 100
        perc2 = 100 - perc1

        for i in range(0, counter2):
            perc1list[totalrows] = perc1
            perc2list[totalrows] = perc2
            numb1list[totalrows] = el1num
            numb2list[totalrows] = nanonum - el1num
            flist[totalrows] = formula

            shuffle(modlist)
            nanop.set_chemical_symbols(modlist)
            celist[totalrows] = ce_calc.calculate_CE(nanop)
            totalrows = totalrows + 1
        if el1num == 0:
            j = 0
        el1num -= 1

    df = DataFrame({'Element 1': el1list, 'Percentage 1': perc1list, 'Element 2': el2list, 'Percentage 2': perc2list,
                    'Number 1': numb1list, 'Number 2': numb2list, 'Cohesive Energy': celist, 'Excess Energy': i})

    df.to_excel(name, sheet_name='sheet1',
                columns=['Element 1', 'Percentage 1', 'Number 1', 'Element 2', 'Percentage 2', 'Number 2',
                         'Cohesive Energy', 'Excess Energy', ], index=False)
