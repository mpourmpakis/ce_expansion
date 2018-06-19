#!/usr/bin/env python
from random import shuffle
import numpy
import ce_calc
from math import floor

def atomchanger(nanop, el1, el2):
    # nanop = ase Atom object
    # el1 = first element requested
    # el2 = second element requested
    nanoplist = nanop.get_chemical_symbols()
    nanonum = len(nanoplist)  # number of atoms in nanop
    # Read in data from atom object.


    totalloops = 5
    cemeanlist = [None]*21
    k=0

    for j in range(0, 101, 5):
        divider = floor(j * nanonum / 100)
        x = 0
        while x < nanonum:
            if x < divider:
                nanoplist[x] = el1
            elif x >= divider:
                nanoplist[x] = el2
            x = x + 1

        modlist = list(nanoplist)
        celist = [None] * totalloops
        for i in range(0, totalloops):
            shuffle(modlist)
            nanop.set_chemical_symbols(modlist)
            celist[i] = ce_calc.calculate_CE(nanop)
            if i == totalloops - 1:
                cemeanlist[k] = numpy.mean(celist)
                k += 1
    return cemeanlist
