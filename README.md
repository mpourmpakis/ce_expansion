![CE Expansion Logo](/images/logo.png)


## James Dean, Michael Cowan, Jonathan Estes, Mahmoud Ramadan, Giannis Mpourmpakis

Explore bimetallic nanoparticles with our program, which implements the bond-centric model of Yan<sup>1</sup> in conjunction with a genetic algorithm.

© 2019 [CANELa Lab](https://mpourmpakis.com/), University of Pittsburgh


### Installation

To install this program, make sure you've fulfilled the following requirements:

* Python 3.7 or greater is installed  
* Numpy and ASE are both present with your Python installation (or environment)

Simply download this repository, and run `src/ga.py` to begin searching potential NPs.

### Overview

This repository contains a variety of files to facilitate the exploration of the morphological space a nanoparticle can exist in. We utilize the Bond-Centric Model of Yan Et al<sup>1</sup> to rapidly calculate the cohesive energy of a nanoparticle, which can then be used to determine its excess energy.

Nanoparticles can be described in terms of their size (how many atoms there are), shape (where the atoms are), and chemical ordering (what the chemical identity of each atom is). This is a fairly large search space. Even if we hold size and shape constant, and only look at bimetallic systems, the number of possible NPs grows exponentially: ignoring symmetry, the number of possible chemical orderings for an arbitrary NP is 2<sup>N</sup>. For a 55-atom nanoparticle with some arbitrary shape, that's over 36 quadrillion possible chemical orderings for just one bimetallic pair. Even if we restrict ourselves to the d-block metals, if we want to investigate that same 55-atom nanoparticle, that's 3.6E16 orderings/pair times (40 * 40 / 2) pairs, which is over 2.8 *quin*tillion possible NPs. If we want to explore even a fraction of that search space, we need a fast algorithm to estimate CE.

But finding a fast algorithm is only one part of the problem. Even if we calculate one nanoparticle per CPU-nanosecond, exploring this space would still require over 900 CPU-years of computing time. Although this is still feasible with current supercomputers, that's just one possible NP morphology andone possible NP size. NPs come in a variety of sizes and shapes, and polymetallic systems are also of interest, hence a brute-force search of the chemical space is simply not possible with current technology. Instead of a brute-force search, we resort to a genetic algorithm-based approach.

In our genetic algorithm, the "DNA" of the NP is represented as a string of 1's and 0's representing its chemical ordering. For example, if we had a 13-atom NP containing 6 copper atoms and 6 silver atoms, here are a few possible chemical orderings:

* 111111000000
* 111010001100
* 101010101010

Essentially what we're doing here, is we're taking each atom in the NP, and arbitrarily assigning it an index. It doesn't matter how that index is assigned, as long as every index is sequential, and is consistently assigned. See the section titled "The AtomGraph" for more detailed information on exactly how it's assigned. This method is particularly convenient, because it can easily be extended to ternary or even polymetallic systems.

Hence, for a particular nanoparticle size and shape, we now have a vector that represents the identity of each atom. This strand of "DNA" can then be mutated and bred within a genetic algorithm to try and find a minimum-energy chemical ordering for the NP's morphology. This converges quite rapidly with our code, and as a result we can very quickly find the lowest-energy chemical ordering for an arbitrary NP composition.

### The Atomgraph

A nanoparticle can be thought of as a graph, with each atom as a vertex and each bond as an edge. In the file `src/adjacency.py`, we include tools to generate an adjacency list, an adjacency matrix, and an edge list (a bond list) for an arbitrary nanoparticle. Because of its ease of use and popularity, we've written these functions to take ASE<sup>2</sup> atoms objects as their arguments.

In the file `src/atomgraph.py` is the AtomGraph class. It takes in a bond list, and information on what a 1 or 0 means (e.g. what element is a 1, what element is a 0, etc), and then exists as a calculator. If the method CalculateTotalCE is called with a chemical ordering, it will then calculate the total CE using the Yan<sup>1</sup> model. In order to speed up the calculation, we wrote a C library (`src/lib.c`) to implement this model, along with a python interface (`src/interface.py`) which gets called by the AtomGraph. We have libraries compiled already for Linux (.so) and Windows (.dll) in the `bin` directory. We've also included a `makefile` for Linux users.

### The Genetic Algorithm
Todo: Write this section

### References
1. Yan, Z.; Taylor, M. G.; Mascareno, A.; Mpourmpakis, G. Size-, Shape-, and Composition-Dependent Model for Metla Nanoparticle Stability Prediction.  Nano Lett. 2018. 18, 4, 2696-2704. DOI: https://doi.org/10.1021/acs.nanolett.8b00670
2. Larson, A. H.; Mortensen, J. J.; Blomqvist, J.; Castelli, I. E.; Christensen, R.; Dulak, M.; Friis, J.; Groves, M. N.; Hammer, B.; Hargus, J. The atomic simulation environment - a Python library for working with atoms. J. Phys. Condens. Matter 2017. 29, 273002. DOI: https://doi.org/10.1088/1361-648X/aa680e , Github repository: https://github.com/rosswhitfield/ase
