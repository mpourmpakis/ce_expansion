##Useful Data
This directory contains some data useful to the BCM. It 
is design with modularity in mind, such that new data can be
added or removed easily.
* bulkdata.csv
    * Contains bulk cohesive energy data
    * Kittel, C. Introduction to Solid State Physics, 8th edition. Hoboken, NJ: John Wiley & Sons, Inc, 2005.
* cndata.csv
    * Contains bulk coordination number data.
    * Currently only for FCC metals, which are CN12 in the bulk.
* experimental_hbe.csv
  * The preferred source of experimental heterolyitc bond dissociation energy.
  * Morse, M. D., *Clusters of transition-metal atoms.* Chem. Rev. 1986, 86 (6), 1049-1109.
* estimated_hbe.csv
  * If no experimental heterolytic bond dissociation energy is present in `experimental_hbe.csv`, we look now in this
    table.
  * Miedema, A. R., *Model predictions of the dissociation energies of homonuclear and heteronuclear diatomic models.*
    Faradaday Symp. Chem. Soc. 1980, 14, 136-148.