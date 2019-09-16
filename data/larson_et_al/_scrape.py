#!/usr/bin/env python

import sys

import ase.io
import requests

#####
sys.exit("Delete this line to re-run database scrape")
#####

def pullRecord(address):
    """
    A record from the database presented in the following article:
      Larson, P. M.; Jacobsen, K. W.; Schiotz, J. Rich Ground-state chemical
      Ordering in Nanoparticles: Exact Solution of a Model for Ag-Au Clusters.
      Phys. Rev. Lett. 2018. 120, 256101.
      DOI: 10.1103/PhysRevLett.120.256101
    Database URL:
      https://cmrdb.fysik.dtu.dk/agau309

    Args:
       address (str): The URL of the XYZ file
    """
    print("Sending GET request to " + address)
    response = requests.get(address)
    record = response.content.decode("utf-8")

    temp = open("tempfile", "w")

    for line in record:
        temp.write(line)

    temp.seek(0)
    atoms = ase.io.read("tempfile")

    filename = atoms.get_chemical_formula() + ".xyz"
    print("Successful. Writing record to " + filename)
    ase.io.write(filename, atoms)

    temp.close()


for i in range(1, 311):
    address = "https://cmrdb.fysik.dtu.dk/agau309/xyz/" + str(i)
    pullRecord(address)
