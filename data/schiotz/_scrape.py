#!/usr/bin/env python

import requests

def pullRecord(filename, address):
 """
 A record from the database presented in the following article:
   Larson, P. M.; Jacobsen, K. W.; Schiotz, J. Rich Ground-state chemical
   Ordering in Nanoparticles: Exact Solution of a Model for Ag-Au Clusters.
   Phys. Rev. Lett. 2018. 120, 256101.
   DOI: 10.1103/PhysRevLett.120.256101
 Database URL:
   https://cmrdb.fysik.dtu.dk/agau309

 Args:
   filename (str): The filename that will be written
   address (str): The URL of the XYZ file
 """
 print("Sending GET request to " + address)
 response = requests.get(address)
 record = response.content.decode("utf-8")
 print("Successful. Writing record to " + filename)
 with open(filename, "w") as outp:
   for line in record:
     outp.write(line)

for i in range(0,310):
 filename = "Ag" + str(309-i) + "Au" + str(i) + ".xyz"
 address = "https://cmrdb.fysik.dtu.dk/agau309/xyz/" + str(i)
 pullRecord(filename, address)
