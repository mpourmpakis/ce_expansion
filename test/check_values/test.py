#!/usr/bin/env python

import os
import sys

# This is a kludge
cwd = os.getcwd()
path = "../../src/"
sys.path.append(path)

import atomgraph
import adjacency

outp = open(cwd + "results.csv")
outp.write("a")
for i in range(2,20):
  pass
outp.close()
