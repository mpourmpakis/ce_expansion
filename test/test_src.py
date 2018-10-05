#!/usr/bin/env python3

import os
import pickle
import unittest

import ase.cluster

import test


class TestFiles(unittest.TestCase):
    def test_files(self):
        print("\nChecking for existence of sources files...")
        for filename in ("src/example.py",
                         "src/atomgraph.py",
                         "src/structure_gen.py"):
            try:
                self.assertTrue(os.path.exists(filename))
                print(filename + " exists.")
            except AssertionError:
                print(filename + " does not exist.")
                raise


class TestGraphRep(unittest.TestCase):
    # TODO Add additional checks for the graph representation object
    def setUp(self):
        with open(test.PRECOMP_CLUSTER, "r") as stored:
            self.stored_cluster = pickle.load(stored)
        with open(test.PRECOMP_ADJ_MAT, "r") as stored:
            self.stored_adj_mat = pickle.load(stored)
        with open(test.PRECOMP_ADJ_LIST, "r") as stored:
            self.stored_adj_list = pickle.load(stored)

    def test_ase(self):
        print("\nTesting ASE cluster generation...")
        current_cluster = ase.cluster.Icosahedron("Cu", 3)


# TODO: Add tests for the chemical ordering

if __name__ == "__main__":
    unittest.__main__()
