#!/usr/bin/env python

import os
import test
import src.graph_representations
import unittest
import pickle
import ase.cluster


class TestFiles(unittest.TestCase):
    def test_files(self):
        print("\nChecking for existence of sources files...")
        for filename in ("src/ce_calc.py",
                         "src/main.py",
                         "src/ce_shuffle.py",
                         "src/cn_counter.py",
                         "src/graph_representations.py",
                         "src/randomizer.py",
                         "src/structure_gen.py"):
            try:
                self.assertTrue(os.path.exists(filename))
                print(filename + " exists.")
            except AssertionError:
                print(filename + " does not exist.")
                raise


class TestGraphRep(unittest.TestCase):
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


# TODO: make this work

#  def test_adj_matrix(self):
#    print "\nChecking generated adjacency matrix..."
#    current_matrix = src.graph_representations.buildAdjacencyMatrix(self.stored_cluster)
#    self.assertEqual(current_matrix, self.stored_adj_mat)
#  def test_adj_list(self):
#    print "\nChecking generated adjacency list..."
#    current_list = src.graph_representations.buildAdjacencyList(self.stored_cluster)
#    self.assertEqual(current_list, self.stored_adj_list)

if __name__ == "__main__":
    unittest.__main__()
