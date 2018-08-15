#!/usr/bin/env python

import os
import test
import src
import unittest

class TestFiles(unittest.TestCase):
  def test_files(self):
    print "\nChecking for existence of sources files..."
    for filename in ("src/ce_calc.py",
                     "src/ce_main_file.py",
                     "src/ce_shuffle.py",
                     "src/cn_counter.py",
                     "src/graph_representation.py",
                     "src/randomizer.py",
                     "src/structure_gen.py"):
      try:
        self.assertTrue(os.path.exists(filename))
        print filename + " exists."
      except AssertionError:
        print filename + " does not exist."

if __name__ == "__main__":
  unittest.__main__()
