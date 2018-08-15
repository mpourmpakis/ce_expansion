#!/usr/bin/env python

import os
import unittest
import test

class testTest(unittest.TestCase):
  def testFiles(self):
    print "\nPerforming meta-test. Testing that test files exist...."
    for filename in ("test/test_project_structure.py",
                     "test/test_src.py",
                     "test/test_tools.py",
                     test.PRECOMP_COEFFS,
                     test.PRECOMP_ADJ_MAT,
                     test.PRECOMP_ADJ_LIST,
                     test.PRECOMP_CLUSTER):
      try:
        self.assertTrue(os.path.exists(filename))
        print filename + " exists."
      except AssertionError:
        print filename + " does not exist."
        raise

if __name__ == "__main__":
  unittest.__main__()
