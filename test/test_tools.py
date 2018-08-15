#!/usr/bin/env python

import unittest
import pickle
import tools.gen_coeffs
import test

class TestTools(unittest.TestCase):
  def setUp(self):
    with open(test.PRECOMP_COEFFS,"r") as precomp:
      self.precomputed_coeffs = pickle.load(precomp)
  def test_coeffs(self):
    print "\nTesting precomputed bond coeffs versus those prodced by gen_coeffs.py..."
    try:
      self.assertEqual(self.precomputed_coeffs, tools.gen_coeffs.precomputed_coeffs)
      print "Values verified."
    except AssertionError:
      print "Values from current script differ from those found in pickle." 

if __name__ == "__main__":
    unittest.main()

