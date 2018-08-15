#!/usr/bin/env python
import os
import unittest

# Globals
GEN_COEFFS = "tools/gen_coeffs.py"
PRECOMP_COEFFS = "test/precomputed.pickle"

class TestDirectory(unittest.TestCase):
  """
  This test enforces the proper directory structure, and also checks that accidental file deletion has not happened.
  """
  def test_directories(self):
    path, directories, files = next(os.walk("."))
    print "\nChecking for existence of folders..."
    for folder in ("test",
                   "src",
                   "docs",
                   "tools"):
      try:
        self.assertTrue(os.path.exists(folder))
        print folder + "/ exists."
      except AssertionError:
        print "Folder " + folder + " not found."
        raise
  def test_files(self):
    print "\nChecking for existence of files..."
    for filename in ("README.md",
                     "LICENSE"):
      try:
        self.assertTrue(os.path.exists(filename))
        print filename + " exists."
      except AssertionError:
        print "File " + filename + " not found."
        raise

if __name__ == "__main__":
  unittest.main()
