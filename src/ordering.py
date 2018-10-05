#!/usr/bin/env python3

import ase


class Ordering(object):
    def __init__(self,
                 atoms: "ase.Atoms",
                 kind0: "str",
                 kind1: "str"):
        """
        Binary-valued string representation of a nanoparticle's chemical ordering. Initialized to all 0's.

        Args:
        atoms (ase.Atoms) : An atoms object the string is tied to.
                           Changes to the string result in changes to the particle.
        kind0 (str) : What element a "0" in the string represents.
        kind1 (str) : What element a "1" in the string represents.

        Attributes:
        atoms (ase.Atoms) : An atoms object the string is tied to.
                           Changes to the string result in changes to the particle.
        kinds (tuple) : A tuple containing the atomic symbol for a "0" value, and the atomic symobl for a "1" value
        string (str) : A binary-valued string representation of a nanoparticle's chemical ordering.
        """
        self._atoms = atoms
        self.kinds = (kind0, kind1)
        self._string = '0' * len(atoms)

        self.update()

    def __len__(self):
        return len(self._string)

    def __getitem__(self, item):
        return self._string[item]

    def __setitem__(self, key, value):
        if value != 0 or value != 1:
            raise ValueError("Expected a 0 or 1 for value, got %i" % value)
        if key > len(self._string):
            raise KeyError("Key of size %i exceeds string length %i" % (key, len(self._string)))
        self._string = self._string[:key] + key + self._string[key + 1:]
        self.update()

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index >= len(self._string):
            raise StopIteration
        result = self._string[self.index]
        self.index += 1
        return result

    def update(self):
        """
        Updates the atomic symbols of the bound atoms object to reflect the current string.

        :return: None
        """
        for value, atom in zip(self._string, self._atoms):
            atom.symbol = self.kinds[value]
        return None
