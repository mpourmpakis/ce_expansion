#!/usr/bin/env python

import pandas
import numpy as np


def calculate_gamma(element1: str,
                    element2: str,
                    exp: "Filename for experimental data" = "data/experimental_hbe.csv",
                    est: "Filename for theoretical data" = "data/estimated_hbe.csv") -> "tuple":
    """
    Given a pair of elements: "element1" and "element2", this function calculates the gamma coefficient from Yan et al.

    :param element1: The first element in the bimetallic pair
    :type element1: str
    :param element2: The second element in the bimetallic pair
    :type element2: str
    :param exp: Experimental bond dissociation energies. This is the preferred source of data.
    :type exp: str
    :param est: Theoretical bond dissociation energies. If no experiment is available, we get our data from here.
    :type est: str

    :return: The two gamma coefficients as two floats inside the tuple, in the order the two elements were provided.
             In other words, calling the function with element1 = "Cu" and element2 = "Ag" would return a tuple where
             the first entry is the gamma coefficient for Cu, and the second entry is the gamma coefficient for Ag.
    """
