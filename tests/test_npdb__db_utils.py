import pytest
import numpy as np
from ce_expansion.npdb import db_utils
import ase.units


def test_sort_2metals__none_returns_none_tuple():
    assert db_utils.sort_2metals(None) == (None, None)


def test_sort_2metals__str_input_returns_sorted():
    assert db_utils.sort_2metals('AuAg') == ('Ag', 'Au')


def test_sort_2metals__invalid_str_length_raises_exception():
    with pytest.raises(ValueError):
        db_utils.sort_2metals('Au')
        db_utils.sort_2metals('AuAgCu')


def test_sort_2metals__iter_input_returns_sorted():
    assert db_utils.sort_2metals(('Au', 'Ag')) == ('Ag', 'Au')


def test_sort_2metals__invalid_input_length_raises_exception():
    with pytest.raises(ValueError):
        db_utils.sort_2metals(('Au'))
        db_utils.sort_2metals(('Au', 'Ag', 'Cu'))


"""
# These will be moved to ce_expansion.ga.Pop object

def test_Smix__x_doesnt_sum_to_one_raises_exception():
    with pytest.raises(ValueError):
        db_utils.Smix([0.1])
        db_utils.Smix([0.5, 0.5, 0.5])


def test_Smix__x_non_iterable_raises_exception():
    with pytest.raises(TypeError):
        db_utils.Smix(0)
        db_utils.Smix(0.5)
        db_utils.Smix(1)


def test_Smix__0_and_1_return_0():
    assert db_utils.Smix([0, 1]) == db_utils.Smix([1, 0]) == 0


def test_Smix__bimetallic_case_returns_float():
    x = [0.5, 0.5]
    correct = -8.617333262145E-5 * 2 * (0.5 * np.log(0.5))
    assert pytest.approx(db_utils.Smix(x), correct)


def test_Smix__polymetallic_case_returns_float():
    x = [0.25, 0.25, 0.25, 0.25]
    correct = -8.617333262145E-5 * 4 * (0.25 * np.log(0.25))
    assert pytest.approx(db_utils.Smix(x), correct)
"""
