from ce_expansion.npdb import db_inter
import random


def test_get_bimet_result__return_metals_matches():
    # get 3 random metal pairs found in BimetallicResults table
    metals = random.sample(set(db_inter.build_metal_pairs_list()), 3)
    for m in metals:
        res = db_inter.get_bimet_result(metals=m, lim=5, return_list=True)
        assert all(r.metal1 == m[0] and r.metal2 == m[1] for r in res)


def test_get_bimet_result__return_shape_matches():
    for shape in ['icosahedron', 'cuboctahedron']:
        res = db_inter.get_bimet_result(shape=shape, lim=5, return_list=True)
        assert all(r.shape == shape for r in res)


def test_get_bimet_result__return_num_atoms_matches():
    for n in [13, 55]:
        res = db_inter.get_bimet_result(num_atoms=n, lim=5, return_list=True)
        assert all(r.num_atoms == n for r in res)


def test_get_bimet_result__return_num_shells_matches():
    for s in [1, 2]:
        res = db_inter.get_bimet_result(num_shells=s, lim=5, return_list=True)
        assert all(r.nanoparticle.num_shells == s for r in res)


def test_get_bimet_result__return_n_metal1_matches():
    for n1 in [30, 40]:
        res = db_inter.get_bimet_result(n_metal1=n1, lim=5, return_list=True)
        assert all(r.n_metal1 == n1 for r in res)


def test_get_bimet_result__return_only_bimet_results():
    res = db_inter.get_bimet_result(only_bimet=True, lim=20, return_list=True)
    assert all(0 < r.n_metal1 < r.num_atoms for r in res)


def test_get_nanoparticle__return_shape_matches():
    for shape in ['icosahedron', 'cuboctahedron']:
        res = db_inter.get_nanoparticle(shape=shape, lim=5, return_list=True)
        assert all(r.shape == shape for r in res)


def test_get_nanoparticle__return_num_atoms_matches():
    for n in [13, 55]:
        res = db_inter.get_nanoparticle(num_atoms=n, lim=5, return_list=True)
        assert all(r.num_atoms == n for r in res)


def test_get_nanoparticle__return_num_shells_matches():
    for s in [1, 2]:
        res = db_inter.get_nanoparticle(num_shells=s, lim=5, return_list=True)
        assert all(r.num_shells == s for r in res)


def test_get_polymet_result__return_empty_list():
    res = db_inter.get_polymet_result(num_atoms=-1)
    assert res == []
