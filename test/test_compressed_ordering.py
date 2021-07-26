from ce_expansion.npdb import db_inter, datatables

"""
Test all results in BimetallicResults to ensure that
the ordering compression algo is working correctly
"""

res = db_inter.get_bimet_result()
for i, r in enumerate(res):
    print(f' testing {i:06d}  {r.num_atoms:05d} atoms', end='\r')
    test = datatables.BimetallicResults(r.metal1, r.metal2, r.shape,
                                        r.num_atoms, r.diameter, r.n_metal1,
                                        r.n_metal2, r.CE, r.EE, r.ordering)
    assert (test.ordering == r.ordering).all()
    assert test.n_metal2 == r.n_metal2 == sum(map(int, test.ordering))
    assert len(r.ordering) == r.num_atoms
