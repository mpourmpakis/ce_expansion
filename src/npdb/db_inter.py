import sqlalchemy as db
import db_utils
from base import Session
from datatables import BimetallicResults as BiMet
from datatables import Atoms
from datatables import Nanoparticles
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

"""
    Functions to interface with NP GA Database
    - all methods follow a specific prefix form

    Function Types:
       get_: retrieves data from DB
             - does NOT edit DB entries
             - NOTE: returns Datatable objects

     build_: uses data from DB to build desired object(s)
             - does NOT edit DB entries
             - NOTE: returns data in other forms besides Datatable objects
                     - e.g. pd.DataFrame, plt.Figure, np.array, etc.

    insert_: adds new data to DB
             - does NOT edit current DB entries
             - adds new rows to DB
    update_: updates data in existing entries
             - does edit current DB entries
             - if no current entry matches criteria, a new
               entry will be made if allow_entry=True
    remove_: removes an entry/entries from DB
"""

# global session
session = Session()

# BUILD FUNCTIONS


def build_srf_plot(metals, shape):
    """
    Creates a 3D surface plot from NP SQL database
    - plots Size vs. Shape vs. Excess Energy (EE)

    Args:
    metals (string || iterable): string(s) containing two metal elements
    shape (string): shape of the NP

    Returns:
        (plt.figure): figure of 3D surface plot
    """
    metal1, metal2 = db_utils.sort_metals(metals)

    runs = session.query(BiMet.diameter,
                         (BiMet.n_metal2 / db.cast(BiMet.num_atoms, db.Float))
                         .label('comps'),
                         BiMet.EE) \
        .filter(db.and_(BiMet.metal1 == metal1,
                        BiMet.metal2 == metal2,
                        BiMet.shape == shape)) \
        .statement
    df = pd.read_sql(runs, session.bind)

    # three parameters to plot
    size = df.diameter.values
    comps = df.comps.values
    ees = df.EE.values

    # plots surface as heat map with warmer colors for larger EEs
    colormap = plt.get_cmap('coolwarm')
    normalize = matplotlib.colors.Normalize(vmin=ees.min(), vmax=ees.max())

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    try:
        ax.plot_trisurf(comps, size, ees,
                        cmap=colormap, norm=normalize)
    except RuntimeError:
        # if not enough data to create surface, make a scatter plot instead
        ax.scatter3D(comps, size, ees,
                     cmap=colormap, norm=normalize)
    ax.set_xlabel('$X_{%s}$' % metal2)
    ax.set_ylabel('Size (nm)')
    ax.set_zlabel('EE (eV)')
    ax.set_title('%s %s %s' % (metal1, metal2, shape.title()))
    return fig


# GET FUNCTIONS


def get_all_tables(lim=10):
    """
    Returns tuple containing lists of entries from all
    datatables (at the moment just Nanoparticles
    and BimetallicResults)

    Kargs:
    lim (int): limit the number of entries returned from each
               datatable

    Returns:
        (tuple): lists of datatable objects
    """
    nps = get_nanoparticle(lim=lim)
    bi = get_bimet_result(lim=lim)
    return nps, bi


def get_bimet_result(metals=None, shape=None, num_atoms=None,
                     n_metal1=None, lim=None, return_query=False):
    """
    Returns BimetallicResults entry that matches criteria
    - if no criteria given, all data (up to <lim> amount)
      returned

    Kargs:
    metals (str || iterable): two metal elements
    shape (str): shape of NP
    num_atoms (int): number of atoms in NP
    n_metal1 (int): number of metal1 atoms in NP
    lim (int): max number of entries returned
               (default: None = no limit)
    return_query (bool): if True, return query and
                         not results

    Returns:
        (BimetallicResults)(s) if match is found else (None)
    """
    match_ls = []
    if metals:
        metal1, metal2 = db_utils.sort_metals(metals)
        match_ls.append(BiMet.metal1 == metal1)
        match_ls.append(BiMet.metal2 == metal2)
    for attr, crit in zip(['shape', 'num_atoms', 'n_metal1'],
                          [shape, num_atoms, n_metal1]):
        if crit is not None:
            match_ls.append(getattr(BiMet, attr) == crit)
    match = db.and_(*match_ls)
    qry = session.query(BiMet).filter(match)
    if return_query:
        return qry
    res = qry.limit(lim).all()
    return res if len(res) != 1 else res[0]


def get_nanoparticle(shape=None, num_atoms=None, num_shells=None,
                     lim=None, return_query=False):
    """
    Returns Nanoparticles entries that match criteria
    - if no criteria given, all data (up to <lim> amount)
      returned

    Kargs:
    shape (str): shape of NP
    num_atoms (int): number of atoms in NP
    num_shells (int): number of shells used to build NP
                      from structure_gen module
    lim (int): max number of entries returned
               (default: None = no limit)
    return_query (bool): if True, returns just the query
                         and not the results

    Returns:
        (Nanoparticles)(s) if match else (None)
    """
    match_ls = []
    for attr, crit in zip(['shape', 'num_atoms', 'num_shells'],
                          [shape, num_atoms, num_shells]):
        if crit is not None:
            match_ls.append(getattr(Nanoparticles, attr) == crit)
    match = db.and_(*match_ls)
    qry = session.query(Nanoparticles).filter(match)
    if return_query:
        return qry
    res = qry.limit(lim).all()
    return res if len(res) != 1 else res[0]


# INSERT FUNCTIONS


def insert_nanoparticle(atom, shape, num_shells=None):
    """
    Inserts NP and their appropriate atoms into DB

    Args:
    atom (ase.Atoms): atoms object of NP skeleton
    shape (string): common name for shape of NP
                    - e.g. icosahedron
                    - always lower case

    Kargs:
    num_shells (int): number of shells used to build NP
                      from structure_gen module

    Returns:
        (bool): True if successful insertion else False
    """
    np = Nanoparticles(shape.lower(), len(atom), num_shells=num_shells)
    session.add(np)
    for i, a in enumerate(atom):
        session.add(Atoms(i, a.x, a.y, a.z, np))
    return db_utils.commit_changes(session)


# UPDATE FUNCTIONS

def update_datarow(datarow):
    """
    Generic method to update datarow object

    Args:
    datarow (DataTable): DataTable object where changes have been made

    Returns:
        (bool): True if successful
    """
    session.add(datarow)
    return db_utils.commit_changes(session)


def update_bimet_result(metals, shape, num_atoms,
                        diameter, n_metal1,
                        CE, EE, ordering, np=None,
                        allow_insert=True, ensure_ce_min=True):
    """
    Takes raw data and inserts the BimetallicResults datatable
    - will update CE, EE, and ordering of an entry if necessary
    """
    res = get_bimet_result(metals, shape, num_atoms, n_metal1)
    if res:
        # if ensure_ce_min, do not overwrite a minimum CE structure
        # with new data
        if ensure_ce_min and res.CE < CE:
            return
        res.CE = CE
        res.EE = EE
        res.ordering = ordering
    elif not allow_insert:
        return
    else:
        metal1, metal2 = db_utils.sort_metals(metals)
        res = BiMet(metal1, metal2, shape, num_atoms,
                    diameter, n_metal1, num_atoms-n_metal1,
                    CE, EE, ordering)
        if np:
            np.bimetallic_results.append(res)
            session.add(np)

    # commit changes
    session.add(res)
    return db_utils.commit_changes(session)


# REMOVE FUNCTIONS
def remove_entry(entry_inst):
    """
    Deletes entry (and its children if applicable)
    from DB

    Args:
    entry_inst (DataTable instance): DB entry to be deleted
    """
    session.delete(entry_inst)
    return db_utils.commit_changes(session)


def remove_nanoparticle(shape=None, num_atoms=None, num_shells=None):
    """
    Removes a single NP and its corresponding atoms
    - only commits if a single np is queried to be removed

    Kargs:
    shape (str): shape of NP
    num_atoms (int): number of atoms in NP
    num_shells (int): number of shells used to build NP
                      from structure_gen module

    Returns:
        True if successful
    """
    res = get_nanoparticle(shape, num_atoms, num_shells)
    if isinstance(res, list):
        raise Exception('Can only remove one Nanoparticle')
    else:
        return remove_entry(res)

if __name__ == '__main__':
    nps, bi = get_all_tables()
