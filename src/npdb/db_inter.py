import sqlalchemy as db
import numpy as np
import pickle
import os
try:
    import db_utils
    import base
    import datatables as tbl
except:
    from npdb import db_utils
    import npdb.base as base
    import npdb.datatables as tbl
import pandas as pd
import matplotlib
import matplotlib.ticker
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
             - does edit current DB entries
"""

# global session
session = base.Session(autoflush=True)

# create all datatables if not present in DB
base.Base.metadata.create_all(base.engine)

# BUILD FUNCTIONS


def build_df(datatable, lim=None, **kwargs):
    """
    GENERIC FUNCTION
    Returns pandas dataframe of entries in <datatable>
    that match criteria in <**kwargs>
    - NOTE: changes 'shape' column name to 'np_shape'!

    Args:
    datatable (Datatable class): datatable to query

    KArgs:
    lim (int): max number of entries returned
               (default: None = no limit)
    **kwargs: arguments whose name(s) matches a column in the datatable

    Returns:
        (pd.DataFrame): df of results
    """

    # convert sql query results into pd.DataFrame
    qry = get_entry(datatable, lim=lim, **kwargs, return_query=True).statement
    df = pd.read_sql(qry, session.bind)
    df.columns = [col if col != 'shape' else 'np_shape' for col in df.columns]
    return df


def build_coefficient_dict(metals):
    """
    Creates a 3D surface plot from NP SQL database
    - plots Size vs. Shape vs. Excess Energy (EE)
    - can also use configurational entropy of mixing
      to plot Size vs. Shape vs. G = EE - T * delS(mix)

    Args:
    metals (string || iterable): string(s) containing two metal elements

    Returns:
        (dict): coefficient dictionary for GA sim
    """
    metal1, metal2 = db_utils.sort_metals(metals)

    # homolytic half bond energies
    res_aa = get_model_coefficient(metal1, metal1)
    res_bb = get_model_coefficient(metal2, metal2)

    # heterolytic half bond energies
    res_ab = get_model_coefficient(metal1, metal2)
    res_ba = get_model_coefficient(metal2, metal1)

    # raise error if coefficients are not found
    errors = []
    for res, ms in zip([res_aa, res_bb, res_ab, res_bb],
                       ((metal1, metal1), (metal2, metal2),
                        (metal1, metal2), (metal2, metal1))):
        if not res:
            errors.append("no info for %s - %s coefficients" % ms)
    if errors:
        raise db_utils.NPDatabaseError('\n'.join(errors))

    # return back coefficients as dictionary of dictionaries
    return {metal1: {metal1: list(map(lambda i: i.bond_energy, res_aa)),
                     metal2: list(map(lambda i: i.bond_energy, res_ab))},
            metal2: {metal2: list(map(lambda j: j.bond_energy, res_bb)),
                     metal1: list(map(lambda j: j.bond_energy, res_ba))}}


def build_metals_list():
    """
    Returns list of possible metal combinations for GA sims
    """
    return [r[0] for r in session.query(tbl.ModelCoefficients.element1)
            .distinct().order_by(tbl.ModelCoefficients.element1).all()]


def build_new_structs_plot(metal_opts, shape_opts, pct=False):
    """
    Uses BimetallicLog to create 2D line plot of
    new structures found vs. datetime

    Args:
    metal_opts (list): list of metal options
    shape_opts (list): list of shape options

    Kargs:
    pct (bool): if True, y-axis = % new structures
                else, y-axis = number of new structures

    Returns:
        (plt.Figure): 2D line plot object
    """
    if isinstance(metal_opts, str):
        metal_opts = [metal_opts]
    if isinstance(shape_opts, str):
        shape_opts = [shape_opts]

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    fig, ax = plt.subplots()
    i = 0
    tot_lines = len(metal_opts) * len(shape_opts)
    for j, m in enumerate(metal_opts):
        metal1, metal2 = db_utils.sort_metals(m)
        for point, shape in zip(['o', 'x', '^', 's'], shape_opts):
            # use abbreviated names for shapes
            lbl_shape = shape \
                .replace('icosahedron', 'Ico') \
                .replace('cuboctahedron', 'Cuboct') \
                .replace('fcc-cube', 'FCC-Cube') \
                .replace('elongated-pentagonal-bipyramid', 'J16')

            # pd.DataFrame of bimetallic log data
            df = build_df(tbl.BimetallicLog, metals=m, shape=shape)
            x = df.date.values
            label = '%s%s: %s' % (metal1, metal2, lbl_shape)
            # color = plt.cm.tab20c((i / float(tot_lines)) * 0.62)

            y = df.new_min_structs.values
            if pct:
                y = y / df.tot_structs.values[0]

            ax.plot(x, y, '%s-' % point, label=label,
                    color=colors[j])
            i += 1

    if pct:
        ax.yaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter('{0:.0%}'.format))
        ax.set_ylim(0, 1)
        ax.set_ylabel('Percent Count')
    else:
        ax.set_ylabel('Total Count')
    ax.legend(ncol=len(metal_opts))
    ax.set_title('New Minimum Structures Found')
    ax.set_xlabel('Batch Run Date')

    # (month/day) x axis tick labels
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%m/%d'))
    fig.tight_layout()
    return fig


def build_srf_plot(metals, shape, delg=False, T=298):
    """
    Creates a 3D surface plot from NP SQL database
    - plots Size vs. Shape vs. Excess Energy (EE)
    - can also use configurational entropy of mixing
      to plot Size vs. Shape vs. G = EE - T * delS(mix)

    Args:
    metals (string || iterable): string(s) containing two metal elements
    shape (string): shape of the NP

    Returns:
        (plt.figure): figure of 3D surface plot
    """
    metal1, metal2 = db_utils.sort_metals(metals)

    runs = session.query(tbl.BimetallicResults.diameter,
                         (tbl.BimetallicResults.n_metal2 /
                          db.cast(tbl.BimetallicResults.num_atoms,
                                  db.Float))
                         .label('comps'),
                         tbl.BimetallicResults.EE) \
        .filter(db.and_(tbl.BimetallicResults.metal1 == metal1,
                        tbl.BimetallicResults.metal2 == metal2,
                        tbl.BimetallicResults.shape == shape)) \
        .statement
    df = pd.read_sql(runs, session.bind)

    # three parameters to plot
    size = df.diameter.values
    comps = df.comps.values
    ees = df.EE.values

    if delg:
        # k_b T [eV] = (25.7 mEV at 298 K)
        kt = 25.7E-3 * (T / 298.)
        del_s = comps * np.ma.log(comps).filled(0) + \
            (1 - comps) * np.ma.log(1 - comps).filled(0)
        del_s *= -kt

        ees -= del_s

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
    if delg:
        ax.set_zlabel('G (eV)')
        ax.set_title('%iK\n%s %s %s' % (T, metal1, metal2, shape.title()))
    else:
        ax.set_zlabel('EE (eV)')
        ax.set_title('%s %s %s' % (metal1, metal2, shape.title()))
    return fig


# GET FUNCTIONS


def get_entry(datatable, lim=None, return_query=False, **kwargs):
    """
    GENERIC FUNCTION
    Returns entry/entries from table if criteria is matched
    - if no criteria given, all data (up to <lim> amount)
      returned

    Args:
    datatable (Datatable class): datatable to query

    Kargs:
    lim (int): max number of entries returned
               (default: None = no limit)
    return_query (bool): if True, return query and
                         not results
    **kwargs: arguments whose name(s) matches a column in the datatable

    Returns:
        (Datatable instance(s)) if match is found else (None)
    """
    match_ls = []
    for attr in kwargs:
        if kwargs[attr] is not None:
            if attr == 'metals':
                metal1, metal2 = db_utils.sort_metals(kwargs[attr])
                match_ls.append(datatable.metal1 == metal1)
                match_ls.append(datatable.metal2 == metal2)
            else:
                match_ls.append(getattr(datatable, attr) == kwargs[attr])
    match = db.and_(*match_ls)
    qry = session.query(datatable).filter(match).limit(lim)
    if return_query:
        return qry
    res = qry.all()
    return res if len(res) != 1 else res[0]


def get_bimet_log(metals=None, shape=None, date=None, lim=None,
                  return_query=False):
    """
    Returns BimetallicLog entry that matches criteria
    - if no criteria given, all data (up to <lim> amount)
      returned

    Kargs:
    metals (str || iterable): two metal elements
    shape (str): shape of NP
    date (datetime.datetime): GA batch run completion time
    lim (int): max number of entries returned
               (default: None = no limit)
    return_query (bool): if True, return query and
                         not results

    Returns:
        (BimetallicResults)(s) if match is found else (None)
    """
    return get_entry(tbl.BimetallicLog, **locals())


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
    return get_entry(tbl.BimetallicResults, **locals())


def get_model_coefficient(element1=None, element2=None, cn=None,
                          lim=None, return_query=False):
    """
    Returns tbl.ModelCoefficients entries that match criteria
    - if no criteria given, all data (up to <lim> amount)
      returned
    - query is always ordered by cn

    Kargs:
    element1 (str): first element
    element2 (str): second element
    cn (int): coordination number of element 1
    lim (int): max number of entries returned
               (default: None = no limit)
    return_query (bool): if True, returns just the query
                         and not the results

    Returns:
        (tbl.ModelCoefficients)(s) if match else (None)
    """
    return get_entry(tbl.ModelCoefficients, **locals())


def get_nanoparticle(shape=None, num_atoms=None, num_shells=None,
                     lim=None, return_query=False):
    """
    Returns tbl.Nanoparticles entries that match criteria
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
        (tbl.Nanoparticles)(s) if match else (None)
    """
    return get_entry(tbl.Nanoparticles, **locals())

# INSERT FUNCTIONS


def insert_model_coefficients(coeffs_dict=None):
    """
    Inserts model coefficients into ModelCoefficients
    - checks DB to ensure entry does not already exist

    Kargs:
    coeffs_dict (dict): dictionary of coefficients
                        if None, read in from data directory
                        (default: None)

    Returns:
        (bool): True if successful insertion else False
    """
    if not coeffs_dict:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '..', '..', 'data', 'coefs_set.pickle')
        with open(path, 'rb') as fid:
            coeffs_dict = pickle.load(fid)

    # dict(element1=dict(element2=[bond_energies where index = cn]))
    for e1 in coeffs_dict:
        for e2 in coeffs_dict[e1]:
            for cn in range(len(coeffs_dict[e1][e2])):
                if not get_model_coefficient(e1, e2, cn, lim=1):
                    session.add(tbl.ModelCoefficients(e1, e2, cn,
                                                      coeffs_dict[e1][e2][cn]))
    return db_utils.commit_changes(session)


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
    np = tbl.Nanoparticles(shape.lower(), len(atom), num_shells=num_shells)
    session.add(np)
    for i, a in enumerate(atom):
        session.add(tbl.Atoms(i, a.x, a.y, a.z, np))
    res = db_utils.commit_changes(session, raise_exception=True)
    return np


def insert_bimetallic_log(start_time, metal1, metal2, shape,
                          ga_generations, shell_range, new_min_structs,
                          tot_structs, batch_run_num=None):
    """
    """
    session.add(tbl.BimetallicLog(start_time, metal1, metal2, shape,
                                  ga_generations, shell_range, new_min_structs,
                                  tot_structs, batch_run_num))
    return db_utils.commit_changes(session)


# UPDATE FUNCTIONS


def update_entry(entry_inst):
    """
    GENERIC METHOD
    Updates datarow object

    Args:
    entry_inst (DataTable): DataTable object where changes have been made

    Returns:
        (bool): True if successful
    """
    session.add(entry_inst)
    return db_utils.commit_changes(session)


def update_bimet_result(metals, shape, num_atoms,
                        diameter, n_metal1,
                        CE, ordering, EE=None, nanop=None,
                        allow_insert=True, ensure_ce_min=True):
    """
    Takes raw data and inserts the BimetallicResults datatable
    - will update CE, EE, and ordering of an entry if necessary

    Returns:
        BimetallicResults entry after successfully updating
    """
    res = get_bimet_result(metals=metals,
                           shape=shape,
                           num_atoms=num_atoms,
                           n_metal1=n_metal1)
    if res:
        # if ensure_ce_min, do not overwrite a minimum CE structure
        # with new data
        if ensure_ce_min and res.CE < CE:
            return
        res.CE = CE
        res.ordering = ordering
        res.EE = EE
    elif not allow_insert:
        return
    else:
        metal1, metal2 = db_utils.sort_metals(metals)
        res = tbl.BimetallicResults(
            metal1=metal1,
            metal2=metal2,
            shape=shape,
            num_atoms=num_atoms,
            diameter=diameter,
            n_metal1=n_metal1,
            n_metal2=num_atoms - n_metal1,
            CE=CE,
            ordering=ordering,
            EE=EE)
        if nanop:
            nanop.bimetallic_results.append(res)
            session.add(nanop)

    # commit changes
    session.add(res)
    db_utils.commit_changes(session, raise_exception=True)
    return res


# REMOVE FUNCTIONS
def remove_entry(entry_inst):
    """
    GENERIC FUNCTION
    Deletes entry (and its children if applicable) from DB

    Args:
    entry_inst (DataTable instance): datarow to be deleted
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
        raise db_utils.NPDatabaseError('Can only remove one Nanoparticle')
    elif not res:
        raise db_utils.NPDatabaseError('Unable to find matching Nanoparticle')
    else:
        return remove_entry(res)


# ensure model coefficients are in DB
# insert_model_coefficients()

if __name__ == '__main__':
    # get all bimetallic NPs of given metals and shape
    metals = 'aucu'
    shape = 'icosahedron'
    f = build_new_structs_plot(metals, shape, True)
    # only bimetallic NPs
    only_bimet = db.and_(tbl.BimetallicResults.n_metal1 != 0,
                         tbl.BimetallicResults.n_metal2 != 0)

    nps = get_bimet_result(metals=metals, shape=shape, return_query=True) \
        .filter(only_bimet).all()
