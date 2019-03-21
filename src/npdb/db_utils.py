import sqlalchemy as db
import traceback

"""
Helper functions/classes used in db_inter.py
"""


def commit_changes(session, raise_exception=False):
    """
    Attempts to commit changes to DB
    - if it fails, error is logged and DB is rolled back

    Args:
    session (sqlalchemy.Session): session connected to DB

    Kargs:
    raise_exception (bool): if False, returns False instead of
                            raising exception
                            (default: False)

    Returns:
        (bool): True if successful commit else False
    """
    try:
        session.commit()
        return True
    except Exception as e:
        # write to error log file in npdb
        with open('io_error.log', 'a') as fid:
            fid.write(traceback.format_exc() + '\n\n')

        session.rollback()
        session.flush()

        if raise_exception:
            raise e
        return False


def get_table(session, datatable, lim=None, return_query=False, **kwargs):
    """
    Returns entry/entries from table if criteria is matched
    - generic function that is called from specific functions
      in db_inter
    - if no criteria given, all data (up to <lim> amount)
      returned

    Args:
    session (sqlalchemy.Session): session connected to DB
    datatable (Datatable object): datatable to query

    Kargs:
    lim (int): max number of entries returned
               (default: None = no limit)
    return_query (bool): if True, return query and
                         not results
    **kwargs: arguments whose name(s) matches a column in the datatable

    Returns:
        (Datatable)(s) if match is found else (None)
    """
    match_ls = []
    for attr in kwargs:
        if kwargs[attr] is not None:
            if attr == 'metals':
                metal1, metal2 = sort_metals(kwargs[attr])
                match_ls.append(datatable.metal1 == metal1)
                match_ls.append(datatable.metal2 == metal2)
            else:
                match_ls.append(getattr(datatable, attr) == kwargs[attr])
    match = db.and_(*match_ls)
    qry = session.query(datatable).filter(match)
    if return_query:
        return qry
    res = qry.limit(lim).all()
    return res if len(res) != 1 else res[0]


def sort_metals(metals):
    """
    Handles iterable or string of metals and returns them
    in alphabetical order

    Args:
    metals (str || iterable): two metal element names

    Returns:
        (tuple): element names in alphabetical order
    """
    if isinstance(metals, str):
        metal1, metal2 = sorted([metals[:2], metals[2:]])
    else:
        metal1, metal2 = sorted(metals)
    return metal1.title(), metal2.title()


class NPDatabaseError(Exception):
    """Custom exception for DB IO errors"""
