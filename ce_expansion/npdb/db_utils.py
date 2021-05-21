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


def sort_2metals(metals):
    """
    Handles iterable or string of 2 metals and returns them
    in alphabetical order

    Args:
    metals (str || iterable): two metal element names

    Returns:
        (tuple): element names in alphabetical order
    """
    # return None's if metals is None
    if metals is None:
        return None, None
    if isinstance(metals, str):
        if len(metals) != 4:
            raise ValueError('str can only have two elements.')
        metal1, metal2 = sorted([metals[:2], metals[2:]])
    else:
        metal1, metal2 = sorted(metals)
    return metal1.title(), metal2.title()


class NPDatabaseError(Exception):
    """Custom exception for DB IO errors"""
