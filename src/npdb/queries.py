import sqlalchemy as db
from base import Session
from bimetallic_results import BimetallicResults as BiMet
from atoms import Atoms
from nanoparticles import Nanoparticles
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

session = None


def __cxn__(func):
    """
    Decorator to ensure sessions with DB are
    properly handled
    """
    global session
    session = Session()

    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    session.close()

    return wrapper


@__cxn__
def get_all_tables(lim=10):
    nps = session.query(Nanoparticles).limit(lim).all()
    bi = session.query(BiMet).limit(lim).all()
    return nps, bi


@__cxn__
def srf_plot(metals, shape):
    """
    Creates a 3D surface plot from NP SQL database
    - plots Size vs. Shape vs. Excess Energy (EE)

    Args:
    metals (string || iterable): string(s) containing two metal elements
    shape (string): shape of the NP

    Returns:
        (plt.figure): figure of 3D surface plot
    """
    if isinstance(metals, str):
        metal1, metal2 = sorted([metals[:2], metals[2:]])
    else:
        metal1, metal2 = sorted(metals)

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


if __name__ == '__main__':
    nps, bi = get_all_tables()
