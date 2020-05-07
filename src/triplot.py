import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.tri as tri
from npdb import db_inter
import atomgraph
try:
    from ce_expansion.src.npdb import db_inter
    import ce_expansion.src.atomgraph as atomgraph
except:
    pass

# GLOBAL fontsize of axis labels and text
FS = 40


def get_fracs(metals=None, shape=None, num_shells=None, return_ee=False,
              x_metal1=None, **kwargs):
    res = db_inter.get_bimet_result(metals=metals, shape=shape,
                                    num_shells=num_shells, **kwargs)
    # limit by composition
    if x_metal1 is not None:
        res = filter(lambda r: abs(r.n_metal1 /
                                   r.num_atoms - x_metal1) <= 0.05,
                     res)

    res = sorted(res, key=lambda r: [r.shape, r.num_atoms])

    fracs = np.zeros((len(res), 3))
    ee = np.zeros(len(res))

    # track shape and size so if they don't change, you don't reload atomgraph
    prevsize = res[0].num_atoms
    prevshape = res[0].shape

    # load the first atomgraph
    bonds = res[0].nanoparticle.load_bonds_list()
    ag = atomgraph.AtomGraph(bonds, res[0].metal1, res[0].metal2)

    for i, a in enumerate(res):
        # only create new atomgraph for new size and/or shape
        if a.num_atoms != prevsize or a.shape != prevshape:
            bonds = a.nanoparticle.load_bonds_list()
            ag = atomgraph.AtomGraph(bonds, a.metal1, a.metal2)

            # update previous size and shape
            prevsize = a.num_atoms
            prevshape = a.shape

        # get bond type counts
        counts = ag.countMixing(np.array([int(z) for z in a.ordering]))

        # normalize bond counts to get bond fractions
        fracs[i] = counts / counts.sum()
        ee[i] = a.EE

    # return bond fraction for metal1-metal1 (aa) and metal2-metal2 (bb)
    aa = fracs[:, 0]
    bb = fracs[:, 1]

    # option to return excess energy with bond fractions
    if return_ee:
        return aa, bb, ee

    return aa, bb


def tri_plot(aa, bb, ax=None, marker='o', label=None, legend=False,
             z=None, zmin=None, zmax=None, cmap=None, alpha=1,
             xlab='$\\rm F_{A-A}$', ylab='$\\rm F_{B-B}$'):
    # time plot making
    start = time.time()

    if z is not None:
        assert len(z) == len(aa)

    # remove top and right axis lines
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False

    # set borderwidth of triangle
    borderwidth = 3
    plt.rcParams['axes.linewidth'] = borderwidth
    plt.rcParams['xtick.major.width'] = borderwidth / 2
    plt.rcParams['ytick.major.width'] = borderwidth / 2

    if ax is None:
        # initialize figure and axis object
        fig, ax = plt.subplots()

        # plot bottom of triangle
        ax.plot([0, 1], [1, 0], color='k', label='_nolabel_', lw=borderwidth,
                zorder=0)

        # plot grid lines
        for n in np.arange(2, 9, 2) / 10:
            ax.plot([0, n], [n, 0], color='lightgray', zorder=-1,
                    label='_nolabel_')

        # fraction limits should be [0, 1]
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # set and rotate labels
        ax.set_xlabel(xlab, rotation=180, fontsize=FS)
        ax.set_ylabel(ylab, rotation=90, fontsize=FS)

        # rotate tick labels
        for xtick, ytick in zip(ax.get_xticklabels(), ax.get_yticklabels()):
            xtick.set_rotation(90)
            ytick.set_rotation(180)

        # set aspect ratio to 1:1
        ax.set_aspect(1)
    else:
        fig = ax.figure

    if label is None:
        label = '_nolabel_'

    # plot data
    linewidth = 0.5
    ax.scatter([], [], s=100, edgecolor='k', color='white', label=label,
               marker=marker, linewidth=linewidth)

    scatter = ax.scatter(aa, bb, s=100, edgecolor='k', zorder=10,
                         label='_nolabel_', linewidth=linewidth, c=z,
                         vmin=zmin, vmax=zmax, clip_on=False, marker=marker,
                         cmap=cmap, alpha=alpha)

    # add legend
    if legend:
        ax.legend(fontsize=18, ncol=4, labelspacing=0.5, columnspacing=0.5,
                  handletextpad=0.)

    fig.tight_layout()

    print('Time to create plot: %0.2f s' % (time.time() - start))
    return fig, ax, scatter


if __name__ == '__main__':
    # fig, ax, s = tri_plot([], [])
    # fig.savefig('C:\\users\\yla\\desktop\\tutorial_triangle_AABB.svg')
    # plt.show()
    # import sys
    # sys.exit()
    # PLOT DATA PARAMS
    alpha = 1
    metals_ls = [('Ag', 'Au'), ('Ag', 'Cu'), ('Au', 'Cu')]
    metals = metals_ls[0]

    shapes = 'icosahedron', 'cuboctahedron', 'elongated-pentagonal-bipyramid'
    shell_sizes = range(1, 11)

    colors = ['lime', 'red', 'royalblue']
    markers = ['o', 's', '^']
    n = 309

    min_n = db_inter.get_shell2num(shapes[0], shell_sizes[0])
    max_n = db_inter.get_shell2num(shapes[0], shell_sizes[-1])

    fig, ax, s = tri_plot([], [], cmap=cm.get_cmap('rainbow'), zmin=min_n, zmax=max_n)
    # Create the fully random line!
    xa = np.linspace(0, 1)
    xb = 1 - xa
    aaq = xa**2
    bbq = xb**2
    ax.plot(aaq, bbq, '--', zorder=5000, color='k', lw=5)

    fig.colorbar(s, aspect=40,
                 ticks=[13, 500, 1000, 1500, 2000, 2500, 3000, 3500, 3871])

    fig.tight_layout()
    fig.savefig('C:\\users\\mcowa\\desktop\\FULLYRANDOM_tutorial_triangle.svg')
    plt.show()
    import sys
    sys.exit()

    for metals in metals_ls:
        ax = None
        fig, ax, s = tri_plot([], [])
        # Create the fully random line!
        xa = np.linspace(0, 1)
        xb = 1 - xa
        aaq = xa**2
        bbq = xb**2
        ax.plot(aaq, bbq, '--', zorder=5000, color='k', lw=3)
        for shell in shell_sizes:
            for i, shape in enumerate(shapes):
                num_atoms = db_inter.get_shell2num(shape, shell)
                aa, bb, ee = get_fracs(metals=metals,
                                       return_ee=True, shape=shape,
                                       num_atoms=num_atoms)
                z = [num_atoms] * len(aa)
                fig, ax, s = tri_plot(
                    bb, aa, ax=ax,
                    xlab='$\\rm F_{%s-%s}$' % (metals[0], metals[0]),
                    ylab='$\\rm F_{%s-%s}$' % (metals[1], metals[1]),
                    z=z, zmin=min_n, zmax=max_n, alpha=alpha,
                    marker='o',  # markers[i],
                    cmap=cm.get_cmap('rainbow'))

        ax.text(0.5, 0.5, ''.join(metals), fontsize=FS, rotation=135,
                va='bottom', ha='center')
        ax.text(0.5, 1, '$\\rm N_{Atoms}$', fontsize=FS)
        fig.colorbar(s, orientation='horizontal', aspect=40,
                     ticks=[13, 500, 1000, 1500, 2000, 2500, 3000, 3500, 3871])

        fig.tight_layout()
        fig.savefig('C:\\users\\mcowa\\desktop\\FULLYRANDOMLINE_%s_colorbar_allcircles.svg'
                    % ''.join(metals))
    plt.show()
