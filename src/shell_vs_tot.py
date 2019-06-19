import os
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from npdb import db_inter
import plot_defaults

palette = sns.color_palette('husl', 10)
# sns.palplot(palette)
sns.set_palette(palette)

plt.rcParams['axes.spines.top'] = False
plt.rcParams['legend.fontsize'] = 12

for opt in plt.rcParams:
    if plt.rcParams[opt] == 'bold':
        plt.rcParams[opt] = 'normal'

# first shell for all shapes
first = {'icosahedron': 2,
         'elongated-pentagonal-bipyramid': 2,
         'cuboctahedron': 1,
         'fcc-cube': 1}

COLORS = ['cyan', 'blue', 'red', 'green', 'gold',
          'orange', 'purple', 'pink', '#bcbd22', '#8c564b', 'gray']


def build_nmet2_nmet2shell_plot(metals, shape, num_shells,
                                show_ee=True, show=True, save=False,
                                pctx=False, pcty=False):
    """plots number of metal2 in shell_i vs number of metal2 in NP

    Arguments:
        metals {str} -- two metals (element names) in NP
        shape {str} -- shape of NP
        num_shells {int} -- number of shells the NP is made of
                            (excluding core atom)

    Keyword Arguments:
        show_ee {bool} -- plots excess energy (EE) on same plot with
                          secondary axis
                          (default: {True})
        show {bool} -- shows figure is True (default: {True})
        save {bool} -- saves figure is True (default: {False})
        pctx {bool} -- x-axis plotted as concentration of metal2 if True
                       (default: {False})
        pcty {bool} -- y-axis plotted as percentage of shell_i that is metal2
                       (default: {False})
    """
    if num_shells < first[shape]:
        raise ValueError("num_shells is too small")

    shell_dict = {s: db_inter.build_atoms_in_shell_list(shape, s)
                  for s in range(first[shape], num_shells + 1)}

    for s in shell_dict:
        print('%i: %i atoms' % (s, len(shell_dict[s])))

    # if pct use X else N
    xtype = 'X' if pctx else 'N'
    ytype = 'X' if pcty else 'N'

    # all results run with specific size, shape, and metals
    nanoparticles = db_inter.get_bimet_result(metals=metals, shape=shape,
                                              num_shells=num_shells)

    # get NP with lowest Cohesive Energy (CE) and Excess Energy (EE)
    min_atoms = {'CE': min(nanoparticles, key=lambda i: i.CE),
                 'EE': min(nanoparticles, key=lambda i: i.EE)}

    # initialize results matrix
    results = np.zeros((len(nanoparticles), len(shell_dict) + 2))

    # map shell number to column index of results
    shelli = [0] + sorted(shell_dict)
    for i, nanop in enumerate(nanoparticles):
        results[i, 0] = nanop.n_metal2
        for s in shell_dict:
            col = shelli.index(s)
            shell_atoms = nanop.build_atoms_obj()[shell_dict[s]]
            nmet2_shell = sum([1 for a in shell_atoms
                               if a.symbol == nanop.metal2])
            results[i, col] = nmet2_shell
            # calculate as percentage of atoms that are "metal2" in given shell
            if pcty:
                results[i, col] = nmet2_shell / len(shell_atoms)
        results[i, -1] = nanop.EE

    # sort results by n_metal2
    results = results[results[:, 0].argsort()]

    if pctx:
        results[:, 0] = results[:, 0] / nanop.num_atoms

    if not (pctx or pcty):
        # ensure shell counts were correctly calculated
        pass
        # assert (results[:, 0] == results[:, 1:-1].sum(1)).all()

    fig, ax = plt.subplots(figsize=(9, 7))

    plt.rcParams['axes.spines.right'] = show_ee
    if show_ee:
        # excess energy secondary axis
        ee_color = 'dodgerblue'

        ax2 = ax.twinx()
        ax2.tick_params(axis='y', labelcolor=ee_color)

        ax.scatter([], [], marker='o', label='EE', color=ee_color,
                   s=50, edgecolor='k')

        ax2.scatter(results[:, 0], results[:, -1], marker='o',
                    color=ee_color, edgecolor='k', s=50)
        ax2.set_ylabel('Excess Energy (eV / atom)', color=ee_color)

        # set ylim
        mag = (results[:, -1].max() - results[:, -1].min())
        buffmin = mag * 0.1
        buffmax = mag * 0.3
        ax2.set_ylim(results[:, -1].min() - buffmin,
                     results[:, -1].max() + buffmax)

    # plot (n or x)i vs (n or x)i-shell
    for i, col in enumerate(range(results.shape[1] - 2, 0, -1)):
        if not i:
            lab = 'Surface'
        elif col == 1:
            lab = 'Core'
        else:
            lab = 'Shell %i' % col
        ax.plot(results[:, 0], results[:, col], 'o-',
                label=lab, zorder=col)

    # used to match Larsen et al.'s plot
    if num_shells == 5 and not (pcty or pctx) and shape == 'icosahedron':
        ax.set_xlim(0, 315)
        ax.set_ylim(0, 180)

    if pctx:
        ax.set_xlim(-0.1, 1.1)

    if pcty:
        ax.set_ylim(0, 1.19)
        ax.set_yticklabels(['{:,.0%}'.format(x) for x in ax.get_yticks()])

    ax.set_xlabel('$\\rm %s_{%s}$ in NP' % (xtype, nanop.metal2))
    ax.set_ylabel('$\\rm %s_{%s}$ in Shell$\\rm _i$' % (ytype, nanop.metal2))

    title = '%s atom %s%s %s NP' % (nanop.num_atoms, nanop.metal1,
                                    nanop.metal2, nanop.shape)
    ax.set_title(title)

    ax.legend(loc='upper center', ncol=4,
              handletextpad=0.2, columnspacing=0.6)
    fig.tight_layout()
    user = os.path.expanduser('~')
    path = os.path.join(user,
                        'Box Sync',
                        'Michael_Cowan_PhD_research',
                        'np_ga',
                        'FIGURES',
                        '%smet_vs_%smetshell-LARSON\\%s%s-%s'
                        % (xtype, ytype, nanop.metal1,
                           nanop.metal2, nanop.shape[:3]))
    if save:
        if not os.path.isdir(path):
            pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        if show_ee:
            path = os.path.join(path, 'ee')
            if not os.path.isdir(path):
                os.mkdir(path)
        fig.savefig(os.path.join(path, title.replace(' ', '_') + '.svg'))
        res = min_atoms['EE']
        res.save_np(os.path.join(path, 'EE-%s-%s.xyz'
                                 % (res.build_chem_formula(), res.shape[:3])))
    if show:
        plt.show()
    return fig, results, min_atoms


def batch_build_figs():
    pctx = False
    shell_range = range(2, 12)

    # NESTED MESS
    for shape in ['fcc-cube', 'cuboctahedron', 'icosahedron']:
        for pcty in [False, True]:
            for metals in ['agau', 'agcu', 'aucu']:
                for show_ee in [False, True]:
                    for num_shells in shell_range:
                        build_nmet2_nmet2shell_plot(
                            metals,
                            shape,
                            num_shells,
                            show=False,
                            save=True,
                            show_ee=show_ee,
                            pctx=pctx,
                            pcty=pcty)

                        plt.close('all')

if __name__ == '__main__':
    desk = os.path.join(os.path.expanduser('~'), 'Desktop')

    metals = 'agcu'
    shape = 'icosahedron'
    pctx = False
    pcty = False

    save = True
    show = True
    show_ee = True

    shell_range = range(2, 12)
    for num_shells in [5]:
        fig, results, min_atoms = build_nmet2_nmet2shell_plot(
            metals,
            shape,
            num_shells,
            show=False,
            save=save,
            show_ee=show_ee,
            pctx=pctx,
            pcty=pcty)

        ee = min_atoms['EE']
        print(ee.EE)
        ee.save_np(os.path.join(desk, ee.build_chem_formula() + '_gaopt.xyz'))
        if show:
            plt.show()
