import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ce_expansion.npdb import db_inter

palette = sns.color_palette('husl', 10)
# sns.palplot(palette)
sns.set_palette(palette)

# GET BOX SYNC PATH
user = os.path.expanduser('~')
box = user
for option in [os.path.join(user, 'Box Sync'),
               'D:\\MCowan\\Box Sync']:
    if os.path.isdir(option):
        box = option
        break
else:
    if __name__ == "__main__":
        print('Could not find Box Sync path.')

plt.rcParams['axes.spines.top'] = False
plt.rcParams['legend.fontsize'] = 12

for opt in plt.rcParams:
    if plt.rcParams[opt] == 'bold':
        plt.rcParams[opt] = 'normal'

COLORS = ['cyan', 'blue', 'red', 'green', 'gold',
          'orange', 'purple', 'pink', '#bcbd22', '#8c564b', 'gray']


def build_nmet2_nmet2shell_plot(metals, shape, num_shells,
                                show_ee=True, show=True, save=False,
                                pctx=False, pcty=False):
    """Plots number of metal2 in shell_i vs number of metal2 in NP

    Args:
    - metals (str): two metals (element names) in NP
    - shape (str): shape of NP
                   CURRENTLY SUPPORTS:
                   - cuboctahedron
                   - elongated-pentagonal-bipyramid
                   - icosahedron
    - num_shells (int): number of shells the NP is made of
                        (excluding core atom)

    KArgs:
    show_ee (bool): plots excess energy (EE) on same plot with secondary axis
                    (default: {True})
    - show (bool): shows figure is True
                   (default: True)
    - save (bool): saves figure is True
                   (default: False)
    - pctx (bool): x-axis plotted as concentration of metal2 if True
                   (default: {False})
    - pcty (bool): y-axis plotted as percentage of shell_i that is metal2
                   (default: {False})

    Returns:
    - (plt.Figure): matplotlib plot of results
    - (np.ndarray): matrix of results: [n_metal2, (n|x)metal2-shell_i, ..., EE]
    - (dict): minimum CE and minimum EE NP as datatable.BimetallicResults objs

    Raises:
    ValueError: num_shells must be greater than 0
    """
    if num_shells < 1:
        raise ValueError("num_shells is too small")

    # build dictionary of atom shell indices
    shell_dict = db_inter.build_atoms_in_shell_dict(shape, num_shells)

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

    # record counts in results matrix
    for i, nanop in enumerate(nanoparticles):
        results[i, 0] = nanop.n_metal2
        for s in shell_dict:
            col = s + 1
            # add number of metal2 in shell <s> to results
            results[i, col] = (nanop.build_atoms_obj()
                               [shell_dict[s]].symbols == nanop.metal2).sum()

            # calculate as percentage of atoms that are "metal2" in given shell
            if pcty:
                results[i, col] = results[i, col] / len(shell_dict[s])

        # record EE of nanoparticle
        results[i, -1] = nanop.EE

    # sort results by n_metal2
    results = results[results[:, 0].argsort()]

    # convert x axis to metal 2 concentration if pctx
    if pctx:
        results[:, 0] = results[:, 0] / nanop.num_atoms

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
        # outer shell = Surface
        if not i:
            lab = 'Surface'

        # DO NOT PLOT CORE ATOM (ignore it)
        elif col == 1:
            lab = 'Core'
            continue
        else:
            lab = 'Shell %i' % (col - 1)
        ax.plot(results[:, 0], results[:, col], 'o-',
                label=lab, zorder=col)

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
    path = os.path.join(box,
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
        fig.savefig(os.path.join(path, title.replace(' ', '_') + '.png'),
                    dpi=50)
        fig.savefig(os.path.join(path, title.replace(' ', '_') + '.svg'))
        res = min_atoms['EE']
        res.save_np(os.path.join(path, 'EE-%s-%s.xyz'
                                 % (res.build_chem_formula(), res.shape[:3])))
    if show:
        plt.show()
    return fig, results, min_atoms


def batch_build_figs():
    pctx = False
    shell_range = range(2, 11)

    # NESTED MESS
    print('')
    for shape in ['cuboctahedron',
                  'elongated-pentagonal-bipyramid',
                  'icosahedron']:
        for pcty in [False, True]:
            for metals in ['AgAu', 'AgCu', 'AuCu']:
                for show_ee in [False, True]:
                    for num_shells in shell_range:
                        print(' %s-%s: %02i shells %s %s'
                              % (shape.upper()[:3],
                                 metals,
                                 num_shells,
                                 ['', 'PCTY'][pcty],
                                 ['', 'EE'][show_ee]),
                              end='\r')
                        build_nmet2_nmet2shell_plot(
                            metals,
                            shape,
                            num_shells,
                            show=False,
                            save=True,
                            show_ee=show_ee,
                            pctx=pctx,
                            pcty=pcty)
                        print(' ' * 100, end='\r')

                        plt.close('all')


if __name__ == '__main__':
    # batch_build_figs()

    metals = 'aucu'
    shape = 'cuboctahedron'
    num_shells = 5
    pcty = False
    fig, res, min_atoms = build_nmet2_nmet2shell_plot(
        metals,
        shape,
        num_shells,
        pcty=pcty)
