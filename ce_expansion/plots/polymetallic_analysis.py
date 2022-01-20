import os

from ase.data.colors import jmol_colors
from ase.data import chemical_symbols
from pprint import pprint
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pkg_resources
from ase.visualize import view
import ase
import molgif
import ase.build
import json


from ce_expansion.ga import GA
from ce_expansion.atomgraph.bcm import BCModel
from ce_expansion.atomgraph import adjacency
from ce_expansion.ga import structure_gen
from ce_expansion.npdb import db_inter
from ce_expansion.npdb import datatables

#incorperate into plots folder as triemtallic/polymetallic tools/plots
# (FUTURE) add docstrings to functions to help ease of use - try and be consistent with BCModel

def batch_GA_run(bcm, shape, iteration=1, **ga_kwargs):
    """
        Code which gives similar plot to atom_rangomizer_trimetallic_MC.py, giving us a triangle graph of EE of
        best compositions, now enhanced by the GA
    """

    details = '_'.join([f'{key}_{value}' for key, value in ga_kwargs.items()])
    # Loading/Creating data for Plots
    fname = f"{len(bcm)}_{bcm.metal_types[0]}{bcm.metal_types[1]}{bcm.metal_types[2]}_{details}generations_optimized_np.txt"

    print("Generating Data:: " + fname)

    iterby = iteration
    data = []
    for num_m1 in range(1, len(bcm) + 1, iterby):
        for num_m2 in range(1, len(bcm) - num_m1, iterby):
            num_cu = len(bcm) - num_m2 - num_m1

            comp = np.array([num_m1, num_m2, num_cu])

            mono_check = np.count_nonzero(comp)

            if mono_check == 1:
                pass
            else:
                ga = GA(bcm, comp, shape, random=True)
                ga.run(**ga_kwargs)
                ga.save_to_db()

                ee = bcm.calc_ee(ga[0].ordering)

                data.append([num_m1 / len(bcm), num_m2 / len(bcm), ee])


def vis_layers(res: datatables.PolymetallicResults):
    bcm = BCModel(res.atoms_obj)
    layers = bcm.shell_map

    # for layer in layers:
    #     view(res.atoms_obj[layers[layer]])

    surf_indicies = bcm.shell_map[bcm.num_shells]

    kern_indicies = list(set(range(len(bcm))) - set(surf_indicies))

    surface = res.atoms_obj[surf_indicies]
    kernal = res.atoms_obj[kern_indicies]

    surf = molgif.Molecule(surface)
    surf.save_rot_gif(loop_time=12)

    kern = molgif.Molecule(kernal)
    kern.save_rot_gif(loop_time=12)

    view(surface)
    view(kernal)


def plot_EE_dB(metal_types=None, num_atoms=None, shape=None):
    all_res = db_inter.get_polymet_result(num_atoms=num_atoms, metals=metal_types, shape=shape, return_list=True)

    # Plotting Data
    res: datatables.PolymetallicResults

    ees = np.zeros(len(all_res))
    xcomps = np.zeros((len(all_res), 3))

    for i, res in enumerate(all_res):
        ees[i] = res.EE
        xcomps[i, :] = res.composition / res.num_atoms

    x_ag = xcomps[:, 0]
    x_au = xcomps[:, 1]

    current_max_ee = abs(ees).max()
    abs_max_ee = current_max_ee * 1.01

    # make sure abs_max_ee is greater than all ees in data
    assert abs_max_ee > current_max_ee, f"Current max EE ({current_max_ee:.3f}) is greater than {abs_max_ee}"

    # define a colormap (you can look these up if you want to change it)
    cmap = 'bwr_r'

    # create the scatter plot and set c=ces
    scatter = plt.scatter(x_ag, x_au, c=ees, cmap=cmap, vmin=-abs_max_ee, vmax=abs_max_ee, s=25)

    # add the colorbar
    cbar = plt.colorbar(scatter)

    # set a colorbar label
    cbar.ax.set_ylabel('BCM EE (eV/atom)')
    plt.axes().set_facecolor("gainsboro")

    # add axis labels (using Latex math for subscripts)
    plt.xlabel('$\\rm X_{Ag}$')
    plt.ylabel('$\\rm X_{Au}$')
    plt.title(f'{num_atoms} {metal_types} {shape} EE Plot')
    plt.show()


def plot_gMix_dB(metal_types=None, num_atoms=None, shape=None, Temp=273):
    all_res = db_inter.get_polymet_result(num_atoms=num_atoms, metals=metal_types, shape=shape, return_list=True)

    # Plotting Data
    res: datatables.PolymetallicResults

    gMix = np.zeros(len(all_res))
    xcomps = np.zeros((len(all_res), 3))

    bcm = BCModel(all_res[300].atoms_obj)

    for i, res in enumerate(all_res):
        gMix[i] = bcm.calc_gmix(res.ordering, T=Temp)
        xcomps[i, :] = res.composition / res.num_atoms

    x_ag = xcomps[:, 0]
    x_au = xcomps[:, 1]

    current_max_gMix = abs(gMix).max()
    abs_max_gMix = current_max_gMix * 1.01

    # make sure abs_max_ee is greater than all ees in data
    assert abs_max_gMix > current_max_gMix, f"Current max gMix ({current_max_gMix:.3f}) is greater than {abs_max_gMix}"

    # define a colormap (you can look these up if you want to change it)
    cmap = 'bwr_r'

    # create the scatter plot and set c=ces
    scatter = plt.scatter(x_ag, x_au, c=gMix, cmap=cmap, vmin=-abs_max_gMix, vmax=abs_max_gMix, s=25)

    # add the colorbar
    cbar = plt.colorbar(scatter)

    # set a colorbar label
    cbar.ax.set_ylabel('gMix (eV/atom)')
    plt.axes().set_facecolor("gainsboro")

    # add axis labels (using Latex math for subscripts)
    plotname = f'{num_atoms}_{metal_types}_{shape}_{Temp}K_gMix_Plot'

    path = os.path.dirname(r"C:/Users/Brenno_Ferreira/Documents/Research_CANELa/gMix_Plots/")
    save_path = path + "/" + plotname + ".png"

    plt.xlabel('$\\rm X_{Ag}$')
    plt.ylabel('$\\rm X_{Au}$')
    plt.title(plotname)
    plt.savefig(save_path)
    plt.clf()


def show_dB(metal_types=None, num_atoms=None, shape=None, composition=None):
    res = db_inter.get_polymet_result(num_atoms=num_atoms, metals=metal_types, shape=shape, return_list=True,
                                      composition=composition)

    print(f"Found {len(res)} results")

    # Showing Specific NP's
    res[0].show()


def vis_layer_comps(result: db_inter.get_polymet_result()):
    bcm = BCModel(result.atoms_obj)
    layers = bcm.shell_map
    layer_comp = []
    for i in sorted(layers):
        n = len(layers[i])
        layer_atoms = ase.Atoms(result.atoms_obj[layers[i]])
        comp = []
        for m in result.metals:
            n_m = (layer_atoms.symbols == m).sum()
            x_m = n_m       #number of atoms in each layer
            # x_m = n_m / n #percentages of atoms in each layer
            comp.append(x_m)
        layer_comp.append(comp)
    return (layer_comp)


def layer_plot(metals_used=None, compositions=None, num_atoms=55, shape="icosahedron"):
    res: datatables.PolymetallicResults
    res = db_inter.get_polymet_result(metals=metals_used, composition=compositions, num_atoms=num_atoms,
                                      shape=shape)
    current_layer_comp = vis_layer_comps(res)

    dF_dict = {}

    for i in range(len(current_layer_comp)):
        layer = np.array([])
        for k in range(len(current_layer_comp[i])):
            layer = np.append(layer, current_layer_comp[k][i])
        dF_dict[metals_used[i]] = list(layer)


    title = res.get_chemical_formula()+'_barchart'
    df = pd.DataFrame(dF_dict)
    symbols = list(chemical_symbols)
    colors = [list(jmol_colors[symbols.index(m)]) for m in metals_used]
    plot = df.plot(kind='bar', stacked=True, title=title, mark_right=True, color=colors)
    fig = plot.get_figure()
    fig.savefig("C:/Users/Brenno_Ferreira/Documents/Research_CANELa/Comp_Plots/" + title + ".png")


def cn_neighbor_analysis(metals_used, composition, num_atoms=55, shape="icosahedron",
                         savepath=None, percentage=False, group_by_srf_bulk=False):
    res: datatables.PolymetallicResults
    res = db_inter.get_polymet_result(metals=metals_used, composition=composition, num_atoms=num_atoms, shape=shape)

    bcm = BCModel(res.atoms_obj)
    symbols = list(chemical_symbols)

    formula = bcm.atoms.numbers
    metals = {}
    index = {}
    for i, metal in enumerate(metals_used):
        metals[metal] = {}
        # metals[metal] = np.zeros((len(metals_used)))
        index[metal] = i

    all_cns = set()
    for bond in bcm.coord_dict:
        metal1 = symbols[formula[bond]]
        cn = bcm.cn[bond]

        # set cn to bulk or surface instead
        if group_by_srf_bulk:
            cn = 'Bulk' if cn == 12 else 'Surface'

        all_cns.add(cn)
        if cn not in metals[metal1]:
            metals[metal1][cn] = [0] * len(metals_used)
        bonded_atoms = bcm.coord_dict[bond]
        for element in bonded_atoms:
            metal2 = symbols[formula[element]]
            metals[metal1][cn][index[metal2]] += 1

    # make sure all cns exist for each metal (even if 0s)
    for m in metals:
        for cn in all_cns:
            if cn not in metals[m]:
                metals[m][cn] = [0] * len(metals_used)

    first_col_name = 'Position' if group_by_srf_bulk else 'CN'
    columns = [first_col_name] + metals_used
    dfs = {}
    for m in metals:
        rows = []
        for cn in sorted(metals[m]):
            counts = metals[m][cn]
            if percentage:
                total = sum(counts)
                if total:
                    counts = [c / total for c in counts]

            rows.append([cn] + counts)

        dfs[m] = pd.DataFrame(rows, columns=columns).set_index(first_col_name)

    symbols = list(chemical_symbols)
    colors = [list(jmol_colors[symbols.index(m)]) for m in metals_used]

    fig, axes = plt.subplots(1, len(dfs), figsize=(12, 5))
    ylabel_set = False
    for (m, df), ax in zip(dfs.items(), axes):
        df.plot(kind='bar', stacked=True, color=colors, ax=ax,
                title=m, legend=False)

        #########################################################################################
        # NOTE: taken from stackoverflow answer:
        # https://stackoverflow.com/questions/41296313/stacked-bar-chart-with-centered-labels
        # .patches is everything inside of the chart
        for rect in ax.patches:
            # Find where everything is located
            height = rect.get_height()
            width = rect.get_width()
            x = rect.get_x()
            y = rect.get_y()

            # The height of the bar is the data value and can be used as the label
            if percentage:
                label_text = f'{height:.2%}'
            else:
                label_text = f'{int(height)}'

            # ax.text(x, y, text)
            label_x = x + width / 2
            label_y = y + height / 2

            # plot only when height is greater than specified value
            if height > 0:
                ax.text(label_x, label_y, label_text, ha='center', va='center', fontsize=10)
        #########################################################################################

        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        if group_by_srf_bulk:
            ax.set_xlabel(None)

        if not ylabel_set:
            if percentage:
                ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
                ax.set_ylabel('% Neighbor Type')
                ax.set_ylim(ymax=1.1)
            else:
                ax.set_ylabel('Neighbor Type Count')

            ylabel_set = True

    title = f'{res.get_chemical_formula(latex=True)} {res.shape}'
    fig.suptitle(title)
    fig.tight_layout()
    ax.legend(frameon=False)

    filename = res.get_chemical_formula() + '_cn_neighbor'
    fig.savefig(r"C:/Users/Brenno_Ferreira/Documents/Research_CANELa/Neighbor_Analysis/" + filename + ".png")


def bonded_chart(metals_used, composition, num_atoms=55, shape="icosahedron", percentage=False):
    res: datatables.PolymetallicResults
    res = db_inter.get_polymet_result(metals=metals_used, composition=composition, num_atoms=num_atoms, shape=shape)
    bcm = BCModel(res.atoms_obj)
    symbols = list(chemical_symbols)
    formula = bcm.atoms.numbers
    metals = {}
    index = {}

    for i, metal in enumerate(metals_used):
        metals[metal] = np.zeros((len(metals_used)))
        index[metal] = i
    for bond in bcm.coord_dict:
        bonded_atoms = bcm.coord_dict[bond]
        metal1 = symbols[formula[bond]]
        for element in bonded_atoms:
            metal2 = symbols[formula[element]]
            metals[metal1][index[metal2]] += 1

    # Note the latex argument, which returns a string that will format for figures
    title = f'{res.get_chemical_formula(latex=True)} {res.shape}\n Neighbor Analysis'

    if percentage:
        for m in metals:
            metals[m] = metals[m] / sum(metals[m])

    df = pd.DataFrame(metals, index=metals_used).transpose()
    symbols = list(chemical_symbols)
    colors = [list(jmol_colors[symbols.index(m)]) for m in metals_used]
    ax = df.plot(kind='bar', stacked=True, title=title, mark_right=True, color=colors,
                 edgecolor='k', lw=1)
    ax.set_xticklabels(metals_used, rotation=0)
    ax.legend(frameon=False, ncol=len(metals_used))

    #########################################################################################
    # NOTE: taken from stackoverflow answer:
    # https://stackoverflow.com/questions/41296313/stacked-bar-chart-with-centered-labels
    # .patches is everything inside of the chart
    for rect in ax.patches:
        # Find where everything is located
        height = rect.get_height()
        width = rect.get_width()
        x = rect.get_x()
        y = rect.get_y()

        # The height of the bar is the data value and can be used as the label
        if percentage:
            label_text = f'{height:.2%}'
        else:
            label_text = f'{int(height)}'

        # ax.text(x, y, text)
        label_x = x + width / 2
        label_y = y + height / 2

        # plot only when height is greater than specified value
        if height > 0:
            ax.text(label_x, label_y, label_text, ha='center', va='center', fontsize=10)
    #########################################################################################

    if percentage:
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax.set_ylabel('% Bond Type')
        ax.set_ylim(ymax=1.1)
    else:
        ax.set_ylabel('Bond Type Count')

    fig = ax.get_figure()
    fig.tight_layout()

    filename = res.get_chemical_formula() + '_bondchart'
    fig.savefig(r"C:/Users/Brenno_Ferreira/Documents/Research_CANELa/Bond_Plots/" + filename + ".png")


def create_xyz_files(metal_types, num_atoms, shape, tested_comps):
    path = f"C:/Users/Brenno_Ferreira/Documents/Research_CANELa/trimetallic_nps/"

    res = db_inter.get_polymet_result(num_atoms=num_atoms, metals=metal_types, shape=shape, composition=tested_comps)
    gapath = path + res.get_chemical_formula() + '_ga-opt.xyz'

    res.save_np(gapath)

    atoms = res.atoms_obj.copy()
    numbers = atoms.numbers.copy()
    np.random.shuffle(numbers)
    atoms.numbers = numbers
    bcm = BCModel(atoms)
    ordering = np.array([bcm.metal_types.index(s) for s in atoms.symbols])
    atoms.info['shape'] = shape
    atoms.info['CE'] = bcm.calc_ce(ordering)
    if atoms.info['CE'] <= res.CE:
        raise ValueError('Random has same/lower CE?')

    atoms.info['EE'] = bcm.calc_ee(ordering)
    atoms.info['Smix'] = bcm.calc_smix(ordering)
    atoms.info['runtype'] = 'random'
    randompath = path + atoms.get_chemical_formula() + '_random.xyz'
    atoms.write(randompath)
    print(f'Completed {res.get_chemical_formula()}.')


def dataOpener(jsonfilepath):

    #For Values not found in JSON file
    DOES_NOT_EXIST = 0.00

    #Openingn JSON file that is in the same folder as scratch file
    with open(jsonfilepath) as f:
        data = json.load(f)

    #Initializing the Gamma Dictionary
    gamma = {}

    for M1 in data['cebulk']:

        #Creating a temporary dictionary to append and add onto the gamma dictionary
        subEle = {}

        for M2 in data['cebulk']:

            #check for Heterolytic values
            if M1 == M2:
                subEle[M2] = 1.00

            #Check for if the values exist
            elif M1+M2 in data['bde']:
                temp , subEle[M2] = gammaSolver(data['bde'][M1+M1],data['bde'][M2+M2],data['bde'][M1+M2])

            elif M2+M1 in data['bde']:
                temp, subEle[M2] = gammaSolver(data['bde'][M1 + M1], data['bde'][M2 + M2], data['bde'][M2 + M1])

            #For when values do not exist within the JSON File
            elif M1 + M2 not in data['bde']:
                subEle[M2] = DOES_NOT_EXIST

            elif M2 + M1 not in data['bde']:
                subEle[M2] = DOES_NOT_EXIST

        gamma[M1] = subEle

    return data['cebulk'], gamma


def gammaSolver(BDEM1 , BDEM2 , BDEM1M2):
    A = np.array([[BDEM1,BDEM2],[1,1]])
    B = np.array([2*BDEM1M2,2])
    X = np.linalg.solve(A,B)
    (gamma_M1,gamma_M2) = X
    return round(gamma_M1,3) , round(gamma_M2,3)

ce_bulk , gamma = dataOpener('dft_data.json')

# temps = [298.15, 640, 1073]
# tested_comps = [[18, 18, 19], [14, 14, 27], [14, 27, 14], [27, 14, 14]]
res = db_inter.get_polymet_result(num_atoms=147, metals=['Au','Cu','Pd'], shape='icosahedron', composition= [10,23,114])
view(res.atoms_obj)
# print(res)

# atoms = getattr(structure_gen.NPBuilder, 'icosahedron')(3)
# bcm = BCModel(atoms, metal_types=['Au', 'Cu', 'Pd'])
# batch_GA_run(bcm, shape ='icosahedron', max_gens=-1, max_nochange=2000)


# res: datatables.PolymetallicResults
# res = db_inter.get_polymet_result(metals=['Au', 'Pd', 'Pt'], composition=[432,1460,165])
# vis_layers(res)
#
# print(vis_layer_comps(res))
# cn_neighbor_analysis(res.metals, res.composition, res.num_atoms)

# atoms = getattr(structure_gen.NPBuilder, 'icosahedron')(8)
# bcm = BCModel(atoms, metal_types=['Au', 'Pd', 'Pt'])
# ga = GA(bcm, [432,1460,165], 'icosahedron')
# ga.run(max_gens=-1, max_nochange=2000)
# ga.save_to_db()

exit()


bcm_dft = BCModel(atoms, metal_types=['Ag','Au', 'Cu', 'Pd', 'Pt'])
bcm_exp = BCModel(atoms, metal_types=['Ag','Au', 'Cu', 'Pd', 'Pt'])

bcm_dft.ce_bulk = ce_bulk
bcm_dft.gammas = gamma

bcm_dft._get_precomps()


metal_list = [['Ag', 'Au', 'Pt'],['Ag', 'Au', 'Cu']]
tested_comps = [[14, 14, 27], [14, 27, 14], [27, 14, 14]]

comp_list = [[[0,27,0, 14, 14], [0,14,0, 27, 14], [0,14,0, 14, 27]],[ [27, 14, 14,0,0], [14, 27, 14,0,0], [14, 14, 27,0,0]]]
dft_comps = np.array(comp_list).astype(int)

exp_res = [[],[]]
dft_res = [[],[]]

metal_types = ['Ag','Au', 'Cu', 'Pd', 'Pt']

for i, comps in enumerate(tested_comps):
    for k, metals in enumerate(metal_list):
        res = db_inter.get_polymet_result(num_atoms=55, metals=metals, shape='icosahedron',
                                          composition=comps)
        res: datatables.PolymetallicResults
        print(dft_comps[k][i])
        ordering = np.array([metal_types.index(s) for s in res.atoms_obj.symbols])

        exp_res[k].append(bcm_exp.calc_ce(ordering))
        dft_res[k].append(bcm_dft.calc_ce(ordering))


print(exp_res,"EXP RESULTS")
print(dft_res, "DFT RESULTS")


# Order Goes: Ag Au Cu Pd Pt



# ga = GA(bcm, comps, 'icosahedron')
# ga.run(max_gens=-1, max_nochange=2000)
# ga.save_to_db()

# for comps in tested_comps:
#     atoms = getattr(structure_gen.NPBuilder, 'icosahedron')(9)
#     bcm = BCModel(atoms, metals)
#     ga = GA(bcm, comps, 'icosahedron')
#     ga.run(max_gens=-1, max_nochange=2000)
#     ga.save_to_db()

# cn_neighbor_analysis(tested_metals[1], [100,100,361], num_atoms=561, shape='icosahedron', percentage=False)



