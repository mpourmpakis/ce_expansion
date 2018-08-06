import ase.cluster
import ase.io


def shell_separator(shell_num):
    nanop = ase.cluster.Icosahedron("Cu", shell_num)
    #Generates icosahedral nanoparticle with specified number of shells

    idshell = (2 * (shell_num - 2) + 1) * (5 * ((shell_num - 2) ** 2) + 5 * (shell_num - 2) + 3) / 3
    #Calculates the number of particles within the shells below the outermost shell

    for x in range(0, idshell):
        nanop.pop(0)
        # Removes all atoms except those in the outermost shell

    return nanop
