import ase
import numpy as np


class Bytestring(object):
    def __init__(self, string=None):
        """
    Stores all the permutations of a set of bytes with a constant number of 1s and 0s.
    """
        if string == None:
            # Generate a random string for testing purposes if none present
            print "No string supplied; generating a random string 6 characters in length."
            self.string = ""
            for index in range(0, 6):
                self.string += str(random.randint(0, 1))
            print self.string
        else:
            self.string = string
        self.original_string = self.string
        # Come up with a count of how many 1's are present in the string
        # Used later when we set up our permutator
        self.onecount = 0
        for i in self.string:
            if i == "1":
                self.onecount += 1
        # Generator object for permutations
        self._mutate_gen = self.mutator()
        self.mutate()

    def mutator(self):
        """
    Returns a generator for permutations without repeats.
    """
        for combos in itertools.combinations(range(0, len(self.original_string)), self.onecount):
            new_string = [0] * len(self.original_string)
            for index in combos:
                new_string[index] = 1
            yield new_string

    def mutate(self):
        """
    Return the next permutation in the set.
    """
        try:
            self.string = "".join(str(i) for i in self._mutate_gen.next())
        except StopIteration:
            print "Warning: Reached end of combinations. Restarting from beginning."
            self._mutate_gen = self.mutator()
            self.string = self.original_string


def max_distance(index, cn_list, cluster):
    return np.argmax(cluster.get_distances(index, cn_list))


def generate_pairs(cn_list, cluster):
    dupes = dict.fromkeys(cn_list)
    for i in cn_list:
        dupes[i] = cn_list[max_distance(i, cn_list, cluster)]
    # Trim down the list to avoid redundantly choosing the same axis twice
    # Because we don't have many axes of rotation, this isn't too inefficient
    saved_list = []
    cn_pairs = {}
    for i in dupes:
        if i in saved_list or dupes[i] in saved_list:
            pass
        else:
            saved_list.append(i)
            saved_list.append(dupes[i])
            cn_pairs[i] = dupes[i]
    return cn_pairs


def calculate_axes(cn_pairs, cluster):
    axes = []
    for atom1_index in cn_pairs:
        # Pull the un-normalized vector
        atom2_index = cn_pairs[atom1_index]
        atom1_position = cluster[atom1_index].position
        atom2_position = cluster[atom2_index].position
        raw_vector = atom1_position - atom2_position

        # Normalize the vector
        vector = raw_vector / np.linalg.norm(raw_vector)
        axes.append(vector)
    return axes


def create_rotation_matrix(vector, degree):
    theta = np.deg2rad(degree)
    ux = vector[0]
    uy = vector[1]
    uz = vector[2]
    c = np.cos(theta)
    s = np.sin(theta)
    matrix = np.matrix([[theta * ux ** 2 + c,
                         theta * ux * uy - s * uz,
                         theta * ux * uz + s * uy],

                        [theta * ux * uy + s * uz,
                         theta * uy ** 2 + c,
                         theta * uy * uz - s * ux],

                        [theta * ux * uz - s * uy,
                         theta * uy * uz + s * ux,
                         theta * uz ** 2 + c]
                        ])
    return matrix


def calc_icos_symmetry():
    # Pull cn6 and cn9 from the 4-layer
    # Matrices shouldn't vary with the number of layers anyway
    x = ase.cluster.Icosahedron("Cu", 4)
    x.center(about=0)
    y = np.bincount(ase.neighborlist.neighbor_list("i", x, 3))

    cn6 = []
    cn9 = []

    for count, cn in enumerate(y):
        if cn == 6:
            cn6.append(count)
        elif cn == 9:
            cn9.append(count)

    cn6_pairs = generate_pairs(cn6, x)
    cn9_pairs = generate_pairs(cn9, x)

    cn6_axes = calculate_axes(cn6_pairs, x)
    cn9_axes = calculate_axes(cn9_pairs, x)

    # Pull cn8 from 3-layer
    x = ase.cluster.Icosahedron("Cu", 3)
    x.center(about=0)
    y = np.bincount(ase.neighborlist.neighbor_list("i", x, 3))

    cn8 = []

    for count, cn in enumerate(y):
        if cn == 8:
            cn8.append(count)
        else:
            pass

    cn8_pairs = generate_pairs(cn8, x)
    cn8_axes = calculate_axes(cn8_pairs, x)

    print "Axes discovered are:"
    print str(len(cn6_axes)) + " C5 axes about points"
    for i in cn6_axes:
        print i
    print "\n"
    print str(len(cn8_axes)) + " C2 axes about edges"
    for i in cn8_axes:
        print i
    print "\n"
    print str(len(cn9_axes)) + " C3 axes about faces"
    for i in cn9_axes:
        print i
    print "\n"
