import sys
import random
import numpy as np
import matplotlib.pyplot as plt

# random.seed(9876)


class Chromo(object):
    def __init__(self, atomg, n_dope=0, arr=None):
        self.n_atoms = atomg.n_atoms
        self.n_dope = n_dope
        self.atomg = atomg
        self.arr = np.zeros(atomg.n_atoms).astype(int)

        if n_dope > self.n_atoms:
            raise ValueError("Can't dope more atoms than there are atoms...")

        if isinstance(arr, np.ndarray):
            self.arr = arr
            self.n_dope = arr.sum()
        else:
            self.arr[:n_dope] = 1
            random.shuffle(self.arr)
            self.arr = np.array(self.arr)

        # calculate initial CE
        self.calc_score()

    # create getAtomicCE method
    def get_atomic_ce(self, i):
        return self.atomg.getAtomicCE(i, self.arr)

    def mutate(self, nps=1):
        if not self.n_dope:
            print('Warning: attempting to mutate, but system is monometallic')
            return

        # shift a "1" <nps> times
        for i in range(nps):
            ones = list(np.where(self.arr == 1)[0])
            s = random.sample(ones, 1)[0]
            self.arr[s] = 0

            # shift '1' over to the left
            shift = s - 1
            while 1:
                if shift not in ones:
                    self.arr[shift] = 1
                    break
                shift -= 1

        self.calc_score()

    def cross(self, chrom2):
        x = 0

        # column = child
        children = np.zeros((self.n_atoms, 2)).astype(int)
        swapped = 0
        for i in range(self.n_atoms - 1, -1, -1):
            if self.arr[i] == chrom2.arr[i]:
                children[i, :] = self.arr[i]
            else:
                children[i, x] = 1
                children[i, x-1] = 0
                x = 0 if x else 1
                swapped += 1
                if swapped >= self.n_dope - 4:
                    break

        return [Chromo(self.atomg, arr=children[:, 0]),
                Chromo(self.atomg, arr=children[:, 1])]

    def calc_score(self):
        self.score = self.atomg.getTotalCE(self.arr)


class Pop(object):
    def __init__(self, atomg, n_dope=1, popsize=100, kill_rate=0.2,
                 mate_rate=0.25, mute_rate=0.1, mute_num=1):
        self.pop = [Chromo(atomg, n_dope=n_dope) for i in range(popsize)]
        self.sort_pop()
        self.info = []
        self.stats()

        self.nkill = int(popsize * kill_rate)
        self.nmut = int(popsize * mute_rate - self.nkill)
        self.n_mate = int(popsize * kill_rate * mate_rate)
        self.mute_num = mute_num
        self.mute_rate = mute_rate
        self.kill_rate = kill_rate
        self.popsize = popsize

    def step(self):
        self.pop = self.pop[:self.nkill]

        mated = 0
        # start mating from the 2nd down
        tomate = 1
        while len(self.pop) < self.popsize:
            if mated < self.n_mate:
                n1, n2 = tomate, tomate + 1
                # n1, n2 = random.sample(range(len(self.pop)), 2)
                self.pop += self.pop[n1].cross(self.pop[n2])
                mated += 2
                tomate += 1
            else:
                newc = Chromo(self.pop[0].atomg, self.pop[0].n_dope)
                self.pop += [newc]

        self.pop = self.pop[:self.popsize]

        for j in range(self.nmut):
            self.pop[random.randrange(1, self.popsize)].mutate(self.mute_num)

        self.sort_pop()
        self.stats()

    def run(self, nsteps, std_cut=0):
        for i in range(int(nsteps)):
            self.step()
            print('\tMin: %.5f eV \t %i' % (self.info[-1][0], i), end='\r')
            # if STD less than std_cut end the GA
            if self.info[-1][-1] < std_cut:
                break
        self.info = np.array(self.info)

    def sort_pop(self):
        self.pop = sorted(self.pop,
                          key=lambda j: j.score)

    def stats(self):
        s = np.array([i.score for i in self.pop])
        self.info.append([s[0],
                          s.mean(),
                          s.std()])

if __name__ == '__main__':
    from atomgraph import AtomGraph
    from adjacency import buildAdjacencyList
    import ase.cluster

    metal1 = 'Cu'
    metal2 = 'Au'

    n_dope = 8
    max_runs = 1E2
    pop = 50
    kill_rate = 0.2
    mate_rate = 0.4
    mute_rate = 0.15

    atom = ase.cluster.Icosahedron('Cu', 3)
    adj = buildAdjacencyList(atom)
    ag = AtomGraph(adj, metal1, metal2)

    p = Pop(ag, n_dope=n_dope, popsize=pop, mate_rate=mate_rate,
            mute_rate=mute_rate, kill_rate=kill_rate)
    p.run(max_runs)

    res = ' Min: %.5f\nMean: %.3f\n STD: %.3f\n' % tuple(p.info[-1, :])
    res += 'Mute: %.2f\nKill: %.2f\n' % (p.mute_rate, p.kill_rate)
    res += ' Pop: %i\n' % p.popsize
    res += 'nRun: %i\n' % max_runs
    res += 'Form: %s%i_%s%i\n' % (metal1, len(atom) - n_dope, metal2, n_dope)
    res += 'Done: %i\n' % (len(p.info) - 1) + '\n\n\n'

    # print results to terminal
    print(res.strip('\n'))

    with open('results.txt', 'a') as fid:
        fid.write(res)

    # see if results are new max
    with open('best.txt', 'r') as rfid:
        best = float(rfid.readline().strip('\n').split()[-1])

    # if new max, write it to best.txt
    if best > p.info[-1, 0]:
        print('NEW MIN!'.center(50, '-'))
        with open('best.txt', 'w') as bestfid:
            bestfid.write(res)

    fig, ax = plt.subplots(figsize=(7, 7))

    ax.fill_between(range(len(p.info)), p.info[:, 1] + p.info[:, 2],
                    p.info[:, 1] - p.info[:, 2], color='lightblue',
                    label='STD')
    # ax.plot(range(len(p.info)), p.info[:, 1] + p.info[:, 2])
    # ax.plot(range(len(p.info)), p.info[:, 1] - p.info[:, 2])

    ax.plot(p.info[:, 1], color='k', label='MEAN')
    ax.plot(p.info[:, 0], ':', color='k', label='MIN')
    ax.legend()
    ax.set_ylabel('Score')
    ax.set_xlabel('Step')
    ax.set_title('Min Val: %.5f' % (p.pop[0].score))
    fig.tight_layout()
    fig.show()

    for i, dope in enumerate(p.pop[0].arr):
        atom[i].symbol = metal2 if dope else metal1

    atom.write("C:/users/yla/desktop/%s%i_%s%i.xyz" % (metal1,
                                                    len(atom) - n_dope,
                                                    metal2,
                                                    n_dope))
