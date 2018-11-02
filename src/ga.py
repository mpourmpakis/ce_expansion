import os
import sys
import time
import random
from atomgraph import AtomGraph
from adjacency import buildAdjacencyList
import ase.cluster
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

        assert self.arr.sum() == self.n_dope
        # shift a "1" <nps> times
        for i in range(nps):
            ones = list(np.where(self.arr == 1)[0])
            s = random.sample(ones, 1)[0]
            self.arr[s] = 0

            # shift '1' over to the left
            shift = s - 1
            while 1:
                if self.arr[shift] == 0:
                    self.arr[shift] = 1
                    break
                shift -= 1

        assert self.arr.sum() == self.n_dope
        self.calc_score()

    def cross(self, chrom2):
        x = 0

        child1 = self.arr.copy()
        child2 = chrom2.arr.copy()
        s1 = s2 = 2
        ones = np.where(child1 == 1)[0]
        diff = np.where(child1 != child2)[0][::-1]
        for i in diff:
            if i in ones:
                if s1:
                    child1[i] = 0
                    child2[i] = 1
                    s1 -= 1
            elif s2:
                child1[i] = 1
                child2[i] = 0
                s2 -= 1
            if s1 == s2 == 0:
                break

        assert child1.sum() == child2.sum() == self.n_dope
        return [Chromo(self.atomg, n_dope=self.n_dope, arr=child1),
                Chromo(self.atomg, n_dope=self.n_dope, arr=child2)]

    def calc_score(self):
        self.score = self.atomg.getTotalCE(self.arr)


class Pop(object):
    def __init__(self, atomg, n_dope=1, popsize=100, kill_rate=0.2,
                 mate_rate=0.25, mute_rate=0.1, mute_num=1):
        self.atomg = atomg
        self.popsize = popsize
        self.n_dope = n_dope

        # build random population
        self.build_pop()

        self.sort_pop()
        self.info = []
        self.stats()

        self.nkill = int(popsize * kill_rate)
        self.nmut = int((popsize - self.nkill) * mute_rate)
        self.n_mate = int(popsize * kill_rate * mate_rate)
        self.mute_num = mute_num
        self.mute_rate = mute_rate
        self.kill_rate = kill_rate

        # track runtime
        self.runtime = None

    def build_pop(self):
        self.pop = [Chromo(self.atomg, n_dope=self.n_dope)
                    for i in range(self.popsize)]

    def get_min(self):
        return self.info[:, 0].min()

    def step(self, rand=False):
        if rand:
            self.build_pop()
        else:
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
                    self.pop.append(Chromo(self.pop[0].atomg,
                                           n_dope=self.n_dope))

            self.pop = self.pop[:self.popsize]

            for j in range(self.nmut):
                self.pop[random.randrange(1,
                                          self.popsize)].mutate(self.mute_num)

        self.sort_pop()
        self.stats()

    def run(self, nsteps, std_cut=0, rand=False):
        start = time.time()
        for i in range(int(nsteps)):
            print('\tMin: %.5f eV \t %i' % (self.info[-1][0], i), end='\r')
            self.step(rand)
            # if STD less than std_cut end the GA
            if self.info[-1][-1] < std_cut:
                break
        print('\tMin: %.5f eV \t %i' % (self.info[-1][0], i + 1), end='\r')
        self.info = np.array(self.info)
        self.runtime = time.time() - start

    def sort_pop(self):
        self.pop = sorted(self.pop,
                          key=lambda j: j.score)

    def stats(self):
        s = np.array([i.score for i in self.pop])
        self.info.append([s[0],
                          s.mean(),
                          s.std()])


def results_str(pop, disp=True):
    res = ' Min: %.5f\nMean: %.3f\n STD: %.3f\n' % tuple(p.info[-1, :])
    res += 'Mute: %.2f\nKill: %.2f\n' % (p.mute_rate, p.kill_rate)
    res += ' Pop: %i\n' % p.popsize
    res += 'nRun: %i\n' % max_runs
    res += 'Form: %s%i_%s%i\n' % (metal1, len(atom) - n_dope, metal2, n_dope)
    res += 'Done: %i\n' % (len(p.info) - 1) + '\n\n\n'
    if disp:
        print(res.strip('\n'))
    return res


def log_results(results):
    with open('results.txt', 'a') as fid:
        fid.write(results)

    # see if results are new max
    with open('best.txt', 'r') as rfid:
        best = float(rfid.readline().strip('\n').split()[-1])

    # if new max, write it to best.txt
    if best > p.info[-1, 0]:
        print('NEW MIN!'.center(50, '-'))
        with open('best.txt', 'w') as bestfid:
            bestfid.write(results)


def make_plot(pop):
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
    return fig, ax


def make_xyz(atom, arr, metals, path=None):
    for i, dope in enumerate(p.pop[0].arr):
        atom[i].symbol = metal2 if dope else metal1

    if not path:
        n_dope = sum(arr)
        path = os.path.expanduser('~') + \
            '/desktop/%s%i_%s%i.xyz' % (metals[0], len(atom) - n_dope,
                                        metals[1], n_dope)
    atom.write(path)
    print('Saved as %s' % path)
    return atom


if __name__ == '__main__':
    desk = os.path.expanduser('~') + '/desktop/'
    metal1 = 'Cu'
    metal2 = 'Au'

    n_dope = 27
    max_runs = 50
    pop = 500
    kill_rate = 0.8
    mate_rate = 0.8
    mute_rate = 0.2
    mute_num = 1

    atom = ase.cluster.Icosahedron('Cu', 3)
    adj = buildAdjacencyList(atom)
    ag = AtomGraph(adj, metal1, metal2)

    # track properties effect on runtime
    tot_pops = []
    tot_rts = []
    tot_mins = []

    n_its = 10
    rts = np.zeros(n_its)
    mins = np.zeros(n_its)

    # random runs data
    rand_mins = np.array([-3.2444183, -3.23168432,  0.00598345])

    x = np.linspace(0.1, 0.9, 9)
    ce = 0
    min_struct = None
    xlabel = 'Mate Rate'
    for mate_rate in x:
        print('%s: %.2f' % (xlabel, mate_rate))
        for j in range(n_its):
            p = Pop(ag, n_dope=n_dope, popsize=pop, mate_rate=mate_rate,
                    mute_rate=mute_rate, kill_rate=kill_rate,
                    mute_num=mute_num)
            p.run(max_runs)
            if ce > p.get_min():
                ce = p.get_min()
                min_struct = p.pop[0]
            print(' ' * 50, end='\r')

            # pops.append(p.popsize)
            rts[j] = p.runtime
            mins[j] = p.get_min()

        tot_rts.append([rts.min(), rts.mean(), rts.std()])
        tot_mins.append([mins.min(), mins.mean(), mins.std()])

    formula = metal1 + str(len(atom) - n_dope) + metal2 + str(n_dope)
    make_xyz(atom, min_struct.arr, [metal1, metal2],
             desk + '/%s.xyz' % formula)
    print('Min CE: %.5f eV' % ce)

    f, a = make_plot(p)
    plt.show()
    tot_rts = np.array(tot_rts)
    tot_mins = np.array(tot_mins)

    plt.close('all')
    fig, ax = plt.subplots(2, 1, sharex=True)
    ax1, ax2 = ax

    ax1.errorbar(x, tot_rts[:, 1], yerr=tot_rts[:, 2], color='blue',
                 label='Runtimes (s)', ecolor='k', marker='o',
                 capsize=5, markeredgecolor='k', linestyle='')

    ax1.plot(x, tot_rts[:, 0], 'x', color='k')
    # ax1.plot(x, rts, color='red', label='Runtimes (s)')
    ax1.set_ylabel('Runtime for %i Runs (s)' % max_runs)

    ax2.errorbar(x, tot_mins[:, 1], yerr=tot_mins[:, 2], color='red',
                 label='GA', ecolor='k', marker='o',
                 capsize=5, markeredgecolor='k', linestyle='')

    ax2.plot(x, tot_mins[:, 0], 'x', color='k')

    # plot results from random iterations
    ax2.errorbar(0, rand_mins[1], yerr=rand_mins[2], color='gold',
                 label='Random', ecolor='k', marker='o',
                 capsize=5, markeredgecolor='k', linestyle='')

    ax2.plot(0, rand_mins[0], 'x', color='k')

    # ax2.plot(x, mins, color='blue', label='CE (eV)')
    ax2.set_ylabel('Minimum CE (eV)')
    ax2.set_xlabel(xlabel)
    ax2.legend()
    # ax2.axes.invert_yaxis()
    # fig.legend(loc='upper center')
    fig.tight_layout()
    fig.savefig(desk + '/rt_%s_%i_dope%i.png' % (xlabel.replace(' ',
                                                                '').lower(),
                                                 pop, n_dope))
    fig.show()
    # res = results_str(p)
    # log_results(res)
