import numpy as np
import matplotlib.pyplot as plt

class BernoulliBandit:
    def __init__(self,K):
        self.probs = np.random.uniform(size=K)
        self.best_idx = np.argmax(self.probs)
        self.best_prob = self.probs[self.best_idx]
        self.K = K

    def step(self,k):
        if np.random.rand() < self.probs[k]:
            return 1
        else:
            return 0

class Solver:
    def __init__(self,bandit):
        self.bandit = bandit
        self.counts = np.zeros(self.bandit.K)
        self.regret = 0
        self.actions = []
        self.regrets = []

    def update_regret(self,k):
        self.regret += (self.bandit.best_prob - self.bandit.probs[k])
        self.regrets.append(self.regret)

    def run_one_step(self):
        return NotImplementedError

    def run(self,num_steps):
        for _ in range(num_steps):
            k = self.run_one_step()
            self.counts[k] += 1
            self.actions.append(k)
            self.update_regret(k)

class EpsilonGreedy(Solver):
    def __init__(self,bandit,epsilon=0.01,init_prob=1.0):
        super(EpsilonGreedy,self).__init__(bandit)
        self.epsilon = epsilon
        self.estimates = np.array([init_prob] * self.bandit.K)

    def run_one_step(self):
        if np.random.random() < self.epsilon:
            k = np.random.randint(0,self.bandit.K)
        else:
            k = np.argmax(self.estimates)
        r = self.bandit.step(k)
        self.estimates[k] += 1. / (self.counts[k] + 1) * (r - self.estimates[k])
        return k

def plot_results(solvers, solver_names):
    for idx,solver in enumerate(solvers):
        time_list = range(len(solver.regrets))
        plt.plot(time_list,solver.regrets,label=solver_names[idx])
    plt.xlabel('Time steps')
    plt.ylabel('Cumulative regrets')
    plt.title('%d-armed bandit' % solvers[0].bandit.K)
    plt.legend()
    plt.show()

np.random.seed(500)
K = 10
bandit_10_arm = BernoulliBandit(K)

#贪心算法
epsilons = [1e-4, 0.01, 0.1, 0.25, 0.5]
epsilon_greedy_solver_list = [
    EpsilonGreedy(bandit_10_arm,epsilon = e) for e in epsilons
]
epsilon_greedy_solver_names = ["epsilon={}".format(e) for e in epsilons]
for solver in epsilon_greedy_solver_list:
    solver.run(5000)

plot_results(epsilon_greedy_solver_list,epsilon_greedy_solver_names)
