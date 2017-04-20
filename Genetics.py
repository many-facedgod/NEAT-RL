from __future__ import print_function
import Network
import numpy as np
from Config import *
import copy


def rand():
    return 1 - 2 * np.random.random()


class GeneN:
    def __init__(self, id, is_input, is_output, bias):
        self.id = id
        self.is_input = is_input
        self.is_output = is_output
        self.bias = bias

    def mutate_bias(self):
        self.bias += np.random.normal() * RATE
        self.bias = np.clip(self.bias, MIN, MAX)


class GeneE:
    next_hm = 0
    history = {}

    def __init__(self, source, dest, weight, active, HM=None):
        self.source = source
        self.dest = dest
        self.weight = weight
        self.active = active
        if HM is not None:
            self.hm = HM
        else:
            try:
                hm = GeneE.history[(source, dest)]
            except KeyError:
                self.hm = GeneE.get_new_hm()
                GeneE.history[(source, dest)] = self.hm
            else:
                self.hm = hm
        self.id = (source, dest)

    def toggle(self):
        self.active = not self.active

    def mutate_weight(self):
        self.weight += np.random.normal() * RATE
        self.weight = np.clip(self.weight, MIN, MAX)

    @classmethod
    def get_new_hm(cls):
        cls.next_hm += 1
        return cls.next_hm - 1

    @classmethod
    def clear_history(cls):
        cls.history = {}


class Chromosome:
    def __init__(self, n_inputs, n_outputs):
        self.species = 0
        self.nodes = []
        self.edges = {}
        self.n_inputs = n_inputs
        self.inputs = []
        self.outputs = []
        self.hidden = []
        self.n_nodes = 0
        self.n_edges = 0
        self.n_outputs = n_outputs
        self.fitness = 0

    def create_input_output(self):
        for i in xrange(self.n_inputs):
            self.add_node(True, False, 0)
        for i in xrange(self.n_outputs):
            self.add_node(False, True, 0)

    def add_node(self, is_input, is_output, bias):
        id = self.n_nodes
        self.n_nodes += 1
        n = GeneN(id, is_input, is_output, bias)

        self.nodes.append(n)
        if is_input:
            self.inputs.append(n)
        elif is_output:
            self.outputs.append(n)
        else:
            self.hidden.append(n)
        return id

    def add_edge(self, source, dest, weight, active, HM=None):
        id = (source, dest)
        self.n_edges += 1
        self.edges[id] = GeneE(source, dest, weight, active, HM)

        return id

    def make_sparse(self, prob=0.2):
        self.create_input_output()
        for i in self.inputs:
            for j in self.outputs:
                if np.random.random() < prob:
                    self.add_edge(i.id, j.id, rand(), True)

    def make_empty(self):
        self.create_input_output()

    def make_full(self):
        self.create_input_output()
        for i in self.inputs:
            for j in self.outputs:
                self.add_edge(i.id, j.id, rand(), True)

    def mutate_toggle(self):
        child = copy.deepcopy(self)
        np.random.choice(child.edges.values()).toggle()
        return child

    def mutate_add_node(self):
        child = copy.deepcopy(self)
        edge = np.random.choice(child.edges.values())
        edge.active = False
        id = child.add_node(False, False, 0)
        child.add_edge(edge.source, id, 1.0, True)
        child.add_edge(id, edge.dest, edge.weight, True)
        return child

    def mutate_add_edge(self):
        child = copy.deepcopy(self)
        src = np.random.choice(xrange(child.n_nodes))
        outgoing = {x[1] for x in filter(lambda x: x[0] == src, child.edges.keys())}
        candidates = list(set([x.id for x in child.outputs + child.hidden]) - outgoing)
        if candidates:
            child.add_edge(src, np.random.choice(candidates), rand(), True)
        return child

    def mutate_crossover(self, c2):
        child = Chromosome(self.n_inputs, self.n_outputs)
        fitter, other = (self, c2) if self.fitness > c2.fitness else (c2, self)
        for id in fitter.edges.keys():
            e1 = fitter.edges[id]
            try:
                e2 = other.edges[id]
            except KeyError:
                child.add_edge(e1.source, e1.dest, e1.weight, e1.active, e1.hm)
            else:
                if e1.hm == e2.hm:
                    c = np.random.choice([e1, e2])
                    child.add_edge(c.source, c.dest, c.weight, c.active, c.hm)
                else:
                    child.add_edge(e1.source, e1.dest, e1.weight, e1.active, e1.hm)

        for id in xrange(fitter.n_nodes):
            if id < other.n_nodes:
                c = np.random.choice([fitter.nodes[id], other.nodes[id]])
                child.add_node(c.is_input, c.is_output, c.bias)
            else:
                n = fitter.nodes[id]
                child.add_node(n.is_input, n.is_output, n.bias)
        return child

    def mutate_weights(self):
        child = copy.deepcopy(self)
        for id in xrange(child.n_inputs, child.n_nodes):
            child.nodes[id].mutate_bias()
        for edge in child.edges.values():
            edge.mutate_weight()
        return child

    def mutate_weights_severe(self):
        child = copy.deepcopy(self)
        e = child.edges.keys()
        N = child.n_nodes - child.n_inputs + child.n_edges
        end_part = N * 0.8
        num = 0.0
        severe = np.random.random() <= 0.5
        for i in xrange(N):
            if severe:
                gauss_point = 0.3
                cold_gauss_point = 0.1
            elif (N >= 10.0) and (num > end_part):
                gauss_point = 0.5
                cold_gauss_point = 0.3
            else:
                gauss_point = 1.0 - MUTATION_RATE
                if np.random.random() <= 0.5:
                    cold_gauss_point = 0.9 - MUTATION_RATE
                else:
                    cold_gauss_point = 1.0 - MUTATION_RATE

            rnd_num = rand() * MUTATION_POWER
            rnd_choice = np.random.random()
            if gauss_point < rnd_choice:
                if i < child.n_edges:
                    child.edges[e[i]].weight += rnd_num
                else:
                    child.nodes[i - child.n_edges + child.n_inputs].bias += rnd_num
            elif cold_gauss_point < rnd_choice:
                if i < child.n_edges:
                    child.edges[e[i]].weight = rnd_num
                else:
                    child.nodes[i - child.n_edges + child.n_inputs].bias = rnd_num
            num += 1.0
        return child

    def phenotype(self):
        net = Network.Network()
        nodes = [net.add_node(n.is_input, n.is_output, n.bias) for n in self.nodes]
        for edge in self.edges.values():
            if edge.active:
                net.add_edge(edge.weight, nodes[edge.source], nodes[edge.dest])
        return net

    def set_fitness(self, fitness):
        self.fitness = fitness

    def set_species(self, spid):
        self.species = spid

    def distance(self, c2):
        a, b = (self, c2) if len(self.edges) > len(c2.edges) else (c2, self)
        w = 0
        m = 0
        d = 0
        e = 0
        max_hm2 = max([x.hm for x in b.edges.values()])
        for edge in a.edges.values():
            try:
                edge2 = b.edges[(edge.source, edge.dest)]
            except KeyError:
                if edge.hm > max_hm2:
                    e += 1
                else:
                    d += 1
            else:
                w += np.abs(edge.weight - edge2.weight)
                m += 1
        d += len(b.edges) - m
        dist = (EXCESS_W * e + DISJOINT_W * d) / len(a.edges)
        if m > 0:
            dist += WEIGHT_W * (w / m)
        return dist

    def __lt__(self, other):
        return self.fitness < other.fitness

    def __gt__(self, other):
        return self.fitness > other.fitness


"""x=Chromosome(2, 2)
x.make_full()
#x.mutate_add_edge()
#x.mutate_add_node()
#x.mutate_weights()
y=Chromosome(2,2)
y.make_full()
a=y.mutate_add_node()
#y.mutate_add_node()
#y.mutate_add_edge()
#y.mutate_toggle()
z=y.mutate_crossover(a)
u=z.mutate_crossover(a)
g=z.phenotype()
print(y.distance(x))
g.set_input([0.5, 1.0])
g.eval_asynch()
print(g.get_current_output())"""
