import Network
import numpy as np


def rand():
    return 1 - 2 * np.random.random()


class GeneN:
    def __init__(self, id, is_input, is_output, bias):
        self.id = id
        self.is_input = is_input
        self.is_output = is_output
        self.bias = bias

    def mutate_bias(self, rate, min=-5, max=5):
        self.bias += np.random.normal() * rate
        self.bias = np.clip(self.bias, min, max)


class GeneE:
    def __init__(self, source, dest, weight, active, HM):
        self.source = source
        self.dest = dest
        self.weight = weight
        self.active = active
        self.HM = HM
        self.id = (source, dest)

    def toggle(self):
        self.active = not self.active

    def mutate_weight(self, rate, min=-5, max=5):
        self.weight += np.random.normal() * rate
        self.weight = np.clip(self.weight, min, max)


class Chromosome:
    def __init__(self, n_inputs, n_outputs):
        self.species = 0
        self.nodes = []
        self.edges = {}
        self.n_inputs = n_inputs
        self.inputs = []
        self.outputs = []
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
        return id

    def add_edge(self, source, dest, weight, active, HM):
        id = (source, dest)
        self.n_edges += 1
        self.edges[id] = GeneE(source, dest, weight, active, HM)
        return id

    def mutate_toggle(self, hm):
        np.random.choice(self.edges).toggle()
        return hm

    def mutate_add_node(self, hm):
        edge = np.random.choice(self.edges.values())
        edge.active = False
        id = self.add_node(False, False, 0)
        self.add_edge(edge.source, id, 1.0, True, hm)
        hm += 1
        self.add_edge(id, edge.dest, edge.weight, True, hm)
        hm += 1
        return hm

    def mutate_add_edge(self, hm):
        src = np.random.choice(xrange(self.n_nodes))
        outgoing = {x[0] for x in filter(lambda x: x[0] == src, self.edges.keys())}
        candidates = list(set(xrange(self.n_inputs, self.n_nodes)) - outgoing)
        if not candidates:
            return hm
        else:
            self.add_edge(src, np.random.choice(candidates), rand(), True, hm)
            return hm + 1

    def mutate_crossover(self, hm, c2):
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

        return hm

    def mutate_weights(self, hm, rate, min=-5, max=5):
        for id in xrange(self.n_inputs, self.n_nodes):
            self.nodes[id].mutate_bias(rate, min, max)
        for edge in self.edges.values():
            edge.mutate_weight(rate, min, max)
        return hm

    def phenotype(self):
        net=Network.Network()
        nodes=[net.add_node(n.is_input, n.is_input, n.bias) for n in self.nodes]
        for edge in self.edges.values():
            net.add_edge(edge.weight, nodes[edge.source], nodes[edge.dest])
        return net

    def set_fitness(self, fitness):
        self.fitness=fitness




