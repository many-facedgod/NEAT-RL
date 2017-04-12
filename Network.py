from __future__ import print_function
import networkx as nx
import numpy as np

SIGMOID_POWER = 1


def sigmoid(x, z):
    return 1.0 / (1 + np.exp(-z * x))


class Neuron:
    count = 0

    def __init__(self, bias, is_input, is_output):
        self.id = Neuron.count
        Neuron.count += 1
        self.bias = bias
        self.current = 0
        self.is_input = is_input
        self.is_output = is_output


class Edge:
    count = 0

    def __init__(self, weight, rec):
        self.id = Edge.count
        Edge.count += 1
        self.weight = weight
        self.isrec = rec


class Network:
    count = 0

    def __init__(self):
        self.graph = nx.DiGraph()
        self.DAG = nx.DiGraph()
        self.id = Network.count
        Network.count += 1
        self.isrec = False
        self.inputs = []
        self.outputs = []
        self.nodes = []
        self.topocache=[]

    def node_count(self):
        return len(self.nodes)

    def edge_count(self):
        return len(self.graph.edges())

    def add_edge(self, weight, source, dest):
        assert (source, dest) not in self.graph.edges(), "Edge already present"
        rec = nx.has_path(self.DAG, dest, source)
        e = Edge(weight, rec)
        self.graph.add_edge(source, dest, object=e)
        if not rec:
            self.DAG.add_edge(source, dest, object=e)
        else:
            self.isrec = True
        self.topocache=[]

    def add_node(self, is_input, is_output, bias=0.0):
        n = Neuron(bias, is_input, is_output)
        self.graph.add_node(n)
        self.DAG.add_node(n)
        if is_input:
            self.inputs.append(n)
        if is_output:
            self.outputs.append(n)
        self.nodes.append(n)
        self.topocache=[]
        return n

    def eval_node(self, node):
        assert not node.is_input, "Input nodes cannot be evaluated"
        l = self.graph.in_edges(nbunch=[node], data=True)
        s = 0
        for source, _, e in l:
            s += source.current * e["object"].weight
        s += node.bias
        return sigmoid(s, SIGMOID_POWER)

    def set_input(self, inputs):
        for n, v in zip(self.inputs, inputs):
            n.current = v

    def eval_synch(self, k=5):
        l = self.nodes
        for j in xrange(k):
            new_vals = []
            for n in l:
                if not n.is_input:
                    new_vals.append(self.eval_node(n))
            i = 0
            for n in l:
                if not n.is_input:
                    n.current = new_vals[i]
                    i += 1

    def eval_asynch(self):
        if not self.topocache:
            self.topocache = nx.topological_sort(self.DAG)
        for n in self.topocache:
            if not n.is_input:
                n.current = self.eval_node(n)

    def get_current_output(self):
        return [n.current for n in self.outputs]

    def clear_state(self):
        for n in self.nodes:
            n.current=0.0



net=Network()
n1=net.add_node(True, False)
n2=net.add_node(True, False)
n3=net.add_node(False, False, 0.0)
n4=net.add_node(False, False, 0.0)
n5=net.add_node(False, True, 0.0)
net.add_edge(1, n1, n3)
net.add_edge(1, n1, n4)
net.add_edge(1, n2, n3)
net.add_edge(1, n2, n4)
net.add_edge(1, n3, n5)
net.add_edge(-1, n5, n3)
net.add_edge(1, n4, n5)
print(net.isrec)
net.set_input([0.1, 0.2])
net.eval_asynch()
print(net.get_current_output())
net.set_input([0.1, 0.2])
net.eval_asynch()
print(net.get_current_output())
net.clear_state()
net.set_input([0.0, -0.5])
net.eval_asynch()
print(net.get_current_output())
net.set_input([0.1, 0.2])
net.eval_asynch()
print(net.get_current_output())
net.set_input([0.4, -1])
net.eval_asynch()
print(net.get_current_output())