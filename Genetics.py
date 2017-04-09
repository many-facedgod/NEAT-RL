import Network
import numpy as np

class GeneN:
    def __init__(self, id, is_input, is_output):
        self.id=id
        self.is_input=is_input
        self.is_output=is_output


class GeneE:
    def __init__(self, source, dest, weight, active, HM):
        self.source=source
        self.dest=dest
        self.weight=weight
        self.active=active
        self.HM=HM


class Chromosome:
    def __init__(self, n_inputs, n_outputs, prob=0.5):
        self.species=0
        self.nodes=[]
        self.edges=[]
        self.n_inputs=n_inputs
        self.n_nodes=0
        self.n_outputs=n_outputs
        for i in xrange(n_inputs):
            self.nodes.append(GeneN(self.n_nodes, True, False))
            self.n_nodes+=1
        for i in xrange(n_outputs):
            self.nodes.append(GeneN(self.n_nodes, False, True))
            self.n_nodes+=1
        for i in xrange

