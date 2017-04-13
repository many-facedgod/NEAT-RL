from Genetics import *
import copy


class Species:
    def __init__(self, sid):
        self.speciesID = sid
        self.chromosomes = []
        self.repID = -1
        self.bestChromosomeID = -1
        self.bestFitness = 0
        self.offsprings = []

    def add(self, chromosome):
        if chromosome.fitness > self.bestFitness:
            self.bestFitness = chromosome.fitness
            self.bestChromosomeID = self.getSize()
        self.chromosomes.append(chromosome)
        chromosome.set_species(self.speciesID)
        self.repID = np.random.randint(0, self.getSize())

    def addOffspring(self, chromosome):
        self.offsprings.append(chromosome)

    def get(self, index):
        return self.chromosomes[index]

    def getSpeciesID(self):
        return self.speciesID

    def getSize(self):
        return len(self.chromosomes)

    def setRepresentativeID(self, index):
        self.repID = index

    def getRepresentative(self):
        return self.chromosomes[self.repID]

    def getAvgSpeciesFitness(self):
        return np.sum([chromosome.fitness for chromosome in self.chromosomes]) / self.getSize()

    def truncatedSelect(self):
        self.chromosomes.sort(reverse=True)
        self.chromosomes = self.chromosomes[0:int(np.ceil(REPRODUCTION_FRACTION * self.getSize()))]
        self.setRepresentativeID(np.random.randint(0, self.getSize()))

    def rouletteWheelSelect(self):
        fitness = [chromosome.fitness for chromosome in self.chromosomes]
        total_fitness = np.sum(fitness)
        normalized_fitness = [x / total_fitness for x in fitness]
        self.chromosomes = np.random.choice(self.chromosomes, int(np.ceil(REPRODUCTION_FRACTION * self.getSize())),
                                            p=normalized_fitness)
        self.setRepresentativeID(np.random.randint(0, self.getSize()))

    def getRandomChromosomeID(self):
        chromosomeID = np.random.randint(0, self.getSize())
        return chromosomeID

    def getOffsprings(self):
        return self.offsprings

    def reproduce(self, n_offsprings):
        for i in n_offsprings:
            x = np.random.random()
            if x <= PROB_MUTATE:
                parent = self.chromosomes[self.getRandomChromosomeID()]
                mutation = np.random.choice(3, p=[PROB_ADD_EDGE, PROB_ADD_NODE, 1 - (PROB_ADD_EDGE + PROB_ADD_NODE)])
                if mutation == 0:
                    child = parent.mutate_add_edge()
                    self.addOffspring(child)
                elif mutation == 1:
                    child = parent.mutate_add_node()
                    self.addOffspring(child)
                else:
                    toggle = np.random.choice(2, p=[PROB_TOGGLE, 1 - PROB_TOGGLE])
                    child = parent.mutate_weights()
                    if toggle == 0:
                        child_t = child.mutate_toggle()
                        self.addOffspring(child_t)
                    else:
                        self.addOffspring(child)
            else:
                parent1 = self.chromosomes[self.getRandomChromosomeID()]
                parent2 = self.chromosomes[self.getRandomChromosomeID()]
                child = parent1.mutate_crossover(parent2)
                self.addOffspring(child)
        self.addOffspring(copy.deepcopy(self.chromosomes[self.bestChromosomeID]))