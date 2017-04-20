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
        self.bestChromosomeID=0

    def rouletteWheelSelect(self):
        fitness = [chromosome.fitness for chromosome in self.chromosomes]
        total_fitness = np.sum(fitness)
        normalized_fitness = [x / total_fitness for x in fitness]
        best=copy.deepcopy(max(self.chromosomes))
        self.chromosomes = np.random.choice(self.chromosomes, int(np.ceil(REPRODUCTION_FRACTION * self.getSize())),
                                            p=normalized_fitness).tolist()
        self.setRepresentativeID(np.random.randint(0, self.getSize()))
        self.chromosomes.insert(0, best)
        self.bestChromosomeID=0

    def getRandomChromosomeID(self):
        chromosomeID = np.random.randint(0, self.getSize())
        return chromosomeID

    def getOffsprings(self):
        return self.offsprings

    def getBestChromosome(self):
        return max(self.chromosomes)

    def reproduce(self, n_offsprings):
        for i in xrange(n_offsprings):
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
                    if MUTATE_WEIGHT_OPTION == "Default":
                        child = parent.mutate_weights()
                    elif MUTATE_WEIGHT_OPTION == "Severe":
                        child = parent.mutate_weights_severe()
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
        self.addOffspring(copy.deepcopy(max(self.chromosomes)))
        self.chromosomes=self.offsprings
        print("Fitness for species "+str(self.offsprings[-1].fitness))

    def __lt__(self, other):
        if DELTA_PARAMETER == "Best":
            return self.getBestChromosome().fitness < other.getBestChromosome().fitness
        elif DELTA_PARAMETER == "Avg":
            return self.getAvgSpeciesFitness() < other.getAvgSpeciesFitness()

    def __gt__(self, other):
        if DELTA_PARAMETER == "Best":
            return self.getBestChromosome().fitness > other.getBestChromosome().fitness
        elif DELTA_PARAMETER == "Avg":
            return self.getAvgSpeciesFitness() > other.getAvgSpeciesFitness()
