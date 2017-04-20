from Species import *
from Genetics import *


class Population:
    def __init__(self, n_inputs, n_outputs):
        self.size = INIT_SIZE
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.chromosomes = []
        self.species_list = []
        self.generation = 0
        self.bestperformance = 0
        self.n_deltagen = 0
        self.distance_threshold = DISTANCE_THRESHOLD_DEFAULT

    def create(self, mode=POPULATION_INIT_MODE):
        for i in range(self.size):
            chromosome = Chromosome(self.n_inputs, self.n_outputs)
            if mode == "Full":
                chromosome.make_full()
            elif mode == "Sparse":
                chromosome.make_sparse()
            elif mode == "Empty":
                chromosome.make_empty()
            self.chromosomes.append(chromosome)
        self.generation = 1

    def evaluate(self, function):
        for chromosome in self.chromosomes:
            chromosome.set_fitness(function(chromosome.phenotype()))

    def select(self, mode=SELECTION_MODE):
        self.chromosomes = []
        for species in self.species_list:
            if mode == "Truncated":
                species.truncatedSelect()
            elif mode == "RouletteWheel":
                species.rouletteWheelSelect()
            for chromosome in species.chromosomes:
                self.chromosomes.append(chromosome)

    def speciate(self):
        for chromosome in self.chromosomes:
            if len(self.species_list) == 0:
                species = Species(len(self.species_list))
                species.add(chromosome)
                self.species_list.append(species)

            else:
                flag = 0
                for species in self.species_list:
                    representative = species.getRepresentative()
                    if chromosome.distance(representative) < DISTANCE_THRESHOLD:
                        species.add(chromosome)
                        flag = 1
                        break
                if flag == 0:
                    species = Species(len(self.species_list))
                    species.add(chromosome)
                    self.species_list.append(species)
            self.distance_threshold = self.getDistanceThreshold()

    def reproduce(self):
        avg_species_fitness_list = [species.getAvgSpeciesFitness() for species in self.species_list]
        total_avg_species_fitness = np.sum(avg_species_fitness_list)
        delta_parameter = 0
        if DELTA_PARAMETER == "Best":
            delta_parameter = self.getBestChromosome().fitness
        self.chromosomes = []
        for species, avg_species_fitness in zip(self.species_list, avg_species_fitness_list):
            n_offsprings = int(self.size * avg_species_fitness / total_avg_species_fitness)
            species.reproduce(n_offsprings)
            self.chromosomes = self.chromosomes + species.getOffsprings()
            if DELTA_PARAMETER == "Best" and delta_parameter < species.getBestChromosome().fitness:
                delta_parameter = species.getBestChromosome().fitness
        if DELTA_PARAMETER == "Avg":
            delta_parameter = self.getAverageFitness()
        if delta_parameter > self.bestperformance:
            self.bestperformance = delta_parameter
            self.n_deltagen = 0
        else:
            self.n_deltagen += 1
        if self.n_deltagen == DELTA_GENERATIONS and len(self.species_list)>1:
            self.chromosomes = self.species_list.sort(reverse=True)[:2]
        self.species_list = []

    def incrementGeneration(self):
        self.generation += 1
        
    def getDistanceThreshold(self):
		if len(self.species_list) < DISTANCE_THRESHOLD_MIN_LIMIT:
			return (1-DISTANCE_THRESHOLD_CHANGE)*self.distance_threshold
		elif len(self.species_list) >= DISTANCE_THRESHOLD_MIN_LIMIT and len(self.chrosomes) <= DISTANCE_THRESHOLD_MAX_LIMIT:
			return self.distance_threshold
		else:
			return self.distance_threshold*(1+DISTANCE_THRESHOLD_CHANGE)
        
    def getAverageFitness(self):
        return np.mean([c.fitness for c in self.chromosomes])

    def getBestChromosome(self):
        return max([s.getBestChromosome() for s in self.species_list])
