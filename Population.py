import numpy as np
from Config import *
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

	def create(self, mode=POPULATION_INIT_MODE):
		for i in range(self.size):
			chromosome = Chromosome(self.n_inputs, self.n_outputs)
			if mode=="Full":
				chromosome.make_full()
			elif mode=="Sparse":
				chromosome.make_sparse()
			elif mode=="Empty":
				chromosome.make_empty()
			self.chromosomes.append(chromosome)
		self.generation = 1

	def evaluate(self, function):
		for chromosome in self.chromosomes:
			chromosome.setFitness(function(chromosome.phenotype()))

	def select(self, pop, mode=SELECTION_MODE):
		self.chromosomes = []
		for i in range(len(self.species_list)):
			if mode=="Truncated":
				species.truncateSelect(REPRODUCTION_FRACTION)
			elif mode=="RouletteWheel":
				species.rouletteWheelSelect(REPRODUCTION_FRACTION)
			for chromosome in self.species_list[i].chromosomes:
				self.chromosomes.append(chromosome)

	def speciate(self):
		for chromosome in self.chromosomes:
			if len(self.species_list) == 0 
				species = Species(len(self.species_list))
				species.add(chromosome)
				self.species_list.append(species)
			else:
				flag = 0
				for species in species_list:
					representative = species.getRepresentative()
					if chromosome.distance(representative)<DISTANCE_THRESHOLD:
						species.add(chromosome)
						flag = 1
						break
				if flag==0:
					species = Species(len(self.species_list))
					species.add(chromosome)
					self.species_list.append(species)

	def reproduce(self):
		avg_species_fitness_list = [species.getAvgSpeciesFitness() for species in self.species_list]
		total_avg_species_fitness = np.sum(avg_species_fitness_list)
		self.chromosomes = []
		for species,avg_species_fitness in zip(self.species_list, avg_species_fitness_list):
			n_offsprings = self.size*avg_species_fitness/total_avg_species_fitness
			species.reproduce(n_offsprings)
			self.chromosomes = self.chromosomes + species.getOffsprings()
		self.species = []

	def incrementGeneration(self):
		self.generation = self.generation + 1
