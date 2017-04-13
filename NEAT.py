from Population import *
from Species import *

def fitness(network):
    network.clear_state()
    err = 0
    network.set_input([0, 0])
    network.eval_asynch()
    err += np.abs(network.get_current_output()[0])
    network.clear_state()
    network.set_input([0, 1])
    network.eval_asynch()
    err += np.abs(network.get_current_output()[0] - 1)
    network.clear_state()
    network.set_input([1, 0])
    network.eval_asynch()
    err += np.abs(network.get_current_output()[0] - 1)
    network.clear_state()
    network.set_input([1, 1])
    network.eval_asynch()
    err += np.abs(network.get_current_output()[0])
    return -err

def NEAT(fitness_fn, n_inputs, n_outputs, n_generations):
	population = Population(n_inputs, n_outputs)
	population.create()
	for generation in range(n_generations):
		population.evaluate(fitness_fn)
		print (generation, population.getBestChromosome().fitness)
		population.speciate()
		population.select()
		population.reproduce()
		GeneE.clear_history()
		population.incrementGeneration()


NEAT(fitness, 2, 1, 20)
