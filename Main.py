from GeneticComputation import GeneticAlgorithm

# Author: Guni Deyo Haness 215615519

def main():
    gc = GeneticAlgorithm() 
    winning_chromosome = gc.evolve(generations=50) # run genetic algorithm and get best 'winning' chromosome
    gc.simulate_winning_chromosome(winning_chromosome) # simulate the best chromosome (configuration)

if __name__ == "__main__":
    main()
