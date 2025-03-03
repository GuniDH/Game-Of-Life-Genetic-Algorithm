import torch
import matplotlib.pyplot as plt
from Constants import Sizes
from GameOfLife import GameOfLife


class GeneticAlgorithm:
    def __init__(self, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.population = self.create_initial_population() # current population

    def create_initial_population(self):
        """Create the initial population with random configurations."""
        return [
            GameOfLife(
                torch.bernoulli(torch.ones(Sizes.INIT_SIZE, Sizes.INIT_SIZE) * 0.5).bool().to(self.device)
            )
            for _ in range(Sizes.POPULATION_SIZE)
        ]

    def pair(self, parent1, parent2, stagnated):
        """Create offspring from two parents using a two part crossover and dynamic mutation."""
        split_point = Sizes.INIT_SIZE // 2
        child_init = torch.cat((parent1.init[:split_point], parent2.init[split_point:]), dim=0)
        # This crossoverapproach was too random and didn't perform well:
        '''
        top_half = torch.where(
            torch.bernoulli(torch.ones_like(parent1.init[:split_point]) * 0.5).bool(),
            parent1.init[:split_point],
            parent2.init[:split_point]
        )
        bottom_half = torch.where(
            torch.bernoulli(torch.ones_like(parent1.init[split_point:]) * 0.5).bool(),
            parent2.init[split_point:],
            parent1.init[split_point:]
        )
        child_init = torch.cat((top_half,bottom_half), dim=0)
        '''
        # Mutation rate gets doubled when stagnation is detected (maximum fitness doesn't improve for a certain amount of generations)
        mutation_rate = Sizes.MUTATION_PROBABILITY * (2 if stagnated else 1.0)
        mutation_mask = torch.bernoulli(torch.ones_like(child_init, dtype=torch.float) * mutation_rate).bool()
        child_init = torch.where(mutation_mask, ~child_init, child_init)
        return GameOfLife(child_init)

    def evolve_population(self, stagnated):
        """Evolve the population"""
        fitness_scores = torch.tensor([chromosome.fitness for chromosome in self.population], device=self.device, dtype=torch.float32)
        # this probabilities vector is for using roulette selection
        probabilities = (fitness_scores / fitness_scores.sum()).float()

        # Select the elite chromosome and keep the elite chromosome in the next generation
        elite = max(self.population, key=lambda c: c.fitness)
        next_gen = [elite]  

        # Generate the rest of the population
        for _ in range(Sizes.POPULATION_SIZE - 2):  # -2 because the elite is already included and leave place for random chromosome
            parents = [
                self.population[idx] for idx in torch.multinomial(probabilities, 2, replacement=True).tolist()
                # i allow the same parent to be chosen twice
            ]
            next_gen.append(self.pair(parents[0], parents[1], stagnated))

        random_chromosome = GameOfLife(
            torch.bernoulli(torch.ones(Sizes.INIT_SIZE, Sizes.INIT_SIZE) * 0.5).bool().to(self.device)
        )
        next_gen.append(random_chromosome)

        return next_gen

    def evolve(self, generations=100, stagnation_threshold=5):
        """Run the genetic algorithm for a number of generations."""
        max_fitnesses, avg_fitnesses = [], []
        stagnation_counter, previous_max = 0, 0

        for generation in range(generations):

            for chromosome in self.population:
                chromosome.run() # run each chromosome (game of life configuration)

            fitness_scores = [chromosome.fitness for chromosome in self.population]
            max_fitness = max(fitness_scores)
            avg_fitness = sum(fitness_scores) / len(fitness_scores)

            max_fitnesses.append(max_fitness)
            avg_fitnesses.append(avg_fitness)

            # detect stagnation by the maximum fitness not improving for a certain amount of generations (stagnation_threshold)
            if max_fitness <= previous_max:
                stagnation_counter += 1
            else:
                stagnation_counter = 0

            previous_max = max_fitness

            stagnated = stagnation_counter >= stagnation_threshold
            self.population = self.evolve_population(stagnated) # if stagnated, mutation rate is doubled

            print(f"Generation {generation+1}: Max Fitness = {max_fitness}, Avg Fitness = {avg_fitness}")

        self.plot_progress(max_fitnesses, avg_fitnesses)
        return max(self.population, key=lambda c: c.fitness) # return winning chromosome

    @staticmethod
    def plot_progress(max_fitness, avg_fitness):
        """Plot fitness progress over generations."""
        plt.plot(max_fitness, label="Max Fitness", color='blue')
        plt.plot(avg_fitness, label="Avg Fitness", color='orange')
        plt.xlabel("Generations")
        plt.ylabel("Fitness")
        plt.title("Fitness Progress Over Generations")
        plt.legend()
        plt.grid()
        plt.show()

    def simulate_winning_chromosome(self, winning_chromosome):
        game = GameOfLife(winning_chromosome.init)
        print("Running the winning chromosome")
        game.simulate()
        game.display_stats()
