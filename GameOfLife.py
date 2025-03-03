import pygame
import torch
import math
from Constants import Sizes, GameState, Colors


class GameOfLife:
    def __init__(self,init):
        """
        Initialize the Game of Life with a grid configuration.
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
        self.state = GameState.RUNNING
        self.cells = torch.zeros((Sizes.ROW_SIZE, Sizes.COL_SIZE), dtype=torch.bool, device=self.device)
        self.init = init

        # Apply the initialization (configuration) into the game board
        start_row = (self.cells.shape[0] - init.shape[0]) // 2
        start_col = (self.cells.shape[1] - init.shape[1]) // 2
        self.cells[start_row:start_row + init.shape[0], start_col:start_col + init.shape[1]] = init

        self.fitness=0
        self.generation = 0 # current generation number
        self.alive_count = torch.sum(self.cells).item() # current amount of alive cells
        self.first_alive=self.alive_count # amount of alive cells in initialization
        self.max_alive = self.alive_count # maximum amount of alives cells tracked in some generation
        self.max_alive_gen=1 # the generation in which the max amount of alive cells was tracked
        self.final_alive=0 # amount of alive cells in stability
        self.last_update = 0
        self.last_hashes = set() # keeping track of the last 10 generations in order to track stability
                                 #(if the current generation is equal to one of them, the configuration is stable (static or cyclic))


    # hashing the configuration matrice for efficiency (instead of saving matrices)
    def hash_generation(self):
        """
        Generate a hash representing the current grid state.
        """
        return hash(self.cells.cpu().numpy().tobytes())


    def check_stability(self):
        """
        Check if the grid has reached a stable state by comparing hashes.
        (if the current generation is equal to one of them, the configuration is stable (static or cyclic))
        """
        # if the game passes the threshold and doesnt get stable until it, its fitness is 0
        if self.generation == Sizes.MAX_GENERATION:
            self.state = GameState.STABLE
            self.fitness = 0
            return

        current_hash = self.hash_generation()
        if current_hash in self.last_hashes:
            self.state = GameState.STABLE
            self.final_alive=torch.sum(self.cells).item()
            # As requested in instructions, taking account of the original config size, final size and its lifespan (i also added max alive)
            self.fitness = ((self.max_alive + self.final_alive) / (self.first_alive*2+1)) + math.log2(self.generation+1)
        else:
            self.last_hashes.add(current_hash)
            if len(self.last_hashes) > Sizes.HISTORY_SIZE:
                self.last_hashes.pop()


    def update(self, simulated=False):
        """
        Update the grid by applying Game of Life rules.
        """

        # This is only for the simulation of the winning chromosome, therefore the simulated field tells us
        # if we are currently just running the genetic algorithm, or simulating the winning best chromosome
        if simulated and pygame.time.get_ticks() - self.last_update < Sizes.EVOLUTION_SPEED:
            return
        if simulated:
            self.last_update = pygame.time.get_ticks()

        # count neighbours using convolution
        kernel = torch.tensor([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
        padded_cells = self.cells.unsqueeze(0).unsqueeze(0).float()
        neighbors = torch.nn.functional.conv2d(padded_cells, kernel, padding=1).squeeze()

        # Apply game of life rules
        self.cells = (neighbors == 3) | ((neighbors == 2) & self.cells)

        self.generation += 1
        self.alive_count = torch.sum(self.cells).item()

        if self.alive_count > self.max_alive:
            self.max_alive = self.alive_count
            self.max_alive_gen = self.generation

        self.check_stability()


    def run(self):
        """
        Run game of life for some chromosome during genetic algorithm running
        """
        while self.state == GameState.RUNNING:
            self.update()


    def simulate(self):
        """
        Simulate game of life for winning best chromosome
        """
        pygame.init()
        screen = pygame.display.set_mode((Sizes.WINDOW_WIDTH, Sizes.WINDOW_HEIGHT), pygame.NOFRAME)
        self.draw(screen)
        # press space to start simualtion
        while not self.space_clicked():
            pass
        while self.state == GameState.RUNNING:
            self.update(simulated=True)
            self.draw(screen)
        # press space to end simualtion
        while not self.space_clicked():
            pass
        pygame.quit()


    def draw(self, screen):
        """
        Render the grid on the screen.
        """
        screen.fill(Colors.BLACK.value)
        rows, cols = torch.where(self.cells)
        for row, col in zip(rows.tolist(), cols.tolist()):
            pygame.draw.rect(screen, Colors.GREEN.value, pygame.Rect(col * Sizes.CELL_SIZE, row * Sizes.CELL_SIZE, Sizes.CELL_SIZE - 1, Sizes.CELL_SIZE - 1))
        pygame.display.flip()


    def space_clicked(self):
        """
        Check for space key press.
        """
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                return True
        return False


    def display_stats(self):
        """
        Print the final stats of the simulation (winning chromosome stats).
        """
        print(f"Fitness: {self.fitness}")
        print(f"Max Alive Cells: {self.max_alive} at Generation {self.max_alive_gen}")
        print(f"Stabled at Generation {self.generation}")
