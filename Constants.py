from enum import Enum,IntEnum


class GameState(Enum):
    RUNNING = 1
    STABLE = 2
   
class Sizes(IntEnum):
    MAX_GENERATION=1000 # maximum generation i allow the game to get stable
    INIT_SIZE = 10 # sqrt of the size of initialization matrice
    POPULATION_SIZE = 10 
    HISTORY_SIZE=10 # size of history of generations to look back in game of life to trace stability
    CELL_SIZE=10 # sqrt of the size of cell in game of life
    MUTATION_PROBABILITY=0.03
    ROW_SIZE=108
    COL_SIZE=192
    EVOLUTION_SPEED=1e-10 # speed for game of life simulation
    WINDOW_HEIGHT=ROW_SIZE * CELL_SIZE 
    WINDOW_WIDTH=COL_SIZE * CELL_SIZE

class Colors(Enum):
    GREEN=(0,255,0)
    BLACK=(0,0,0)
