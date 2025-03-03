# Game of Life Genetic Algorithm by Guni

## Overview

This project implements a **Genetic Algorithm (GA) to optimize initial configurations** in **Conway’s Game of Life**. The goal is to evolve an initial pattern that remains **alive and dynamic for as long as possible** before stabilizing.

The algorithm uses **evolutionary principles** such as:
- **Selection** (Roulette selection based on fitness)
- **Crossover** (Merging parent patterns)
- **Mutation** (Flipping cells to introduce diversity)

The **best evolved configuration** is visualized using **Pygame** after the evolution process.

## Features

- **Genetic Algorithm Optimization**
  - Evolves **Game of Life configurations** using **mutation & crossover**.
  - Uses **PyTorch** for optimized matrix operations.
- **Fitness Evaluation**
  - Tracks **number of alive cells**, **max alive count**, and **stability detection**.
- **Pygame-Based Visualization**
  - Simulates the best evolved configuration.
  - Allows user interaction for **starting/stopping** the simulation.
- **Configurable Evolution Parameters**
  - Supports **variable mutation rates, population sizes, and simulation speeds**.

## Genetic Algorithm Details

### **1. Population Initialization**
- The initial population consists of **random 10x10 binary grids**.
- Each cell has a 50% chance of being alive.

### **2. Fitness Function**
- A configuration is evaluated based on:
  1. **Initial number of alive cells**.
  2. **Max alive count during simulation**.
  3. **If and when stability is reached** (static or cyclic behavior).

### **3. Crossover (Recombination)**
- Two parents are selected using **roulette selection**.
- The grid is **split into two halves**, and each parent contributes one part.
- This ensures a mix of beneficial traits.

### **4. Mutation**
- Each new grid undergoes **random cell flipping** based on a **mutation probability**.
- **Mutation probability doubles** if evolution stagnates (no improvement in max fitness for multiple generations).

### **5. Evolutionary Process**
- The algorithm runs for a **fixed number of generations** (default = 50).
- The best configuration is selected and **simulated visually**.

## Installation

Ensure you have Python installed and install the required dependencies:
```sh
pip install torch matplotlib pygame
```

## Running the Program

1. **Clone the Repository**
   ```sh
   git clone https://github.com/GuniDH/Game-Of-Life-Genetic-Algorithm.git
   cd Game-Of-Life-Genetic-Algorithm

   ```
2. **Run the Genetic Algorithm**
   ```sh
   python Main.py
   ```

## Expected Workflow

1. The **Genetic Algorithm** evolves the population for `X` generations.
2. Once complete, a **fitness graph** appears showing improvement over generations.
3. **Close the fitness graph**, and a **Pygame simulation** launches.
4. Press **Spacebar** to start the simulation.
5. Press **Spacebar again** to exit once stability is reached.

## Configuration Parameters

Defined in `Constants.py`:

| Parameter               | Value |
|-------------------------|-------|
| Population Size        | 10    |
| Mutation Probability   | 3%    |
| Max Generations       | 1000  |
| Grid Size (Rows x Cols) | 108x192 |
| Stability Detection Window | 10 Generations |
| Evolution Speed       | `1e-10` |

## Example Code Usage

```python
from GeneticComputation import GeneticAlgorithm

gc = GeneticAlgorithm()
winning_chromosome = gc.evolve(generations=50)
gc.simulate_winning_chromosome(winning_chromosome)
```


## License

This project is licensed under the **MIT License**.

---
### Author
**Guni**  


