2025-08-27 13:46

Link or File: [Clonalg | Algorithm Afternoon](https://algorithmafternoon.com/immune/clonalg/)
[[Original CSA.pdf]]


Status:

Tags: 

# Original CSA

The Clonal Selection Algorithm (CLONALG) is a biologically inspired computational technique that falls under the umbrella of Artificial Immune Systems (AIS) and Computational Intelligence. It is closely related to other immune-inspired algorithms such as Negative Selection and Immune Network Theory.

> [!quote]
antibodies represent candidate solutions, and the antigen represents the problem to be solved or optimized.


Data Structures:

- Antibody: Represents a candidate solution to the problem
- Population: A collection of antibodies
- Clone: A copy of an antibody that undergoes hypermutation

Parameters:

- Population Size: The number of antibodies in the population
- Clone Size: The maximum number of clones generated for each selected antibody
- Mutation Rate: The rate at which clones are mutated (typically inversely proportional to fitness)
- Replacement Ratio: The proportion of the population replaced by new random antibodies at each iteration
- Max Iterations: The maximum number of iterations before termination

Steps:

1. Initialize a population of antibodies randomly
2. Evaluate the fitness (affinity) of each antibody
3. Repeat until stopping criteria are met:
    1. Select the best antibodies based on their fitness
    2. For each selected antibody:
        1. Generate clones proportional to its fitness
        2. Hypermutate the clones, with mutation rate inversely proportional to fitness
        3. Evaluate the fitness of the clones
        4. Select the best clone to replace the parent antibody
    3. Replace a portion of the worst antibodies with new random antibodies
    4. Evaluate the fitness of the new antibodies
4. Return the best antibody found as the solution





