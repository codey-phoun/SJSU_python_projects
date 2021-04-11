"""
Codey Phoun
CS123B/CS223 Programming Assignment
10/07/2020
Problem Description: 
    Find the shortest route of all cities where each city is only
    visited once
    
Objective: 
    Write a genetic algorithm to create a solution to the
    Traveling Salesperson Problem

Genetic Algorithm Implementation:
    Method of Representation:
        Each individual/chromosome is represented by a list of unique cities
    Method of Selection:
        Elitist selection and Roulette Wheel selection are both used to select
        the individuals with the best fitness scores. These two selection methods
        are controlled by the elite_percentile and fitness_percentile parameters.
    Method of Evaluation:
        The fitness of each individual is measured by the formula: 1/distance * 10000
        The shorter a route's distance is, the higher the fitness score will be.
    Method of Reproduction:
        Selected elites are guaranteed to move on to the new generation.
        New offspring are created by selecting parents through the Roulette Wheel
        selection. These offspring undergo both a crossover and a mutation. 

Program ends after # of iterations set by the generations parameter has been reached
or when a local minima is detected. The minima_iterations parameter sets the max
number of consecutive returned identical best routes before ending the program. 
"""

import os
import sys
import pandas as pd
import random
import math
import statistics

# Obtain the distance between 2 cities
def get_distance(city1, city2, city_distances):
    distance = city_distances[city1][city2]
    return distance

# Create the initial population
def initialize_population(population_size, city_distances):
    gen_zero = []
    for n in range(population_size):
        temp = []
        temp = random.sample(city_distances.index.tolist(), len(city_distances))
        gen_zero.append(temp)
    return gen_zero

# Calculate the distances for each individual in the population
def population_distances(population, city_distances):
    distances = []
    total_distance = 0
    for individual in population:
        for city in range(0, len(individual) - 1):
            total_distance += get_distance(individual[city], individual[city + 1], city_distances)
        distances.append(total_distance)
        total_distance = 0
    return distances

# Calculate fitness from distances
def get_fitness(distances):
    fitness_results = []
    for d in distances:
        fitness = 1 / d * 10000
        fitness_results.append(fitness)
    return fitness_results

# Calculate fitness proportion and cumulative sums and add to dataframe
def fitness_proportion(df):
    total_fitness = df["Fitness"].sum()
    df["Fitness_Proportion"] = df["Fitness"] / total_fitness
    df["Cumulative_Proportion"] = df["Fitness_Proportion"].cumsum()

# Select the top percentile of individuals to join new generation
def elite_fitness_selection(df, elite_percentile):
    elite = []
    num_elite = math.floor(len(df) * elite_percentile)
    for i in range(num_elite):
        elite.append(df.iloc[i, 0])
    return elite

# Perform a crossover between two parents
def ordered_crossover(parent1, parent2):
    offspring = []
    city1 = random.randrange(len(parent1))
    city2 = random.randrange(len(parent1))
    cross_start, cross_end = min(city1, city2), max(city1, city2 + 1)
    offspring.extend(parent1[cross_start:cross_end + 1])
    offspring.extend([city for city in parent2 if city not in offspring])
    return offspring

# Mutate all crossovered offspring
def mutate_offspring(offspring):
    city1 = random.randrange(len(offspring))
    city2 = random.randrange(len(offspring))
    offspring[city2], offspring[city1] = offspring[city1], offspring[city2]
    return offspring

# Generate the offspring for the new generation
def create_offspring(fitness_selection, df, population_size, fitness_percentile):
    # fixes index issues from calculating proportions with small population sizes
    if fitness_percentile <= df["Cumulative_Proportion"][0]:
        fitness_percentile = df["Cumulative_Proportion"][0]

    crossover_offspring = []

    # select two parents, weighted by their fitness proportion
    # lower distance = higher fitness and higher fitness proportion, giving higher chance for selection
    # individuals with a chance for selection must have a cumulative proportion below the fitness_percentile parameter
    for i in range(population_size - len(fitness_selection)):
        parent1 = random.choices(df[df["Cumulative_Proportion"] <= fitness_percentile].iloc[:, 0],
                                 weights=df[df["Cumulative_Proportion"] <= fitness_percentile]
                                 ["Fitness_Proportion"].tolist())

        # flatten the list since random.choices returns a list of a list
        parent1 = [city for sublist in parent1 for city in sublist]

        parent2 = random.choices(df[df["Cumulative_Proportion"] <= fitness_percentile].iloc[:, 0],
                                 weights=df[df["Cumulative_Proportion"] <= fitness_percentile]
                                 ["Fitness_Proportion"].tolist())

        parent2 = [city for sublist in parent2 for city in sublist]

        # perform crossover of two parents
        crossover_offspring.append(ordered_crossover(parent1, parent2))

    # mutate all new offspring
    for i in range(len(crossover_offspring)):
        crossover_offspring[i] = mutate_offspring(crossover_offspring[i])

    return crossover_offspring


# Primary genetic algorithm loop
def genetic_algorithm_TS(population, city_distances, population_size, elite_percentile, fitness_percentile):
    # Calculate distances of the population
    distances = population_distances(population, city_distances)

    # Calculate the fitness of the population
    fitness_results = get_fitness(distances)

    # Create a dataframe of the population
    generation_df = pd.DataFrame({
        "Individuals": population,
        "Distances": distances,
        "Fitness": fitness_results})

    # Order the individuals by their fitness
    generation_df = generation_df.sort_values("Fitness", ascending=False, ignore_index=True)

    # Calculate the fitness proportions and cumulative sum for each individual
    fitness_proportion(generation_df)

    # Perform elite selection
    fitness_selection = elite_fitness_selection(generation_df, elite_percentile)

    # Generate the offspring for the new generation by crossover and mutation
    offspring = create_offspring(fitness_selection, generation_df, population_size, fitness_percentile)

    # Combine elite individuals with mutated offspring for the new generation
    new_generation = fitness_selection + offspring

    # Calculate new distances and fitness results for new generation
    new_gen_distances = population_distances(new_generation, city_distances)
    new_gen_fitness_results = get_fitness(new_gen_distances)

    # Create new dataframe of the new generation
    new_generation_df = pd.DataFrame({
        "Individuals": new_generation,
        "Distances": new_gen_distances,
        "Fitness": new_gen_fitness_results})

    new_generation_df = new_generation_df.sort_values("Fitness", ascending=False, ignore_index=True)
    best_route = new_generation_df.iloc[0, 0]
    best_route_string = ' '.join([str(city) for city in best_route])
    best_distance = new_generation_df.iloc[0, 1]

    print("Best route of generation: " + best_route_string)
    print("Best route distance: " + str(best_distance) + "\n")

    return new_generation, best_distance, new_gen_fitness_results

# main control loop
def main(population_size=100, elite_percentile=0.05, fitness_percentile=0.20, generations=25, minima_iterations=5):
    """ population_size: total population size for each generation \n
        elite_percentile: percentile of top performing individuals guaranteed to be in the next generation \n
        fitness_percentile: percentile of top performing individuals with a chance to create offspring \n
        generations: maximum number of generations to create \n
        minima_iterations: max number of iterations with the same result before loop breaks \n
    """
    # set working directory to script's directory
    file_path = os.path.abspath(sys.argv[0])
    directory = os.path.dirname(file_path)
    os.chdir(directory)
    
    # read in file of city distances
    city_distances = pd.read_csv("TS_Distances_Between_Cities.csv", index_col=0).dropna()

    # initialize variables
    population = initialize_population(population_size, city_distances)
    current_best_distance = 0
    minima_counter = 0
    generation_counter = 0

    # open file to write each iteration's fitness results
    iteration_results = open("CodeyPhoun_GA_TS_Info.txt", 'w')

    # run the main genetic algorithm loop
    for i in range(generations):
        population, best_distance, new_gen_fitness = genetic_algorithm_TS(
            population, city_distances, population_size, elite_percentile, fitness_percentile)

        iteration_results.write("Iteration: " + str(generation_counter) + "\n")
        iteration_results.write("Population size: " + str(population_size) + "\n")
        iteration_results.write('Average fitness score : ' + str(statistics.mean(new_gen_fitness)) + '\n')
        iteration_results.write('Median fitness score : ' + str(statistics.median(new_gen_fitness)) + '\n')
        iteration_results.write('STD of fitness scores : ' + str(statistics.stdev(new_gen_fitness)) + '\n')
        iteration_results.write('Size of the selected subset of the population : ' + str(
            int(population_size * fitness_percentile)) + '\n\n')

        generation_counter += 1

        if current_best_distance == best_distance:
            minima_counter += 1
        else:
            current_best_distance = best_distance
            minima_counter = 0

        if minima_counter == minima_iterations:
            print("Local minima detected")
            print("Program has ended after " + str(generation_counter) + " generations")
            print("Minima first reached after " + str(generation_counter - minima_counter) + " generations")
            break

    results = open("CodeyPhoun_GA_TS_Result.txt", 'w')
    final_route = population[0]

    for city in final_route:
        results.write(str(city_distances.index.tolist().index(city)) + " " + str(city) + "\n")

    results.close()
    iteration_results.close()


main(population_size=100, elite_percentile=0.1, fitness_percentile=0.30, generations=50, minima_iterations=10)
