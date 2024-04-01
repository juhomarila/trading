import random
import django
from django.db import connection
from django.db.models import Max

from trading.models import signals, finnish_stock_daily
from trading.simulation import optimize_parameters


def generate_population(param_ranges, population_size):
    population = []
    for _ in range(population_size):
        individual = {}
        for param, (min_val, max_val) in param_ranges.items():  # Access the tuple directly
            individual[param] = [random.uniform(min_val, max_val)]  # Ensure min_val and max_val are accessed directly
        population.append(individual)
    return population


def fitness_function(results, best_victories_threshold, best_victories_profit_threshold):
    # Calculate the distance between current parameters and the threshold values
    distance_victories = abs(results[6] - best_victories_threshold)
    distance_profit = abs(results[2] - best_victories_profit_threshold)

    # Calculate the overall distance
    overall_distance = distance_victories + distance_profit
    print(f"FITNESS ETÄISYYS: {overall_distance}")

    # Inversely weigh the distance to assign a higher fitness score to individuals closer to the thresholds
    fitness_score = 1 / (overall_distance + 1)  # Add 1 to avoid division by zero
    print(f"FITNESS SCORE: {fitness_score}")
    return fitness_score


def select_parents(population, num_parents):
    # Select parents based on fitness (roulette wheel selection, tournament selection, etc.)
    # Here's a simple implementation selecting based on fitness score
    sorted_population = sorted(population, key=lambda x: fitness_function(x), reverse=True)
    return sorted_population[:num_parents]


def crossover(parent1, parent2):
    # Perform crossover between two parents to produce offspring
    # Here's a simple implementation using single-point crossover
    crossover_point = random.randint(1, len(parent1) - 1)
    child = {}
    for param in parent1:
        if random.random() < 0.5:
            child[param] = parent1[param][:crossover_point] + parent2[param][crossover_point:]
        else:
            child[param] = parent2[param][:crossover_point] + parent1[param][crossover_point:]
    return child


def mutate(individual, mutation_rate):
    # Perform mutation on an individual with a given mutation rate
    # Here's a simple implementation flipping a bit with a given probability
    mutated_individual = {}
    for param, values in individual.items():
        mutated_values = []
        for val in values:
            if random.random() < mutation_rate:
                mutated_values.append(random.uniform(param[0], param[1]))
            else:
                mutated_values.append(val)
        mutated_individual[param] = mutated_values
    return mutated_individual


def genetic_algorithm(param_ranges, population_size=100, num_generations=100, mutation_rate=0.1):
    population = generate_population(param_ranges, population_size)

    for _ in range(num_generations):
        # Select parents
        parents = select_parents(population, 2)

        # Create offspring through crossover
        offspring = crossover(parents[0], parents[1])

        # Mutate offspring
        mutated_offspring = mutate(offspring, mutation_rate)

        # Replace least fit individual in population with mutated offspring
        population.sort(key=lambda x: fitness_function(x))
        population[0] = mutated_offspring

    # Return the best individual in the final population
    return max(population, key=lambda x: fitness_function(x))


def generate_random_params(param_ranges, length):
    param_list = []
    for i in range(length):
        param_list.append({param: [random.uniform(min_val, max_val)] for param, (min_val, max_val) in param_ranges.items()})
    return param_list


def optimize_with_learning(buy_param_ranges, sell_param_ranges, max_generations=2, max_iterations=5,
                           best_victories_threshold=0.75,
                           best_victories_profit_threshold=100000.0):
    stock_indicator_data = signals.objects.all().values('symbol', 'adx', 'rsi14', 'aroon_up', 'aroon_down',
                                                        'stock__close', 'stock__date').order_by('stock__date')

    max_date = finnish_stock_daily.objects.exclude(symbol='S&P500').aggregate(max_date=Max('date'))['max_date']
    last_close_values = finnish_stock_daily.objects.filter(date__range=(max_date, max_date)).exclude(
        symbol='S&P500').values('symbol',
                                'close',
                                'date')
    # TODO WIP add here multiprocess return from inner for loop the params with results
    for _ in range(max_generations):
        current_buy_params_list = generate_random_params(buy_param_ranges, max_iterations)
        current_sell_params_list = generate_random_params(sell_param_ranges, max_iterations)
        for i in range(max_iterations):
            results = optimize_parameters(current_buy_params_list[i], current_sell_params_list[i], stock_indicator_data, max_date,
                                          last_close_values)
            _, _, _, _, _, best_victories, _, best_victories_profit = results

            if best_victories >= best_victories_threshold and best_victories_profit >= best_victories_profit_threshold:
                return results
        # buy_param_ranges = adjust_param_ranges(buy_param_ranges, current_buy_params, results, best_victories_threshold,
        #                                        best_victories_profit_threshold)
        # sell_param_ranges = adjust_param_ranges(sell_param_ranges, current_sell_params, results,
        #                                         best_victories_threshold,
        #                                         best_victories_profit_threshold)

        # If max_iterations reached without meeting thresholds, return the best found so far
    return results


# TODO WIP, RUN FIRST 1000/10000 times the algorithm with random numbers.
#  Choose 10/100/100 best results to generate offspring, meaning not a single
#  entry will be same in next the next run. Add some percentage fully random
#  numbers to create genetic diversity, compare results from previous run and if
#  results are worse redo the offspring.
#  Need for some kind of system to evaluate whole round, possibly average of how far
#  results are from desired goal?
#  check your own geneettinen musiikkialgoritmi


def adjust_param_ranges(param_ranges, current_params, results, best_victories_threshold,
                        best_victories_profit_threshold, num_generations=100,
                        population_size=100, mutation_rate=0.1):
    # Define a fitness function based on the performance of the current parameters
    score = fitness_function(results, best_victories_threshold, best_victories_profit_threshold)

    # Generate a population of potential parameter ranges
    population = generate_population(param_ranges, population_size)

    # Run the genetic algorithm to evolve the parameter ranges
    for _ in range(num_generations):
        # Evaluate the fitness of each individual in the population
        fitness_scores = [fitness_function(params) for params in population]

        # Select parents for crossover
        parents = select_parents(population, 2)

        # Create offspring through crossover
        offspring = crossover(parents[0], parents[1])

        # Mutate offspring
        mutated_offspring = mutate(offspring, mutation_rate)

        # Replace least fit individual in population with mutated offspring
        least_fit_index = min(range(len(fitness_scores)), key=fitness_scores.__getitem__)
        population[least_fit_index] = mutated_offspring

    # Return the best-performing individual as the adjusted parameter ranges
    best_individual = max(population, key=fitness_function)
    adjusted_param_ranges = {}
    for param, values in best_individual.items():
        adjusted_param_ranges[param] = values
    return adjusted_param_ranges
