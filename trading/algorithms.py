import itertools
import random
import multiprocessing
from multiprocessing import Manager, Semaphore
import django
import numpy as np
from numpy.random import default_rng
from django.db import connection
from django.db.models import Max

from trading.models import signals, finnish_stock_daily
from trading.simulation import optimize_parameters

MAX_PROCESSES = 8


def generate_population(param_ranges, population_size):
    population = []
    for _ in range(population_size):
        individual = {}
        for param, (min_val, max_val) in param_ranges.items():  # Access the tuple directly
            individual[param] = [random.uniform(min_val, max_val)]  # Ensure min_val and max_val are accessed directly
        population.append(individual)
    return population


def fitness_function(results, victories_threshold, profit_threshold):
    # Calculate the distance between current parameters and the threshold values
    if results[1] != 0:
        distance_victories = abs((results[1] / (results[1] + results[2] + results[3])) / victories_threshold)
    else:
        distance_victories = 0
    if results[0] != 0:
        distance_profit = abs(results[0] / profit_threshold)
    else:
        distance_profit = 0
    # Calculate the fitness score
    fitness_score = distance_victories + distance_profit
    noise_factor = random.uniform(-0.01, 0.01)
    fitness_score += noise_factor
    return fitness_score


def selected_parents_to_pairs(selected, scores):
    parents = []
    for i in range(len(scores)):
        for index, item in enumerate(scores):
            try:
                if index == selected[i]:
                    parents.append(item)

            except(ValueError, IndexError):
                pass
        i += 1

    mothers = parents[:len(parents) // 2]
    fathers = parents[len(parents) // 2:]
    return mothers, fathers


def crossover(mother, father):
    # Convert dictionaries to lists of keys and values
    mother_buy_values = list(mother[0].values())
    mother_buy_keys = list(mother[0].keys())
    mother_sell_values = list(mother[1].values())
    mother_sell_keys = list(mother[1].keys())
    father_buy_values = list(father[0].values())
    father_buy_keys = list(father[0].keys())
    father_sell_values = list(father[1].values())
    father_sell_keys = list(father[1].keys())

    child1, child2 = [], []

    if random.random() < 0.80:
        x = random.randint(1, len(mother_buy_values) - 1)
        y = random.randint(1, len(mother_buy_values) - 1)

        # Create new lists by concatenating sliced parts
        child1_buy_values = mother_buy_values[:x] + father_buy_values[x:]
        child1_buy_keys = mother_buy_keys[:x] + father_buy_keys[x:]
        child1_sell_values = mother_sell_values[:x] + father_sell_values[x:]
        child1_sell_keys = mother_sell_keys[:x] + father_sell_keys[x:]
        child2_buy_values = mother_buy_values[:y] + father_buy_values[y:]
        child2_buy_keys = mother_buy_keys[:y] + father_buy_keys[y:]
        child2_sell_values = mother_sell_values[:y] + father_sell_values[y:]
        child2_sell_keys = mother_sell_keys[:y] + father_sell_keys[y:]

        child1_buy_dict = {k: v for k, v in zip(child1_buy_keys, child1_buy_values)}
        child1_sell_dict = {k: v for k, v in zip(child1_sell_keys, child1_sell_values)}
        child2_buy_dict = {k: v for k, v in zip(child2_buy_keys, child2_buy_values)}
        child2_sell_dict = {k: v for k, v in zip(child2_sell_keys, child2_sell_values)}

        # Construct dictionaries from sliced keys and values
        child1.append(child1_buy_dict)
        child1.append(child1_sell_dict)
        child2.append(child2_buy_dict)
        child2.append(child2_sell_dict)

    else:
        child1 = mother.copy()
        child2 = father.copy()
    children = [child1, child2]
    return children


def mutate(children):
    c1 = children[:len(children) // 2]
    c2 = children[len(children) // 2:]

    for i in range(len(c1)):
        if random.random() < 0.55:
            x = random.randint(0, len(c1[i]) - 1)
            key = random.choice(list(c1[i][x].keys()))
            c1[i][x][key], c2[i][x][key] = c2[i][x][key], c1[i][x][key]
        else:
            c1 = c1
            c2 = c2

    children = c1 + c2
    return children


def add_random_gene(children, buy_param_ranges, sell_param_ranges):
    for i in range(len(children)):
        if random.random() < 0.08:
            x = random.randint(0, len(children[i]) - 1)
            key = random.choice(list(children[i][x].keys()))
            if x == 0:
                random_param_value = generate_random_param(buy_param_ranges[key])
            else:
                random_param_value = generate_random_param(sell_param_ranges[key])
            children[i][x][key] = random_param_value
        else:
            children[i] = children[i]

    return children


def genetic_algorithm(param_ranges, population_size=100, num_generations=100, mutation_rate=0.1):
    population = generate_population(param_ranges, population_size)

    for _ in range(num_generations):
        # Select parents
        parents = selected_parents_to_pairs(population, 2)

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
        param_list.append(
            {param: random.uniform(min_val, max_val) for param, (min_val, max_val) in param_ranges.items()})
    return param_list


def generate_random_param(param_range):
    min_val, max_val = param_range
    return random.uniform(min_val, max_val)


def optimize_with_learning(buy_param_ranges, sell_param_ranges, max_generations=2, max_iterations=6,
                           victories_threshold=0.75,
                           profit_threshold=100000.0):
    stock_indicator_data = signals.objects.all().values('symbol', 'adx', 'rsi14', 'aroon_up', 'aroon_down',
                                                        'stock__close', 'stock__date').order_by('stock__date')

    max_date = finnish_stock_daily.objects.exclude(symbol='S&P500').aggregate(max_date=Max('date'))['max_date']
    last_close_values = finnish_stock_daily.objects.filter(date__range=(max_date, max_date)).exclude(
        symbol='S&P500').values('symbol',
                                'close',
                                'date')

    # TODO WIP add here multiprocess return from inner for loop the params with results
    for i in range(max_generations):
        if i == 0:
            current_buy_params_list = generate_random_params(buy_param_ranges, max_iterations)
            current_sell_params_list = generate_random_params(sell_param_ranges, max_iterations)
        print(f"OSTOPARAMETRIT: {current_buy_params_list}")
        print(f"MYYNTIPARAMETRIT: {current_sell_params_list}")
        all_results = []
        manager = Manager()
        semaphore = Semaphore(MAX_PROCESSES)

        processes = []
        for y in range(max_iterations):
            results = simulate_trading(stock_indicator_data, current_buy_params_list[y], current_sell_params_list[y],
                                       last_close_values)
            profit, victories, losses, even, buy_params, sell_params = results
            all_results.append(results)
            print(
                f"PROFIT: {profit}, OSTO: {format_params(current_buy_params_list[i], True)}, MYYNTI: {format_params(current_sell_params_list[i], False)} VOITTOPROSENTTI: {round((victories / (victories + losses + even)) * 100, 2)}%")
            print(f"VOITTOJA: {victories}, TASAN: {even}, HÄVIÖITÄ: {losses}")
            if victories >= victories_threshold and profit >= profit_threshold:
                return results

        current_buy_params_list, current_sell_params_list = adjust_param_ranges(buy_param_ranges, sell_param_ranges,
                                                                                all_results,
                                                                                victories_threshold,
                                                                                profit_threshold)
        # current_buy_params_list = joku
        # current_sell_params_list = joku

    return results


def format_params(params, buy):
    return [params['adx_threshold'], params['adx_high_threshold'] if buy else params['adx_low_threshold'],
            params['aroon_up_thresholds'],
            params['aroon_down_thresholds'], params['rsi_threshold']]


# TODO WIP, RUN FIRST 1000/10000 times the algorithm with random numbers.
#  Choose 10/100/100 best results to generate offspring, meaning not a single
#  entry will be same in next the next run. Add some percentage fully random
#  numbers to create genetic diversity, compare results from previous run and if
#  results are worse redo the offspring.
#  Need for some kind of system to evaluate whole round, possibly average of how far
#  results are from desired goal?
#  check your own geneettinen musiikkialgoritmi


def adjust_param_ranges(buy_param_ranges, sell_param_ranges, all_results, victories_threshold,
                        profit_threshold):
    scores = {}
    for i in range(len(all_results)):
        score = fitness_function(all_results[i], victories_threshold, profit_threshold)
        scores[score] = ([all_results[i][4], all_results[i][5]])

    selected_parents = roulette(scores)

    mothers, fathers = selected_parents_to_pairs(selected_parents, list(scores.keys()))

    children = []
    if len(mothers) == len(fathers):
        for i in range(len(mothers)):
            mother = scores[mothers[i]]
            father = scores[fathers[i]]
            children += (crossover(mother, father))

    mutated_children = mutate(children)

    random_gene_children = add_random_gene(mutated_children, buy_param_ranges, sell_param_ranges)
    buy_parameters = []
    sell_parameters = []
    for i in range(len(random_gene_children)):
        buy_parameters.append(random_gene_children[i][0])
        sell_parameters.append(random_gene_children[i][1])
    print(f"UUDET OSTOPARAMETRIT: {buy_parameters}")
    print(f"UUDET MYYNTIPARAMETRIT: {buy_parameters}")
    return buy_parameters, sell_parameters


def roulette(scores):
    scores_array = np.array(list(scores.keys()))
    fitness_cumsum = scores_array.cumsum()
    fitness_sum = fitness_cumsum[-1]
    step = fitness_sum / len(scores)
    rng = default_rng()
    start = rng.random() * step
    selectors = np.arange(start, fitness_sum, step)
    selected = np.searchsorted(fitness_cumsum, selectors)
    return selected


def simulate_trading(stock_indicator_data, buy_condition_params, sell_condition_params, last_close_values):
    expenses = 5
    investment = {indicator['symbol']: 500 for indicator in stock_indicator_data}
    hold_investment = {indicator['symbol']: 500 for indicator in stock_indicator_data}
    stocks = {indicator['symbol']: 0 for indicator in stock_indicator_data}
    hold_stocks = {indicator['symbol']: 0 for indicator in stock_indicator_data}
    prev_command = {indicator['symbol']: 'SELL' for indicator in stock_indicator_data}
    i = {indicator['symbol']: 0 for indicator in stock_indicator_data}
    unique_years = set(indicator['stock__date'].year for indicator in stock_indicator_data)
    hold_dividend_by_stock = {indicator['symbol']: {year: 0 for year in unique_years} for indicator in
                              stock_indicator_data}
    dividend_by_stock = {indicator['symbol']: {year: 0 for year in unique_years} for indicator in stock_indicator_data}
    victories = 0
    losses = 0
    even = 0
    first_buy_sell_date = None

    for indicator in stock_indicator_data:
        if i[indicator['symbol']] >= 12:
            if indicator['stock__date'].month == 3:
                if hold_dividend_by_stock[indicator['symbol']][indicator['stock__date'].year] == 0 and hold_stocks[
                    indicator['symbol']] != 0:
                    hold_dividend_by_stock[indicator['symbol']][
                        indicator['stock__date'].year] = 1  # to rule out not to pay again next day in March
                    dividend = hold_stocks[indicator['symbol']] * indicator['stock__close'] * 0.025
                    if dividend >= expenses:
                        hold_stocks[indicator['symbol']] += (dividend - expenses) / indicator['stock__close']
                if dividend_by_stock[indicator['symbol']][indicator['stock__date'].year] == 0 and prev_command[
                    indicator['symbol']] == 'BUY':
                    dividend_by_stock[indicator['symbol']][indicator['stock__date'].year] = stocks[
                                                                                                indicator['symbol']] * \
                                                                                            indicator[
                                                                                                'stock__close'] * 0.025
            if prev_command[indicator['symbol']] == 'SELL' \
                    and buy_condition_params['adx_threshold'] < indicator['adx'] < buy_condition_params[
                'adx_high_threshold'] \
                    and indicator['rsi14'] < buy_condition_params['rsi_threshold'] \
                    and indicator['aroon_up'] > indicator['aroon_down'] \
                    and indicator['aroon_up'] > buy_condition_params['aroon_up_thresholds'] \
                    and indicator['aroon_down'] < buy_condition_params['aroon_down_thresholds'] \
                    and investment[indicator['symbol']] > 5:
                stocks[indicator['symbol']] = (investment[indicator['symbol']] - expenses) / indicator['stock__close']
                if hold_stocks[indicator['symbol']] == 0:
                    hold_stocks[indicator['symbol']] = (hold_investment[indicator['symbol']] - expenses) / indicator[
                        'stock__close']
                prev_command[indicator['symbol']] = 'BUY'
                if first_buy_sell_date is None:
                    first_buy_sell_date = indicator['stock__date']
            elif (prev_command[indicator['symbol']] == 'BUY' and sell_condition_params['adx_threshold'] > indicator[
                'adx'] >
                  sell_condition_params['adx_low_threshold']
                  and indicator['rsi14'] > sell_condition_params['rsi_threshold']
                  and indicator['aroon_up'] < indicator['aroon_down']
                  and indicator['aroon_up'] < sell_condition_params['aroon_up_thresholds']
                  and indicator['aroon_down'] > sell_condition_params['aroon_down_thresholds']):
                investment[indicator['symbol']] = stocks[indicator['symbol']] * indicator['stock__close'] - 5
                stocks[indicator['symbol']] = 0
                prev_command[indicator['symbol']] = 'SELL'
                for year in range(first_buy_sell_date.year, indicator['stock__date'].year):
                    dividend = dividend_by_stock[indicator['symbol']].pop(year, 0)
                    investment[indicator['symbol']] += dividend
        i[indicator['symbol']] += 1
    for entry in last_close_values:
        if hold_stocks[entry['symbol']] != 0:
            hold_investment[entry['symbol']] = hold_stocks[entry['symbol']] * entry['close']
        if stocks[entry['symbol']] != 0:
            investment[entry['symbol']] = stocks[entry['symbol']] * entry['close']
            for year in range(first_buy_sell_date.year, entry['date'].year):
                dividend = dividend_by_stock[entry['symbol']].pop(year, 0)
                investment[entry['symbol']] += dividend
        if investment[entry['symbol']] > hold_investment[entry['symbol']]:
            victories += 1
        elif investment[entry['symbol']] < hold_investment[entry['symbol']]:
            losses += 1
        else:
            even += 1
    return sum(investment.values()) - (
            500 * len(investment)), victories, losses, even, buy_condition_params, sell_condition_params
