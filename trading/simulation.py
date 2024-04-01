import itertools
import multiprocessing
from multiprocessing import Manager, Semaphore
import django
from django.db import connection

from django.db.models import Max

from trading.models import finnish_stock_daily, signals

MAX_PROCESSES = 8


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
                    dividend_by_stock[indicator['symbol']][indicator['stock__date'].year] = stocks[indicator['symbol']] * \
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
    return sum(investment.values()) - (500 * len(investment)), victories, losses, even


def optimize_parameters(buy_param_ranges, sell_param_ranges, stock_indicator_data, max_date, last_close_values):
    best_buy_params = multiprocessing.Array('f', [0.0] * len(buy_param_ranges))
    best_sell_params = multiprocessing.Array('f', [0.0] * len(sell_param_ranges))
    best_victories_buy_params = multiprocessing.Array('f', [0.0] * len(buy_param_ranges))
    best_victories_sell_params = multiprocessing.Array('f', [0.0] * len(sell_param_ranges))
    best_profit = multiprocessing.Value('f', float('-inf'))
    best_profit_victories = multiprocessing.Value('f', float('-inf'))
    best_victories = multiprocessing.Value('f', float('-inf'))
    best_victories_profit = multiprocessing.Value('f', float('-inf'))
    best_victories_loss = multiprocessing.Value('f', float('-inf'))
    best_victories_even = multiprocessing.Value('f', float('-inf'))

    # Generate combinations of buy and sell parameters
    buy_param_combinations = itertools.product(*buy_param_ranges.values())

    manager = Manager()
    semaphore = Semaphore(MAX_PROCESSES)

    processes = []
    # stock_indicator_data = signals.objects.all().values('symbol', 'adx', 'rsi14', 'aroon_up', 'aroon_down',
    #                                                     'stock__close').order_by('stock__date')
    #
    # max_date = finnish_stock_daily.objects.exclude(symbol='S&P500').aggregate(max_date=Max('date'))['max_date']
    # last_close_values = finnish_stock_daily.objects.filter(date__range=(max_date, max_date)).exclude(
    #     symbol='S&P500').values('symbol',
    #                             'close',
    #                             'date')
    for buy_params in buy_param_combinations:
        sell_param_combinations = itertools.product(*sell_param_ranges.values())
        for sell_params in sell_param_combinations:
            p = multiprocessing.Process(target=worker, args=(semaphore, stock_indicator_data,
                                                             buy_params, sell_params, best_profit,
                                                             best_buy_params, best_sell_params,
                                                             buy_param_ranges,
                                                             sell_param_ranges, last_close_values,
                                                             best_victories, best_victories_buy_params,
                                                             best_victories_sell_params, best_profit_victories,
                                                             best_victories_profit, best_victories_loss,
                                                             best_victories_even,))
            processes.append(p)
            p.start()

    for p in processes:
        p.join()

    return (best_buy_params[:], best_sell_params[:], best_profit.value, best_victories_buy_params[:],
            best_victories_sell_params[:],
            (best_victories.value / (best_victories.value + best_victories_loss.value + best_victories_even.value)),
            best_profit_victories.value,
            best_victories_profit.value)


def evaluate_params(stock_indicator_data, buy_params, sell_params, best_profit, best_buy_params,
                    best_sell_params, buy_param_ranges, sell_param_ranges, last_close_values,
                    best_victories, best_victories_buy_params, best_victories_sell_params,
                    best_profit_victories, best_victories_profit, best_victories_loss, best_victories_even):
    total_profit = multiprocessing.Value('f', 0.0)
    total_victories = multiprocessing.Value('f', 0.0)
    total_losses = multiprocessing.Value('f', 0.0)
    total_even = multiprocessing.Value('f', 0.0)

    evaluate_stock(stock_indicator_data, buy_params, sell_params, total_profit, buy_param_ranges, sell_param_ranges,
                   last_close_values, total_victories, total_losses, total_even)

    # Update the best parameters if total profit is higher
    print(
        f"PROFIT: {total_profit.value}, OSTO: {list(map(float, buy_params))}, MYYNTI: {list(map(float, sell_params))} VOITTOPROSENTTI: {round((total_victories.value / (total_victories.value + total_losses.value + total_even.value)) * 100, 2)}%")
    print(f"VOITTOJA: {total_victories.value}, TASAN: {total_even.value}, HÄVIÖITÄ: {total_losses.value}")
    with best_profit.get_lock():
        if total_profit.value > best_profit.value:
            best_profit.value = total_profit.value
            best_profit_victories.value = total_victories.value / (
                    total_victories.value + total_losses.value + total_even.value)
            with best_buy_params.get_lock():
                best_buy_params[:] = list(map(float, buy_params))  # Convert tuple to list of floats
            with best_sell_params.get_lock():
                best_sell_params[:] = list(map(float, sell_params))  # Convert tuple to list of floats
    with best_victories.get_lock():
        if (total_victories.value > best_victories.value and total_profit.value > 80000) or (
                (total_victories.value == best_victories.value)
                and (total_profit.value > best_victories_profit.value) and total_profit.value > 80000):
            best_victories.value = total_victories.value
            best_victories_profit.value = total_profit.value
            best_victories_loss.value = total_losses.value
            best_victories_even.value = total_even.value
            with best_victories_buy_params.get_lock():
                best_victories_buy_params[:] = list(map(float, buy_params))  # Convert tuple to list of floats
            with best_victories_sell_params.get_lock():
                best_victories_sell_params[:] = list(
                    map(float, sell_params))  # Convert tuple to list of floats


def evaluate_stock(stock_indicator_data, buy_params, sell_params, total_profit, buy_param_ranges, sell_param_ranges,
                   last_close_values, total_victories, total_losses, total_even):
    profit, victories, losses, even = simulate_trading(stock_indicator_data,
                                                       dict(zip(buy_param_ranges.keys(), buy_params)),
                                                       dict(zip(sell_param_ranges.keys(), sell_params)),
                                                       last_close_values)
    with total_profit.get_lock():
        total_profit.value += profit
    with total_victories.get_lock():
        total_victories.value += victories
    with total_losses.get_lock():
        total_losses.value += losses
    with total_even.get_lock():
        total_even.value += even


def worker(semaphore, stock_indicator_data, buy_params, sell_params, best_profit, best_buy_params,
           best_sell_params, buy_param_ranges, sell_param_ranges, last_close_values, best_victories,
           best_victories_buy_params, best_victories_sell_params, best_profit_victories, best_victories_profit,
           best_victories_loss, best_victories_even):
    django.setup()
    connection.close()
    semaphore.acquire()
    try:
        evaluate_params(stock_indicator_data, buy_params, sell_params, best_profit, best_buy_params,
                        best_sell_params, buy_param_ranges, sell_param_ranges, last_close_values, best_victories,
                        best_victories_buy_params, best_victories_sell_params, best_profit_victories,
                        best_victories_profit, best_victories_loss, best_victories_even)
    finally:
        semaphore.release()
