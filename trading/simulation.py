import itertools
import multiprocessing
from multiprocessing import Manager, Semaphore

from django.db.models import Max

from trading.models import finnish_stock_daily, signals

MAX_PROCESSES = 8


def simulate_trading(stock_indicator_data, buy_condition_params, sell_condition_params, last_close_values):
    investment = {indicator['stock__symbol']: 500 for indicator in stock_indicator_data}
    stocks = {indicator['stock__symbol']: 0 for indicator in stock_indicator_data}
    prev_command = {indicator['stock__symbol']: 'SELL' for indicator in stock_indicator_data}
    i = {indicator['stock__symbol']: 0 for indicator in stock_indicator_data}
    for indicator in stock_indicator_data:
        if i[indicator['stock__symbol']] >= 12:
            # Buy condition using parameters from buy_condition_params
            if prev_command[indicator['stock__symbol']] == 'SELL' \
                    and buy_condition_params['adx_threshold'] < indicator['adx'] < buy_condition_params[
                'adx_high_threshold'] \
                    and indicator['rsi14'] < buy_condition_params['rsi_threshold'] \
                    and indicator['aroon_up'] > indicator['aroon_down'] \
                    and indicator['aroon_up'] > buy_condition_params['aroon_up_thresholds'] \
                    and indicator['aroon_down'] < buy_condition_params['aroon_down_thresholds'] \
                    and investment[indicator['stock__symbol']] > 5:
                # and stock_indicator_data[i].stock.close > stock_indicator_data[i].ema20 and stock_indicator_data[
                #     i].stock.close > stock_indicator_data[i].ema50 \
                #     and stock_indicator_data[i].stock.close > stock_indicator_data[i].ema100 and stock_indicator_data[
                #         i].stock.close > stock_indicator_data[i].ema200 \
                # and (indicator.macd > buy_condition_params['macd_thresholds']) \
                # and (indicator.macd < buy_condition_params['macd_thresholds']) \
                # and (indicator.aroon_up > buy_condition_params['aroon_up_thresholds']) \
                # and (indicator.aroon_down > buy_condition_params['aroon_down_thresholds']):
                stocks[indicator['stock__symbol']] = (investment[indicator['stock__symbol']] - 5) / indicator['stock__close']
                prev_command[indicator['stock__symbol']] = 'BUY'

            # Sell condition using parameters from sell_condition_params
            elif (prev_command[indicator['stock__symbol']] == 'BUY' and sell_condition_params['adx_threshold'] > indicator[
                'adx'] >
                  sell_condition_params['adx_low_threshold']
                  and indicator['rsi14'] > sell_condition_params['rsi_threshold']
                  and indicator['aroon_up'] < indicator['aroon_down']
                  and indicator['aroon_up'] < sell_condition_params['aroon_up_thresholds']
                  and indicator['aroon_down'] > sell_condition_params['aroon_down_thresholds']):
                # or
                # indicator == last_indicator[indicator['symbol']] and stocks[
                #     indicator['symbol']] != 0):
                # and stock_indicator_data[i].stock.close < stock_indicator_data[i].ema20 and stock_indicator_data[
                #     i].stock.close < stock_indicator_data[i].ema50
                # and stock_indicator_data[i].stock.close < stock_indicator_data[i].ema100 and stock_indicator_data[
                #     i].stock.close < stock_indicator_data[i].ema200
                # and (indicator.macd > sell_condition_params['macd_thresholds']) \
                # and (indicator.macd < sell_condition_params['macd_thresholds']) \
                # and (indicator.aroon_up < sell_condition_params['aroon_up_thresholds']) \
                # and (indicator.aroon_down < sell_condition_params['aroon_down_thresholds']):
                investment[indicator['stock__symbol']] = stocks[indicator['stock__symbol']] * indicator['stock__close'] - 5
                stocks[indicator['stock__symbol']] = 0
                prev_command[indicator['stock__symbol']] = 'SELL'
        i[indicator['stock__symbol']] += 1

    for entry in last_close_values:
        if stocks[entry['symbol']] != 0:
            investment[entry['symbol']] = stocks[entry['symbol']] * entry['close'] - 5

    return sum(investment.values()) - (500 * len(investment))


def optimize_parameters(buy_param_ranges, sell_param_ranges):
    best_buy_params = multiprocessing.Array('f', [0.0] * len(buy_param_ranges))
    best_sell_params = multiprocessing.Array('f', [0.0] * len(sell_param_ranges))
    best_profit = multiprocessing.Value('f', float('-inf'))

    # Generate combinations of buy and sell parameters
    buy_param_combinations = itertools.product(*buy_param_ranges.values())

    manager = Manager()
    semaphore = Semaphore(MAX_PROCESSES)

    processes = []
    stock_indicator_data = signals.objects.all().values('adx', 'rsi14', 'aroon_up', 'aroon_down',
                                                        'stock__close', 'stock__symbol', 'stock__date').order_by('stock__date')

    last_close_values = finnish_stock_daily.objects.exclude(symbol='S&P500').values('symbol').order_by('date').annotate(
        last_close=Max('date')).values('symbol',
                                       'close')
    for buy_params in buy_param_combinations:
        sell_param_combinations = itertools.product(*sell_param_ranges.values())
        for sell_params in sell_param_combinations:
            print(f"OSTOPARAMETRIT: {buy_params}, MYYNTIPARAMETRIT: {sell_params}")
            p = multiprocessing.Process(target=worker, args=(semaphore, stock_indicator_data,
                                                             buy_params, sell_params, best_profit,
                                                             best_buy_params, best_sell_params,
                                                             buy_param_ranges,
                                                             sell_param_ranges, last_close_values,))
            processes.append(p)
            p.start()

    for p in processes:
        p.join()

    return best_buy_params[:], best_sell_params[:], best_profit.value


def evaluate_params(stock_indicator_data, buy_params, sell_params, best_profit, best_buy_params,
                    best_sell_params, buy_param_ranges,
                    sell_param_ranges, last_close_values):
    total_profit = multiprocessing.Value('f', 0.0)

    evaluate_stock(stock_indicator_data, buy_params, sell_params, total_profit, buy_param_ranges, sell_param_ranges,
                   last_close_values)

    # Update the best parameters if total profit is higher
    print(f"PROFIT: {total_profit}, OSTO: {list(map(float, buy_params))}, MYYNTI: {list(map(float, sell_params))}")
    with best_profit.get_lock():
        if total_profit.value > best_profit.value:
            best_profit.value = total_profit.value
            with best_buy_params.get_lock():
                best_buy_params[:] = list(map(float, buy_params))  # Convert tuple to list of floats
            with best_sell_params.get_lock():
                best_sell_params[:] = list(map(float, sell_params))  # Convert tuple to list of floats


def evaluate_stock(stock_indicator_data, buy_params, sell_params, total_profit, buy_param_ranges, sell_param_ranges,
                   last_close_values):
    profit = simulate_trading(stock_indicator_data, dict(zip(buy_param_ranges.keys(), buy_params)),
                              dict(zip(sell_param_ranges.keys(), sell_params)), last_close_values)
    with total_profit.get_lock():
        total_profit.value += profit


def worker(semaphore, stock_indicator_data, buy_params, sell_params, best_profit, best_buy_params,
           best_sell_params, buy_param_ranges, sell_param_ranges, last_close_values):
    semaphore.acquire()
    try:
        evaluate_params(stock_indicator_data, buy_params, sell_params, best_profit, best_buy_params,
                        best_sell_params, buy_param_ranges, sell_param_ranges, last_close_values)
    finally:
        semaphore.release()
