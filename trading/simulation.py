import itertools
import multiprocessing
from multiprocessing import Manager, Semaphore
from trading.models import finnish_stock_daily, signals

MAX_PROCESSES = 32


def simulate_trading(stock_indicator_data, buy_condition_params, sell_condition_params):
    investment = 500
    stocks = 0
    prev_command = 'SELL'
    i = 0
    for indicator in stock_indicator_data:
        if i >= 186:
            # Buy condition using parameters from buy_condition_params
            if prev_command == 'SELL' \
                    and buy_condition_params['adx_threshold'] < indicator['adx'] < buy_condition_params[
                    'adx_high_threshold'] \
                    and indicator['rsi14'] < buy_condition_params['rsi_threshold'] \
                    and indicator['aroon_up'] > indicator['aroon_down'] \
                    and indicator['aroon_up'] > buy_condition_params['aroon_up_thresholds'] \
                    and indicator['aroon_down'] < buy_condition_params['aroon_down_thresholds'] \
                    and investment > 5:
                # and stock_indicator_data[i].stock.close > stock_indicator_data[i].ema20 and stock_indicator_data[
                #     i].stock.close > stock_indicator_data[i].ema50 \
                #     and stock_indicator_data[i].stock.close > stock_indicator_data[i].ema100 and stock_indicator_data[
                #         i].stock.close > stock_indicator_data[i].ema200 \
                # and (indicator.macd > buy_condition_params['macd_thresholds']) \
                # and (indicator.macd < buy_condition_params['macd_thresholds']) \
                # and (indicator.aroon_up > buy_condition_params['aroon_up_thresholds']) \
                # and (indicator.aroon_down > buy_condition_params['aroon_down_thresholds']):
                stocks = (investment - 5) / indicator['stock__close']
                prev_command = 'BUY'

            # Sell condition using parameters from sell_condition_params
            elif (prev_command == 'BUY' and sell_condition_params['adx_threshold'] > indicator['adx'] >
                  sell_condition_params['adx_low_threshold']
                  and indicator['rsi14'] > sell_condition_params['rsi_threshold']
                  and indicator['aroon_up'] < indicator['aroon_down']
                  and indicator['aroon_up'] < sell_condition_params['aroon_up_thresholds']
                  and indicator['aroon_down'] > sell_condition_params['aroon_down_thresholds']
            ) or (
                    i == len(stock_indicator_data) - 1 and stocks != 0):
                # and stock_indicator_data[i].stock.close < stock_indicator_data[i].ema20 and stock_indicator_data[
                #     i].stock.close < stock_indicator_data[i].ema50
                # and stock_indicator_data[i].stock.close < stock_indicator_data[i].ema100 and stock_indicator_data[
                #     i].stock.close < stock_indicator_data[i].ema200
                # and (indicator.macd > sell_condition_params['macd_thresholds']) \
                # and (indicator.macd < sell_condition_params['macd_thresholds']) \
                # and (indicator.aroon_up < sell_condition_params['aroon_up_thresholds']) \
                # and (indicator.aroon_down < sell_condition_params['aroon_down_thresholds']):
                investment = stocks * indicator['stock__close'] - 5
                stocks = 0
                prev_command = 'SELL'
        i += 1
    return investment - 500


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
                                                        'stock__close').order_by('stock__date')
    symbols = finnish_stock_daily.objects.values('symbol').distinct()
    for buy_params in buy_param_combinations:
        sell_param_combinations = itertools.product(*sell_param_ranges.values())
        for sell_params in sell_param_combinations:
            print(f"OSTOPARAMETRIT: {buy_params}, MYYNTIPARAMETRIT: {sell_params}")
            p = multiprocessing.Process(target=worker, args=(semaphore, stock_indicator_data, symbols,
                                                             buy_params, sell_params, best_profit,
                                                             best_buy_params, best_sell_params,
                                                             buy_param_ranges,
                                                             sell_param_ranges))
            processes.append(p)
            p.start()

    for p in processes:
        p.join()

    return best_buy_params[:], best_sell_params[:], best_profit.value


def evaluate_params(stock_indicator_data, symbols, buy_params, sell_params, best_profit, best_buy_params,
                    best_sell_params, buy_param_ranges,
                    sell_param_ranges):
    total_profit = multiprocessing.Value('f', 0.0)

    for stock_symbol in symbols:
        if stock_symbol['symbol'] != 'S&P500':
            filtered_data = stock_indicator_data.filter(stock__symbol=stock_symbol['symbol'])
            evaluate_stock(filtered_data, buy_params, sell_params, total_profit, buy_param_ranges, sell_param_ranges)

    # Update the best parameters if total profit is higher
    print(f"PROFIT: {total_profit}, OSTO: {list(map(float, buy_params))}, MYYNTI: {list(map(float, sell_params))}")
    with best_profit.get_lock():
        if total_profit.value > best_profit.value:
            best_profit.value = total_profit.value
            with best_buy_params.get_lock():
                best_buy_params[:] = list(map(float, buy_params))  # Convert tuple to list of floats
            with best_sell_params.get_lock():
                best_sell_params[:] = list(map(float, sell_params))  # Convert tuple to list of floats


def evaluate_stock(stock_indicator_data, buy_params, sell_params, total_profit, buy_param_ranges, sell_param_ranges):
    profit = simulate_trading(stock_indicator_data, dict(zip(buy_param_ranges.keys(), buy_params)),
                              dict(zip(sell_param_ranges.keys(), sell_params)))
    with total_profit.get_lock():
        total_profit.value += profit


def worker(semaphore, stock_indicator_data, symbols, buy_params, sell_params, best_profit, best_buy_params,
           best_sell_params, buy_param_ranges, sell_param_ranges):
    semaphore.acquire()
    try:
        evaluate_params(stock_indicator_data, symbols, buy_params, sell_params, best_profit, best_buy_params,
                        best_sell_params, buy_param_ranges, sell_param_ranges)
    finally:
        semaphore.release()
