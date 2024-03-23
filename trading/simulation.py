import itertools
import multiprocessing
import django
from django.db import connection
from trading.models import finnish_stock_daily, signals


def simulate_trading(stock_symbol, buy_condition_params, sell_condition_params):
    django.setup()
    connection.close()
    investment = 500
    initial_investment = 500
    stocks = 0
    stock_data = finnish_stock_daily.objects.filter(symbol=stock_symbol).order_by('date')
    prev_command = 'SELL'
    for i in range(len(stock_data)):
        if i >= 203:
            stock_daily = stock_data[i]
            indicator = signals.objects.get(stock=stock_daily)
            close_val = stock_daily.close

            # Buy condition using parameters from buy_condition_params
            if indicator.adx > buy_condition_params['adx_threshold'] \
                    and indicator.rsi14 < buy_condition_params['rsi_threshold'] \
                    and prev_command == 'SELL'\
                    and investment > 5:
                    # and close_val > indicator.ema20 and close_val > indicator.ema50 \
                    # and close_val > indicator.ema100 and close_val > indicator.ema200 \
                    # and (indicator.macd > buy_condition_params['macd_thresholds']) \
                    # and (indicator.macd < buy_condition_params['macd_thresholds']) \
                    # and (indicator.aroon_up > buy_condition_params['aroon_up_thresholds']) \
                    # and (indicator.aroon_down > buy_condition_params['aroon_down_thresholds']):
                stocks = (investment - 5) / close_val
                prev_command = 'BUY'

            # Sell condition using parameters from sell_condition_params
            elif (indicator.adx < sell_condition_params['adx_threshold']
                    and indicator.rsi14 > sell_condition_params['rsi_threshold']
                    and prev_command == 'BUY') or (i == len(stock_data) - 1 and stocks != 0):
                    # and close_val < indicator.ema20 and close_val < indicator.ema50 \
                    # and close_val < indicator.ema100 and close_val < indicator.ema200 \
                    # and (indicator.macd > sell_condition_params['macd_thresholds']) \
                    # and (indicator.macd < sell_condition_params['macd_thresholds']) \
                    # and (indicator.aroon_up < sell_condition_params['aroon_up_thresholds']) \
                    # and (indicator.aroon_down < sell_condition_params['aroon_down_thresholds']):
                investment = stocks * close_val - 5
                stocks = 0
                prev_command = 'SELL'

    return investment - initial_investment


def optimize_parameters(buy_param_ranges, sell_param_ranges):
    best_buy_params = multiprocessing.Array('f', [0.0] * len(buy_param_ranges))
    best_sell_params = multiprocessing.Array('f', [0.0] * len(sell_param_ranges))
    best_profit = multiprocessing.Value('f', float('-inf'))

    # Generate combinations of buy and sell parameters
    buy_param_combinations = itertools.product(*buy_param_ranges.values())

    processes = []

    for buy_params in buy_param_combinations:
        sell_param_combinations = itertools.product(*sell_param_ranges.values())
        for sell_params in sell_param_combinations:
            print(f"OSTOPARAMETRIT: {buy_params}, MYYNTIPARAMETRIT: {sell_params}")
            p = multiprocessing.Process(target=evaluate_params, args=(
                buy_params, sell_params, best_profit, best_buy_params, best_sell_params, buy_param_ranges,
                sell_param_ranges))
            processes.append(p)
            p.start()

    for p in processes:
        p.join()

    return best_buy_params[:], best_sell_params[:], best_profit.value


def evaluate_params(buy_params, sell_params, best_profit, best_buy_params, best_sell_params, buy_param_ranges,
                    sell_param_ranges):
    total_profit = multiprocessing.Value('f', 0.0)

    symbols = finnish_stock_daily.objects.values('symbol').distinct()

    for stock_symbol in symbols:
        if stock_symbol['symbol'] != 'SP&500':
            evaluate_stock(stock_symbol['symbol'], buy_params, sell_params, total_profit, buy_param_ranges, sell_param_ranges)

    # Update the best parameters if total profit is higher
    print(f"PROFIT: {total_profit}, OSTO: {list(map(float, buy_params))}, MYYNTI: {list(map(float, sell_params))}")
    with best_profit.get_lock():
        if total_profit.value > best_profit.value:
            best_profit.value = total_profit.value
            with best_buy_params.get_lock():
                best_buy_params[:] = list(map(float, buy_params))  # Convert tuple to list of floats
            with best_sell_params.get_lock():
                best_sell_params[:] = list(map(float, sell_params))  # Convert tuple to list of floats


def evaluate_stock(stock_symbol, buy_params, sell_params, total_profit, buy_param_ranges, sell_param_ranges):
    profit = simulate_trading(stock_symbol, dict(zip(buy_param_ranges.keys(), buy_params)),
                              dict(zip(sell_param_ranges.keys(), sell_params)))
    with total_profit.get_lock():
        total_profit.value += profit
