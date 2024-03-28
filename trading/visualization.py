import base64
import io
import timeit
from datetime import datetime, timedelta
import multiprocessing
from multiprocessing import Manager, Semaphore
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import django
from django.db import connection
from collections import defaultdict
import pandas as pd

from trading.models import finnish_stock_daily, optimal_buy_sell_points, signals, reverse_signals

MAX_PROCESSES = multiprocessing.cpu_count()


def visualize_stock_and_investment(stock_symbol, buy_sell_points, initial_investment, expenses, search_start_date,
                                   search_end_date):
    first_buy_date = buy_sell_points.filter(command='BUY').order_by('stock__date').first().stock.date
    print(f"PÄIVÄ: {first_buy_date}")
    stock_data = finnish_stock_daily.objects.filter(symbol=stock_symbol,
                                                    date__range=(first_buy_date, search_end_date)).order_by('date')
    sp_500_benchmark_data = finnish_stock_daily.objects.filter(symbol='S&P500', date__range=(
        first_buy_date, search_end_date)).order_by('date')
    signal_data = signals.objects.filter(symbol=stock_symbol,
                                         stock__date__range=(first_buy_date, search_end_date)).order_by(
        'stock__date')
    reverse_signal_data = reverse_signals.objects.filter(symbol=stock_symbol,
                                                         stock__date__range=(
                                                             first_buy_date, search_end_date)).order_by(
        'stock__date')

    buy_sell_points = buy_sell_points.filter(stock__date__range=(first_buy_date, search_end_date)).order_by(
        'stock__date')
    dates = [stock.date for stock in stock_data]
    stock_values = [stock.close for stock in stock_data]
    ema20_values = [signal.ema20 for signal in signal_data]
    ema50_values = [signal.ema50 for signal in signal_data]
    ema100_values = [signal.ema100 for signal in signal_data]
    ema200_values = [signal.ema200 for signal in signal_data]
    rsi14_values = [signal.rsi14 for signal in signal_data]
    adx_values = [signal.adx for signal in signal_data]
    reverse_adx_values = [reverse_signal.adx for reverse_signal in reverse_signal_data]
    sp_500_dates = [sp_500.date for sp_500 in sp_500_benchmark_data]
    sp_500_values = [sp_500.close for sp_500 in sp_500_benchmark_data]

    filled_reverse_adx_values = []
    previous_value = None

    for value in reverse_adx_values:
        if value is not None:
            filled_reverse_adx_values.append(value)
            previous_value = value
        else:
            filled_reverse_adx_values.append(previous_value)

    window_size = 20  # Adjust the window size as needed for your smoothing preference

    smoothed_ema20_values = np.convolve(ema20_values, np.ones(window_size) / window_size, mode='valid')
    smoothed_ema50_values = np.convolve(ema50_values, np.ones(window_size) / window_size, mode='valid')
    smoothed_ema100_values = np.convolve(ema100_values, np.ones(window_size) / window_size, mode='valid')
    smoothed_ema200_values = np.convolve(ema200_values, np.ones(window_size) / window_size, mode='valid')
    # smoothed_rsi14_values = np.convolve(rsi14_values, np.ones(window_size) / window_size, mode='valid')
    # smoothed_adx_values = np.convolve(adx_values, np.ones(window_size) / window_size, mode='valid')
    # smoothed_reverse_adx_values = np.convolve(filled_reverse_adx_values, np.ones(window_size) / window_size,
    #                                          mode='valid')

    dates_smoothed = dates[window_size - 1:]

    initial_stock_value = stock_values[0]
    initial_sp_500_value = sp_500_values[0]

    stocks = 0
    investment = initial_investment
    investments = []
    last_command = 'SELL'
    search_start_date = datetime.strptime(search_start_date, '%Y-%m-%d').date()
    search_end_date = datetime.strptime(search_end_date, '%Y-%m-%d').date()
    buy_sell_dates = []

    stock_data_dict = {stock.date: stock for stock in stock_data}

    for point in buy_sell_points:
        stock_keys = sorted(stock_data_dict.keys())
        index = stock_keys.index(point.stock.date)
        if (point.command == 'BUY' and last_command != 'BUY'
                and search_start_date < point.stock.date < search_end_date):
            if index + 0 < len(stock_keys):
                last_buy_date = stock_keys[index + 0]
                last_buy_stock = stock_data_dict[last_buy_date]
            else:
                last_buy_stock = None
                last_buy_date = point.stock.date
            if investment > 0:
                stocks = (investment - expenses) / last_buy_stock.close if last_buy_stock is not None \
                    else (investment - expenses) / point.value
                investments.append((last_buy_date, stocks, 'BUY'))
                investment = 0
            else:
                stocks = (initial_investment - expenses) / last_buy_stock.close if last_buy_stock is not None \
                    else (investment - expenses) / point.value
                investments.append((last_buy_date, stocks, 'BUY'))
            buy_sell_dates.append(
                (last_buy_date,
                 round(last_buy_stock.close, 2) if last_buy_stock is not None else round(point.stock.close, 2),
                 round(stocks * last_buy_stock.close if last_buy_stock is not None else stocks * point.stock.close, 2),
                 'BUY'))
            last_command = 'BUY'
        elif point.command == 'SELL' and last_command != 'SELL' \
                and search_start_date < point.stock.date < search_end_date:
            if index + 0 < len(stock_keys):
                last_sell_date = stock_keys[index + 0]
                last_sell_stock = stock_data_dict[last_sell_date]
            else:
                last_sell_stock = None
                last_sell_date = point.stock.date
            investment = (stocks * last_sell_stock.close) - expenses if last_sell_stock is not None \
                else (stocks * point.value) - expenses
            stocks = 0
            investments.append((last_sell_date, investment, 'SELL'))
            last_command = 'SELL'
            buy_sell_dates.append(
                (last_sell_date,
                 round(last_sell_stock.close, 2) if last_sell_stock is not None else round(point.stock.close, 2),
                 round(investment, 2), 'SELL'))

    stock_changes = [(100 * (stock_values[i] - initial_stock_value) / initial_stock_value) for i in
                     range(1, len(stock_values))]
    sp_500_changes = [(100 * (sp_500_values[i] - initial_sp_500_value) / initial_sp_500_value) for i in
                      range(1, len(sp_500_values))]

    investment_changes = []

    for i in range(len(investments)):
        date, investment_value_or_stocks, signal = investments[i]

        if signal == 'SELL':
            for x in range(i + 1, len(investments)):
                next_date, _, next_signal = investments[x]
                if next_signal == 'BUY':
                    buy_date = next_date
                    break
            else:
                buy_date = dates[-1]
            date_data = []
            for date_key in stock_data_dict.keys():
                if date <= date_key <= buy_date:
                    date_data.append(stock_data_dict[date_key])
            date_data.sort(key=lambda o: o.date)
            if len(date_data) > 0:
                for r in range(len(date_data)):
                    investment_changes.append((date_data[r].date, 100 * (
                            investment_value_or_stocks - initial_investment) / initial_investment))
        else:
            for j in range(i + 1, len(investments)):
                next_date, _, next_signal = investments[j]
                if next_signal == 'SELL':
                    sell_date = next_date
                    break
            else:
                sell_date = dates[-1]
            stock_change_data = []
            for date_key in stock_data_dict.keys():
                if date <= date_key <= sell_date:
                    stock_change_data.append(stock_data_dict[date_key])
            stock_change_data.sort(key=lambda o: o.date)
            if len(stock_change_data) > 0:
                for y in range(len(stock_change_data)):
                    investment_change_percentage = 100 * (investment_value_or_stocks * stock_change_data[y].close
                                                          - initial_investment) / initial_investment
                    investment_changes.append((stock_change_data[y].date, investment_change_percentage))

    plt.figure(figsize=(16, 8))
    plt.plot(dates[1:], stock_changes, label=stock_symbol + ' Value Change (%)', color='blue')
    plt.plot(sp_500_dates[1:], sp_500_changes, label='SP&500 Value Change (%)', color='yellow', alpha=0.6)
    plt.plot(*zip(*investment_changes), linestyle='-', color='red',
             label='Investment Change (%), initial: ' + str(initial_investment) + "€")
    plt.grid(True, linestyle='--', color='gray', alpha=0.5)
    plt.xticks(rotation=45)
    plt.gca().xaxis.set_major_locator(plt.MultipleLocator(100))
    plt.gca().yaxis.set_major_locator(plt.MultipleLocator(25))
    plt.xlabel('Date')
    plt.ylabel('Percentage Change')
    plt.title('Percentage Change in ' + stock_symbol + ' Stock Value and Investment between '
              + str(search_start_date) + ' and ' + str(search_end_date))
    plt.legend()
    buffer1 = io.BytesIO()
    plt.savefig(buffer1, format='png')
    buffer1.seek(0)

    plt.figure(figsize=(16, 8))
    plt.bar(dates[1:], stock_values[1:], label=stock_symbol + ' Value (€)', color='green', alpha=0.3, width=1)
    plt.plot(dates_smoothed, smoothed_ema20_values, label='EMA20', color='purple')
    plt.plot(dates_smoothed, smoothed_ema50_values, label='EMA50', color='cyan')
    plt.plot(dates_smoothed, smoothed_ema100_values, label='EMA100', color='magenta')
    plt.plot(dates_smoothed, smoothed_ema200_values, label='EMA200', color='orange')
    # plt.plot(dates_smoothed, smoothed_rsi14_values, label='RSI14', color='blue')
    # plt.plot(dates_smoothed, smoothed_adx_values, label='ADX', color='black')
    # plt.plot(dates_smoothed, smoothed_reverse_adx_values, label='REVERSE ADX', color='grey')
    plt.grid(True, linestyle='--', color='gray', alpha=0.5)
    plt.xticks(rotation=45)
    plt.gca().xaxis.set_major_locator(plt.MultipleLocator(100))
    plt.gca().yaxis.set_major_locator(plt.MultipleLocator(25))
    plt.xlabel('Date')
    plt.ylabel('Value (€)')
    plt.title('Technical indicators for ' + stock_symbol + ' between '
              + str(search_start_date) + ' and ' + str(search_end_date))
    plt.legend()

    buffer2 = io.BytesIO()
    plt.savefig(buffer2, format='png')
    buffer2.seek(0)

    plt.close('all')

    plot1 = base64.b64encode(buffer1.read()).decode('utf-8')
    plot2 = base64.b64encode(buffer2.read()).decode('utf-8')

    return plot1, plot2, sorted(buy_sell_dates, key=lambda o: o[0], reverse=True)


def create_strategy(investment, start_date, end_date, chosen_stocks, chosen_provider):
    expenses = 5
    first_buy_sell_stock = optimal_buy_sell_points.objects.filter(symbol__in=chosen_stocks, command='BUY').order_by(
        'stock__date').first().stock
    five_stocks_back = \
        finnish_stock_daily.objects.filter(symbol=first_buy_sell_stock.symbol,
                                           date__lt=first_buy_sell_stock.date).order_by(
            '-date')[4]
    stock_data = finnish_stock_daily.objects.filter(symbol__in=chosen_stocks,
                                                    date__range=(five_stocks_back.date, end_date)) \
        .order_by('date') \
        .values('id', 'symbol', 'date', 'close')
    sp_500_benchmark_data = finnish_stock_daily.objects.filter(symbol='S&P500', date__range=(
        five_stocks_back.date, end_date)).order_by('date')
    optimal_buy_sell_points_data = optimal_buy_sell_points.objects.filter(symbol__in=chosen_stocks).values('stock_id',
                                                                                                           'command',
                                                                                                           'symbol')

    stock_data_list = list(stock_data.values('date', 'symbol', 'close'))  # Extract only necessary fields
    signal_data_list = list(optimal_buy_sell_points_data.values('stock__date', 'symbol', 'command'))
    stock_df = pd.DataFrame(stock_data_list)

    stock_df = stock_df.rename(columns={'close': 'price'})

    unique_dates = stock_df['date'].unique()
    all_symbols = stock_df['symbol'].unique()
    all_combinations = [(date, symbol) for date in unique_dates for symbol in all_symbols]
    all_combinations_df = pd.DataFrame(all_combinations, columns=['date', 'symbol'])

    merged_stock_df = pd.merge(all_combinations_df, stock_df, on=['date', 'symbol'], how='left')

    merged_stock_df['price'] = merged_stock_df.groupby('symbol')['price'].ffill()

    signal_df = pd.DataFrame(signal_data_list)
    signal_df = signal_df.rename(columns={'stock__date': 'date'})
    merged_df = pd.merge(merged_stock_df, signal_df, on=['date', 'symbol'], how='left')

    merged_df['command'] = merged_df.groupby('symbol')['command'].ffill()

    merged_df['command'] = merged_df['command'].fillna('SELL')

    merged_df['date'] = pd.to_datetime(merged_df['date'])
    merged_df['prev_command'] = merged_df.groupby('symbol')['command'].shift(1)
    merged_df['prev_command'] = merged_df['prev_command'].fillna('SELL')

    investment_by_stock = {stock_symbol: int(investment) for stock_symbol in chosen_stocks}
    stocks = {stock_symbol: 0 for stock_symbol in chosen_stocks}
    hold_investment_by_stock = {stock_symbol: int(investment) for stock_symbol in chosen_stocks}
    hold_stocks_buy_price = {stock_symbol: 0 for stock_symbol in chosen_stocks}
    hold_stocks_by_stock = {stock_symbol: 0 for stock_symbol in chosen_stocks}
    unique_years = set(stock['date'].year for stock in stock_data)
    hold_dividend_by_stock = {stock['symbol']: {year: 0 for year in unique_years} for stock in stock_data}
    dividend_by_stock = {stock['symbol']: {year: 0 for year in unique_years} for stock in stock_data}

    transactions = []

    merged_df[['stocks', 'investment']] = merged_df.apply(
        lambda row: calculate_stocks_and_investment(row, int(expenses), len(chosen_stocks) - 1,
                                                    investment_by_stock, stocks, hold_investment_by_stock,
                                                    hold_stocks_buy_price, end_date, transactions, dividend_by_stock,
                                                    first_buy_sell_stock, hold_stocks_by_stock, hold_dividend_by_stock),
        axis=1)

    investments_by_date = merged_df.groupby('date')['investment'].sum().reset_index()
    investments_by_date.iloc[0, 1] = investments_by_date.iloc[1, 1]
    results = []

    for stock_symbol in chosen_stocks:
        orig_diff = investment_by_stock[stock_symbol] - int(investment)
        orig_percentage_diff = (orig_diff / int(investment)) * 100
        if investment_by_stock[stock_symbol] > hold_investment_by_stock[stock_symbol]:
            investment_diff = investment_by_stock[stock_symbol] - hold_investment_by_stock[stock_symbol]
            percentage_diff = (investment_diff / hold_investment_by_stock[stock_symbol]) * 100
            results.append((stock_symbol, round(percentage_diff, 2), 'green', round(orig_percentage_diff, 2),
                            'green' if orig_percentage_diff > 0 else 'red'))
        elif investment_by_stock[stock_symbol] < hold_investment_by_stock[stock_symbol]:
            investment_diff = hold_investment_by_stock[stock_symbol] - investment_by_stock[stock_symbol]
            percentage_diff = (investment_diff / investment_by_stock[stock_symbol]) * 100
            results.append((stock_symbol, round(percentage_diff, 2), 'red', round(orig_percentage_diff, 2),
                            'green' if orig_percentage_diff > 0 else 'red'))
    pd.set_option('display.max_rows', None)

    sp_500_dates = [sp_500.date for sp_500 in sp_500_benchmark_data]
    sp_500_values = [sp_500.close for sp_500 in sp_500_benchmark_data]
    initial_sp_500_value = sp_500_values[0]
    sp_500_changes = [(100 * (sp_500_values[i] - initial_sp_500_value) / initial_sp_500_value) for i in
                      range(1, len(sp_500_values))]
    dates = [mdates.date2num(date) for date in sp_500_dates[1:]]

    initial_total_investment = int(investment) * len(chosen_stocks)
    total_investment_values = investments_by_date['investment']
    total_investment_change = [
        (100 * (total_investment_values[i] - initial_total_investment) / initial_total_investment) for i in
        range(0, len(total_investment_values))]

    initial_investment_total = int(investment) * len(chosen_stocks)
    investment_growth = ((
                                 sum(investment_by_stock.values()) - initial_investment_total) / initial_investment_total) * 100
    hold_investment_growth = ((
                                      sum(hold_investment_by_stock.values()) - initial_investment_total) / initial_investment_total) * 100

    plt.figure(figsize=(16, 8))
    plt.plot(dates, sp_500_changes, label='SP&500 Value Change (%)', color='yellow')
    plt.plot(investments_by_date['date'], total_investment_change,
             label='Combined Investments Value Change (%)', color='blue')
    plt.grid(True, linestyle='--', color='gray', alpha=0.5)
    plt.xticks(rotation=45)
    plt.gca().xaxis.set_major_locator(plt.MultipleLocator(100))
    plt.gca().yaxis.set_major_locator(plt.MultipleLocator(25))
    plt.xlabel('Date')
    plt.ylabel('Percentage Change')
    plt.title('Percentage Change in Stock Value and Investment')
    plt.legend()
    plt.show()

    return transactions, initial_total_investment, round(sum(investment_by_stock.values()), 2), round(
        sum(hold_investment_by_stock.values()), 2), round(investment_growth, 2), round(hold_investment_growth,
                                                                                       2), results


def calculate_stocks_and_investment(row, expenses, length, investment_by_stock, stocks_by_stock,
                                    hold_investment_by_stock, hold_stocks_buy_price, end_date, transactions,
                                    dividend_by_stock, first_buy_sell_stock, hold_stocks_by_stock,
                                    hold_dividend_by_stock):
    stocks = None
    investment = None
    if row.name > length:
        if row['date'].month == 3:
            if hold_dividend_by_stock[row['symbol']][row['date'].year] == 0 and hold_stocks_by_stock[
                row['symbol']] != 0:
                hold_dividend_by_stock[row['symbol']][
                    row['date'].year] = 1  # to rule out not to pay again next day in March
                dividend = hold_stocks_by_stock[row['symbol']] * row['price'] * 0.025
                if dividend >= expenses:
                    hold_stocks_by_stock[row['symbol']] += (dividend - expenses) / row['price']
            if dividend_by_stock[row['symbol']][row['date'].year] == 0 and row['prev_command'] == 'BUY':
                print(
                    f"INVESTMENT NOW: {investment_by_stock[row['symbol']] * 1.025}, DIVIDEND: {investment_by_stock[row['symbol']] * 0.025}, DATE: {row['date']}, STOCK: {row['symbol']}")
                dividend_by_stock[row['symbol']][row['date'].year] = stocks_by_stock[row['symbol']] * row[
                    'price'] * 0.025
        if row['command'] == 'BUY' and row['prev_command'] == 'SELL':
            stocks = (investment_by_stock[row['symbol']] - expenses) / row['price']
            stocks_by_stock[row['symbol']] = stocks
            investment = investment_by_stock[row['symbol']] - expenses
            investment_by_stock[row['symbol']] = investment
            transactions.append(
                (row['date'], row['symbol'], round(row['price'], 2), row['command'],
                 round(sum(investment_by_stock.values()), 2)))
            if hold_stocks_buy_price[row['symbol']] == 0:
                hold_stocks_buy_price[row['symbol']] = row['price']
                hold_stocks_by_stock[row['symbol']] = (hold_investment_by_stock[row['symbol']] - expenses) / row[
                    'price']
        elif row['command'] == 'BUY' and row['prev_command'] == 'BUY':
            stocks = stocks_by_stock[row['symbol']]
            investment = stocks * row['price']
            investment_by_stock[row['symbol']] = investment
        elif row['command'] == 'SELL' and row['prev_command'] == 'BUY':
            investment = stocks_by_stock[row['symbol']] * row['price'] - expenses
            investment_by_stock[row['symbol']] = investment
            stocks = 0
            stocks_by_stock[row['symbol']] = stocks
            transactions.append(
                (row['date'], row['symbol'], round(row['price'], 2), row['command'],
                 round(sum(investment_by_stock.values()), 2)))
            for year in range(first_buy_sell_stock.date.year, row['date'].year):
                dividend = dividend_by_stock[row['symbol']].pop(year, 0)
                investment_by_stock[row['symbol']] += dividend
                investment = investment_by_stock[row['symbol']]
        elif row['command'] == 'SELL' and row['prev_command'] == 'SELL':
            stocks = 0
            investment = investment_by_stock[row['symbol']]
        if row['date'] == pd.to_datetime(end_date):
            if row['symbol'] in hold_stocks_by_stock and hold_stocks_by_stock[row['symbol']] != 0:
                hold_investment_by_stock[row['symbol']] = hold_stocks_by_stock[row['symbol']] * row['price']
            if stocks_by_stock[row['symbol']] != 0:
                investment_by_stock[row['symbol']] = stocks_by_stock[row['symbol']] * row['price']
                investment = investment_by_stock[row['symbol']]
                for year in range(first_buy_sell_stock.date.year, row['date'].year):
                    dividend = dividend_by_stock[row['symbol']].pop(year, 0)
                    investment_by_stock[row['symbol']] += dividend
                    investment = investment_by_stock[row['symbol']]
    return pd.Series({'stocks': stocks, 'investment': investment})
