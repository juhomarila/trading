import base64
import io
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from trading.models import finnish_stock_daily, optimal_buy_sell_points, signals, reverse_signals


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
    smoothed_rsi14_values = np.convolve(rsi14_values, np.ones(window_size) / window_size, mode='valid')
    smoothed_adx_values = np.convolve(adx_values, np.ones(window_size) / window_size, mode='valid')
    smoothed_reverse_adx_values = np.convolve(filled_reverse_adx_values, np.ones(window_size) / window_size,
                                              mode='valid')

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
    #plt.plot(dates_smoothed, smoothed_rsi14_values, label='RSI14', color='blue')
    #plt.plot(dates_smoothed, smoothed_adx_values, label='ADX', color='black')
    #plt.plot(dates_smoothed, smoothed_reverse_adx_values, label='REVERSE ADX', color='grey')
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
    # if chosen_provider == 'Osuuspankki':  # case osuuspankki
    #     expenses = 5
    # else:  # Case nordnet
    #     expenses = 5
    expenses = 5
    first_buy_sell_stock = optimal_buy_sell_points.objects.filter(symbol__in=chosen_stocks, command='BUY').order_by(
        'stock__date').first().stock
    five_stocks_back = \
        finnish_stock_daily.objects.filter(symbol=first_buy_sell_stock.symbol,
                                           date__lt=first_buy_sell_stock.date).order_by(
            '-date')[4]
    stock_data = finnish_stock_daily.objects.filter(symbol__in=chosen_stocks,
                                                    date__range=(five_stocks_back.date, end_date)).order_by('date')
    sp_500_benchmark_data = finnish_stock_daily.objects.filter(symbol='S&P500', date__range=(
        five_stocks_back.date, end_date)).order_by('date')
    signal_data = optimal_buy_sell_points.objects.filter(symbol__in=chosen_stocks)
    stock_data_list = list(stock_data.values('date', 'symbol', 'close'))  # Extract only necessary fields
    signal_data_list = list(signal_data.values('stock__date', 'symbol', 'command'))

    stock_df = pd.DataFrame(stock_data_list)
    signal_df = pd.DataFrame(signal_data_list)

    stock_df = stock_df.rename(columns={'close': 'price'})
    # TODO ADD TO BOTH stock_df and signal_df data for all dates for all stocks
    signal_df = signal_df.rename(columns={'stock__date': 'date'})

    merged_df = pd.merge(stock_df, signal_df, on=['date', 'symbol'], how='left')

    merged_df['command'] = merged_df.groupby('symbol')['command'].ffill()

    merged_df['command'] = merged_df['command'].fillna('SELL')

    merged_df['date'] = pd.to_datetime(merged_df['date'])
    merged_df['command_shifted'] = merged_df.groupby('symbol')['command'].shift(0)
    merged_df['prev_command'] = merged_df.groupby('symbol')['command_shifted'].shift(1)

    stocks = {stock_symbol: 0 for stock_symbol in chosen_stocks}
    investment_by_stock = {stock_symbol: int(investment) for stock_symbol in chosen_stocks}
    hold_investment_by_stock = {stock_symbol: int(investment) for stock_symbol in chosen_stocks}
    hold_stocks_buy_price = {stock_symbol: 0 for stock_symbol in chosen_stocks}
    hold_stocks_value = 0
    transactions = []
    for index, row in merged_df.iterrows():
        stock_symbol = row['symbol']
        price = row['price']
        command = row['command_shifted']
        prev_command = row['prev_command']
        date = row['date']
        if prev_command == 'NaN':
            investment_by_stock[stock_symbol] = int(investment)
        if command == 'BUY' and (prev_command == 'NaN' or prev_command == 'SELL'):
            stocks[stock_symbol] = (investment_by_stock[stock_symbol] - expenses) / price
            investment_by_stock[stock_symbol] -= expenses
            transactions.append(
                (date, stock_symbol, round(price, 2), command, round(sum(investment_by_stock.values()), 2)))
            if hold_stocks_buy_price[stock_symbol] == 0:
                hold_stocks_buy_price[stock_symbol] = price
        elif command == 'BUY' and prev_command == 'BUY':
            investment_by_stock[stock_symbol] = stocks[stock_symbol] * price
        elif command == 'SELL' and prev_command == 'BUY':
            investment_by_stock[stock_symbol] = stocks[stock_symbol] * price - expenses
            stocks[stock_symbol] = 0
            transactions.append(
                (date, stock_symbol, round(price, 2), command, round(sum(investment_by_stock.values()), 2)))

        merged_df.at[index, 'investment'] = investment_by_stock[stock_symbol]
        merged_df.at[index, 'stocks'] = stocks[stock_symbol]
        prev_investment = merged_df.groupby('symbol')['investment'].shift(1).ffill()
        investment_growth = ((merged_df['investment'] - prev_investment) / prev_investment) * 100
        merged_df['growth'] = investment_growth

        if date == pd.to_datetime(end_date):
            if stock_symbol in hold_stocks_buy_price and hold_stocks_buy_price[stock_symbol] != 0:
                hold_stocks = (hold_investment_by_stock[stock_symbol] - expenses) / hold_stocks_buy_price[stock_symbol]
                hold_investment_by_stock[stock_symbol] = hold_stocks * price
                print(
                    f"OSAKE: {stock_symbol}, OSTOHINTA: {hold_stocks_buy_price[stock_symbol]}, OSAKKEITA: {hold_stocks}, INVESTOINTI: {hold_stocks * price}, NYKYHINTA: {price}")
            if stocks[stock_symbol] != 0:
                investment_by_stock[stock_symbol] = stocks[stock_symbol] * price

    initial_investment_total = int(investment) * len(chosen_stocks)
    combined_investments = merged_df.groupby('date').agg(total_investments=('investment', 'sum')).reset_index()

    print(f"HOLDAUS INVESTOINNIT: {hold_investment_by_stock}")
    print(f"OMAT INVESTOINNIT: {investment_by_stock}")
    print(f"OMIEN INVESTOINTIEN SUMMA: {sum(investment_by_stock.values())}")
    hold_stocks_value = sum(hold_investment_by_stock.values())

    missing_data = []
    for date in combined_investments['date']:
        investments_count = merged_df[merged_df['date'] == date]['investment'].count()
        if investments_count < len(chosen_stocks):
            for stock_symbol in chosen_stocks:
                if merged_df[(merged_df['date'] == date) & (merged_df['symbol'] == stock_symbol)].empty:
                    filtered_df = merged_df[(merged_df['symbol'] == stock_symbol) & (merged_df['date'] < date)]
                    if not filtered_df.empty:
                        previous_investment = filtered_df['investment'].iloc[-1]
                        missing_data.append({'date': date, 'symbol': stock_symbol, 'investment': previous_investment})

    if missing_data:
        missing_df = pd.DataFrame(missing_data)
        merged_df = pd.concat([merged_df, missing_df], ignore_index=True)

    combined_investments = merged_df.groupby('date').agg(total_investments=('investment', 'sum')).reset_index()
    combined_investments = combined_investments.sort_values(by='date')
    combined_investments['growth'] = ((combined_investments[
                                           'total_investments'] - initial_investment_total) / initial_investment_total) * 100

    final_investment_total = combined_investments['total_investments'].iloc[-1]
    investment_growth = ((final_investment_total - initial_investment_total) / initial_investment_total) * 100
    hold_investment_growth = ((hold_stocks_value - initial_investment_total) / initial_investment_total) * 100

    sp_500_dates = [sp_500.date for sp_500 in sp_500_benchmark_data]
    sp_500_values = [sp_500.close for sp_500 in sp_500_benchmark_data]
    initial_sp_500_value = sp_500_values[0]
    sp_500_changes = [(100 * (sp_500_values[i] - initial_sp_500_value) / initial_sp_500_value) for i in
                      range(1, len(sp_500_values))]

    dates = [mdates.date2num(date) for date in sp_500_dates[1:]]

    plt.figure(figsize=(16, 8))
    plt.plot(dates, sp_500_changes, label='SP&500 Value Change (%)', color='yellow')
    plt.plot(combined_investments['date'], combined_investments['growth'],
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

    return transactions, initial_investment_total, round(final_investment_total, 2), round(hold_stocks_value,
                                                                                           2), round(investment_growth,
                                                                                            2), round(hold_investment_growth, 2)
