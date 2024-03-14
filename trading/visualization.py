from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

from trading.models import finnish_stock_daily, optimal_buy_sell_points


def visualize_stock_and_investment(stock_symbol, buy_sell_points, initial_investment, expenses, search_start_date,
                                   search_end_date):
    stock_data = finnish_stock_daily.objects.filter(symbol=stock_symbol,
                                                    date__range=(search_start_date, search_end_date)).order_by('date')
    sp_500_benchmark_data = finnish_stock_daily.objects.filter(symbol='S&P500', date__range=(
        search_start_date, search_end_date)).order_by('date')
    dates = [stock.date for stock in stock_data]
    stock_values = [stock.close for stock in stock_data]
    sp_500_dates = [sp_500.date for sp_500 in sp_500_benchmark_data]
    sp_500_values = [sp_500.close for sp_500 in sp_500_benchmark_data]

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
            if index + 5 < len(stock_keys):
                last_buy_date = stock_keys[index + 5]
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
                 round(stocks * last_buy_stock.close if last_buy_stock is not None else stocks * point.stock.close, 2), 'BUY'))
            last_command = 'BUY'
        elif point.command == 'SELL' and last_command != 'SELL' \
                and search_start_date < point.stock.date < search_end_date:
            if index + 5 < len(stock_keys):
                last_sell_date = stock_keys[index + 5]
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
    plt.plot(sp_500_dates[1:], sp_500_changes, label='SP&500 Value Change (%)', color='yellow')
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
    plt.show()

    return sorted(buy_sell_dates, key=lambda o: o[0], reverse=True)


def create_strategy(investment, start_date, end_date, chosen_stocks, chosen_provider):
    # if chosen_provider == 'Osuuspankki':  # case osuuspankki
    #     expenses = 5
    # else:  # Case nordnet
    #     expenses = 5
    expenses = 5
    stock_data = finnish_stock_daily.objects.filter(symbol__in=chosen_stocks,
                                                    date__range=(start_date, end_date)).order_by('date')
    sp_500_benchmark_data = finnish_stock_daily.objects.filter(symbol='S&P500', date__range=(
        start_date, end_date)).order_by('date')
    signal_data = optimal_buy_sell_points.objects.filter(symbol__in=chosen_stocks)
    stock_data_list = list(stock_data.values('date', 'symbol', 'close'))  # Extract only necessary fields
    signal_data_list = list(signal_data.values('stock__date', 'symbol', 'command'))

    stock_df = pd.DataFrame(stock_data_list)
    signal_df = pd.DataFrame(signal_data_list)

    stock_df = stock_df.rename(columns={'close': 'price'})
    signal_df = signal_df.rename(columns={'stock__date': 'date'})

    merged_df = pd.merge(stock_df, signal_df, on=['date', 'symbol'], how='left')

    merged_df['command'] = merged_df.groupby('symbol')['command'].ffill()

    merged_df['command'] = merged_df['command'].fillna('SELL')

    merged_df['date'] = pd.to_datetime(merged_df['date'])
    merged_df['command_shifted'] = merged_df.groupby('symbol')['command'].shift(5)
    merged_df['prev_command'] = merged_df.groupby('symbol')['command_shifted'].shift(1)

    stocks = {stock_symbol: 0 for stock_symbol in chosen_stocks}
    investment_by_stock = {stock_symbol: int(investment) for stock_symbol in chosen_stocks}
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

    initial_investment_total = int(investment) * len(chosen_stocks)
    combined_investments = merged_df.groupby('date').agg(total_investments=('investment', 'sum')).reset_index()

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
    combined_investments['growth'] = ((combined_investments[
                                           'total_investments'] - initial_investment_total) / initial_investment_total) * 100
    combined_investments = combined_investments.sort_values(by='date')

    final_investment_total = combined_investments['total_investments'].iloc[-1]

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

    return transactions, initial_investment_total, round(final_investment_total, 2)
