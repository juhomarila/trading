from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd

from trading.models import finnish_stock_daily


def visualize_stock_and_investment(stock_symbol, buy_sell_points, initial_investment, expenses, search_start_date,
                                   search_end_date):
    stock_data = finnish_stock_daily.objects.filter(symbol=stock_symbol,
                                                    date__range=(search_start_date, search_end_date)).order_by(
        'date')
    sp_500_benchmark_data = finnish_stock_daily.objects.filter(symbol='S&P500', date__range=(search_start_date, search_end_date)).order_by(
        'date')
    dates = [stock.date for stock in stock_data]
    stock_values = [stock.close for stock in stock_data]
    sp_500_dates = [sp_500.date for sp_500 in sp_500_benchmark_data]
    sp_500_values = [sp_500.close for sp_500 in sp_500_benchmark_data]

    initial_stock_value = stock_values[0]
    initial_sp_500_value = sp_500_values[0]
    initial_investment_value = initial_investment

    last_buy_date = None
    last_sell_date = None
    stocks = 0
    investment = initial_investment
    investments = []
    last_command = 'SELL'
    search_start_date = datetime.strptime(search_start_date, '%Y-%m-%d').date()
    search_end_date = datetime.strptime(search_end_date, '%Y-%m-%d').date()
    buy_sell_dates = []

    for point in buy_sell_points:
        if (point.command == 'BUY' and last_command != 'BUY'
                and search_start_date < point.stock.date < search_end_date):
            queryset = finnish_stock_daily.objects.filter(symbol=stock_symbol, date__gt=point.stock.date).order_by(
                'date')
            if queryset.count() >= 5:
                last_buy_stock = queryset[4]
                last_buy_date = last_buy_stock.date
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
            buy_sell_dates.append((last_buy_date, last_buy_stock.close, 'BUY'))
            last_command = 'BUY'
        elif point.command == 'SELL' and last_command != 'SELL' \
                and search_start_date < point.stock.date < search_end_date:
            queryset = finnish_stock_daily.objects.filter(symbol=stock_symbol, date__gt=point.stock.date).order_by(
                'date')
            if queryset.count() >= 5:
                last_sell_stock = queryset[4]
                last_sell_date = last_sell_stock.date
            else:
                last_sell_stock = None
                last_sell_date = point.stock.date
            investment = (stocks * last_sell_stock.close) - expenses if last_sell_stock is not None \
                else (stocks * point.value) - expenses
            stocks = 0
            investments.append((last_sell_date, investment, 'SELL'))
            last_command = 'SELL'
            buy_sell_dates.append((last_sell_date, last_sell_stock.close, 'SELL'))

    # Calculate percentage changes in stock values
    stock_changes = [(100 * (stock_values[i] - initial_stock_value) / initial_stock_value) for i in
                     range(1, len(stock_values))]
    sp_500_changes = [(100 * (sp_500_values[i] - initial_sp_500_value) / initial_sp_500_value) for i in
                      range(1, len(sp_500_values))]

    investment_changes = []

    # Iterate through the investments
    for i in range(len(investments)):
        date, investment_value_or_stocks, signal = investments[i]

        if signal == 'SELL':
            for x in range(i + 1, len(investments)):
                next_date, _, next_signal = investments[x]
                if next_signal == 'BUY':
                    buy_date = next_date
                    break
            else:  # If no sell date was found
                buy_date = dates[-1]
            date_data = finnish_stock_daily.objects.filter(symbol=stock_symbol,
                                                           date__range=(date, buy_date)).order_by(
                'date')
            if len(date_data) > 0:
                for r in range(len(date_data)):
                    investment_changes.append((date_data[r].date, 100 * (
                            investment_value_or_stocks - initial_investment) / initial_investment))  # Draw horizontal line until the first buy signal
        else:
            # Find the next SELL signal
            for j in range(i + 1, len(investments)):
                next_date, _, next_signal = investments[j]
                if next_signal == 'SELL':
                    sell_date = next_date
                    break
            else:  # If no sell date was found
                sell_date = dates[-1]
            # Calculate the change using the last buy date and sell date
            stock_change_data = finnish_stock_daily.objects.filter(symbol=stock_symbol,
                                                                   date__range=(date, sell_date)).order_by(
                'date')
            if len(stock_change_data) > 0:
                for y in range(len(stock_change_data)):
                    investment_change_percentage = 100 * (investment_value_or_stocks * stock_change_data[y].close
                                                          - initial_investment) / initial_investment
                    investment_changes.append((stock_change_data[y].date, investment_change_percentage))

    # Visualize data
    plt.figure(figsize=(16, 8))
    plt.plot(dates[1:], stock_changes, label=stock_symbol + ' Value Change (%)', color='blue')
    plt.plot(sp_500_dates[1:], sp_500_changes, label='SP&500 Value Change (%)', color='yellow')
    plt.plot(*zip(*investment_changes), linestyle='-', color='red',
             label='Investment Change (%), initial: ' + str(initial_investment) + "€")
    plt.grid(True, linestyle='--', color='gray', alpha=0.5)
    # Adjust the frequency of x-axis gridlines
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.gca().xaxis.set_major_locator(plt.MultipleLocator(100))  # Show gridlines every 7 days

    # Adjust the frequency of y-axis gridlines
    plt.gca().yaxis.set_major_locator(plt.MultipleLocator(25))
    plt.xlabel('Date')
    plt.ylabel('Percentage Change')
    plt.title('Percentage Change in ' + stock_symbol + ' Stock Value and Investment between '
              + str(search_start_date) + ' and ' + str(search_end_date))
    plt.legend()
    plt.show()

    return buy_sell_dates
