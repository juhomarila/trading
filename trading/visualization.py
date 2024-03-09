import matplotlib.pyplot as plt
import pandas as pd

from trading.models import finnish_stock_daily


def visualize_stock_and_investment(stock_symbol, buy_sell_points, initial_investment):
    stock_data = finnish_stock_daily.objects.filter(symbol=stock_symbol).order_by('date')
    dates = [stock.date for stock in stock_data]
    stock_values = [stock.close for stock in stock_data]

    initial_stock_value = stock_values[0]
    initial_investment_value = initial_investment

    last_buy_date = None
    last_sell_date = None
    stocks = 0
    investment = initial_investment
    investments = []
    last_command = 'SELL'

    for point in buy_sell_points:
        if point.command == 'BUY' and last_command != 'BUY':
            queryset = finnish_stock_daily.objects.filter(symbol=stock_symbol, date__gt=point.stock.date).order_by(
                'date')
            if queryset.count() >= 5:
                last_buy_stock = queryset[4]
                last_buy_date = last_buy_stock.date
            else:
                last_buy_stock = None
                last_buy_date = point.date
            if investment > 0:
                stocks = (investment - 3) / last_buy_stock.close if last_buy_stock is not None \
                    else (investment - 3) / point.value
                investments.append((last_buy_date, stocks, 'BUY'))
                investment = 0
            else:
                stocks = (initial_investment - 3) / last_buy_stock.close if last_buy_stock is not None \
                    else (investment - 3) / point.value
                investments.append((last_buy_date, stocks, 'BUY'))
            last_command = 'BUY'
        elif point.command == 'SELL' and last_command != 'SELL':
            queryset = finnish_stock_daily.objects.filter(symbol=stock_symbol, date__gt=point.stock.date).order_by(
                'date')
            if queryset.count() >= 5:
                last_sell_stock = queryset[4]
                last_sell_date = last_sell_stock.date
            else:
                last_sell_stock = None
                last_sell_date = point.date
            investment = (stocks * last_sell_stock.close) - 3 if last_sell_stock is not None \
                else (stocks * point.value) - 3
            stocks = 0
            investments.append((last_sell_date, investment, 'SELL'))
            last_command = 'SELL'

    # Calculate profit/loss based on remaining investment and current stock price
    current_stock_price = finnish_stock_daily.objects.filter(symbol=stock_symbol).latest('date').close
    total_value = investment if investment != 0 else stocks * current_stock_price
    profit_loss = total_value

    # Calculate percentage changes in stock values
    stock_changes = [(100 * (stock_values[i] - initial_stock_value) / initial_stock_value) for i in
                     range(1, len(stock_values))]

    investment_changes = []
    last_buy_date = None
    last_sell_date = None
    last_buy_value = None
    sell_value = 0

    # Iterate through the investments
    for i in range(len(investments)):
        date, investment_value_or_stocks, signal = investments[i]

        if signal == 'SELL':
            last_sell_date = date
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
                    investment_changes.append((date_data[r].date, investment_value_or_stocks - initial_investment))  # Draw horizontal line until the first buy signal
            last_buy_date = None
            sell_value = investment_value_or_stocks
        else:
            last_buy_date = date
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
                                                                   date__range=(last_buy_date, sell_date)).order_by(
                'date')
            if len(stock_change_data) > 0:
                for y in range(len(stock_change_data)):
                    investment_change_percentage = 100 * (investment_value_or_stocks * stock_change_data[y].close
                                                          - initial_investment) / initial_investment
                    investment_changes.append((stock_change_data[y].date, investment_change_percentage))
                # initial_stock_value = stock_values[0]
                # final_stock_value = stock_values[-1]
                #
                # # Calculate the percentage change in stock value
                # stock_change_percentage = (final_stock_value - initial_stock_value) / initial_stock_value * 100
                #
                # # Calculate the percentage change in investment value based on the percentage change in stock value
                # investment_change_percentage = (investment_value_or_stocks * stock_change_percentage) / 100
                #
                # investment_changes.append((date, investment_change_percentage))
    # Visualize data
    plt.figure(figsize=(18, 10))
    plt.plot(dates[1:], stock_changes, label=stock_symbol + ' Value Change (%)', color='blue')
    plt.plot(*zip(*investment_changes), linestyle='-', color='red', label='Investment Change (%), initial: ' + str(initial_investment) + "€")
    # for change in zip(investment_changes):
    #     plt.scatter(change[2], color='red', marker='x', s=100)
    plt.xlabel('Date')
    plt.ylabel('Percentage Change')
    plt.title('Percentage Change in ' + stock_symbol + ' Stock Value and Investment')
    plt.legend()
    plt.show()

